"""Read and write MLIR for the supported elementwise arithmetic graphs.

MLIR is a compiler framework that represents calculations as typed operations.
A dialect is a family of operations: ``func`` supplies functions and returns, and
``arith`` supplies arithmetic such as floating-point addition. IREE packages the
native MLIR Python bindings used here to construct, parse, and verify them.

This bridge preserves known tensor shapes and F32 types. It does not define
CENT dialects or convert device instructions inside MLIR. Import produces a
``vllm_cent.ir`` function; the existing Python passes then choose storage and
emit CENT commands.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iree.compiler.ir import Module, Type  # type: ignore[import-not-found]
    from iree.compiler.ir import Operation as NativeOperation

from vllm_cent.ir import (
    AddOp,
    DType,
    Function,
    MulOp,
    Operation,
    TensorType,
    Value,
    value_names,
)

__all__ = ["from_mlir", "optimize_mlir", "to_mlir"]


def _bindings() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Import the optional MLIR libraries only when a bridge function needs them.

    Keeping these imports lazy lets the rest of the compiler run without IREE
    installed. Requesting MLIR support without it produces an installation hint.

    Returns:
        Modules for MLIR objects, function operations, and arithmetic operations,
        in that order.

    Raises:
        RuntimeError: If a required native MLIR module is not installed.
    """
    try:
        return (
            import_module("iree.compiler.ir"),
            import_module("iree.compiler.dialects.func"),
            import_module("iree.compiler.dialects.arith"),
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Native MLIR requires the optional iree-base-compiler==3.11.0 "
            "toolchain; install vllm-cent[mlir] in a dedicated environment."
        ) from error


def _tensor_type(ir: ModuleType, native_type: Type) -> TensorType:
    """Translate a native MLIR tensor shape into this compiler's ``TensorType``.

    All dimensions must already be known; a dimension marked ``?`` cannot be
    allocated by this path.

    Args:
        ir: Loaded MLIR module used to inspect native types.
        native_type: MLIR type to inspect while its owning context is still alive.

    Returns:
        Python ``TensorType`` with the same dimensions and fp32 element format.

    Raises:
        ValueError: If the type is not a ranked tensor, has unknown dimensions,
            uses another element format, or contains a zero/negative dimension.

    Examples:
        ``tensor<2x3xf32>`` becomes ``shape=(2, 3)`` with ``dtype=DType.FP32``.
    """
    if not isinstance(native_type, ir.RankedTensorType):
        raise ValueError("MLIR bridge requires ranked tensors")
    ranked = ir.RankedTensorType(native_type)
    if not ranked.has_static_shape or not isinstance(ranked.element_type, ir.F32Type):
        raise ValueError("MLIR bridge requires static f32 tensors")
    shape = tuple(ranked.shape)
    if any(dim <= 0 for dim in shape):
        raise ValueError("Tensor dimensions must be positive")
    return TensorType(shape=shape, dtype=DType.FP32)


def _string(ir: ModuleType, operation: NativeOperation, key: str, default: str) -> str:
    """Read an optional name or source-block label attached to an MLIR operation.

    Args:
        ir: Loaded MLIR module used to read string attributes.
        operation: Native operation that may carry the requested attribute.
        key: Attribute name, such as ``cent.name``.
        default: Fallback string for ordinary MLIR with no CENT naming information.

    Returns:
        Stored string if the attribute exists, otherwise ``default``.

    Raises:
        ValueError: If the attribute exists but is not a string attribute.
    """
    if key not in operation.attributes:
        return default
    return str(ir.StringAttr(operation.attributes[key]).value)


def _fresh_name(preferred: str, reserved: set[str]) -> str:
    """Choose a generated name without taking a name saved in the MLIR text.

    Args:
        preferred: Readable fallback, such as ``arg0`` or ``value0``.
        reserved: Explicit and previously generated names in this namespace.
            The selected name is added to this set.

    Returns:
        The preferred name, or that name with the first available numeric suffix.
    """
    candidate = preferred
    suffix = 0
    while candidate in reserved:
        candidate = f"{preferred}_{suffix}"
        suffix += 1
    reserved.add(candidate)
    return candidate


def to_mlir(graph: Function) -> str:
    """Build an MLIR function for the IR function, verify it, and return its text.

    Each argument becomes a function argument. Addition becomes ``arith.addf``
    and multiplication becomes ``arith.mulf``. The function returns the same
    results. Extra ``cent.*`` attributes preserve the argument and result
    names, as the IR printer shows them, when that text is later imported back
    into Python.

    Args:
        graph: Function of additions and multiplications over fp32 tensors
            with known shapes.

    Returns:
        Text printed by native MLIR for one function named ``main``.

    Raises:
        RuntimeError: If the optional native MLIR toolchain is unavailable.
        ValueError: If verification fails or an operation/type is unsupported.
    """
    graph.verify()
    ir, func, arith = _bindings()
    if any(not isinstance(op, (AddOp, MulOp)) for op in graph.operations):
        raise ValueError("Native MLIR supports only addition and multiplication")
    all_values = (
        *graph.operands,
        *(result for op in graph.operations for result in op.results),
    )
    if any(value.type.dtype is not DType.FP32 for value in all_values):
        raise ValueError("Native MLIR proof slice supports only f32 tensors")
    names = value_names(graph)
    # Native objects belong to a context. Construct and print them while it is
    # alive; return text rather than objects tied to this temporary context.
    with ir.Context(), ir.Location.unknown():

        def native_type(value: Value) -> object:
            """Return the MLIR tensor type with ``value``'s shape."""
            return ir.RankedTensorType.get(value.type.shape, ir.F32Type.get())

        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "main",
                (
                    [native_type(value) for value in graph.operands],
                    [native_type(value) for value in graph.results],
                ),
            )
            # MLIR may print arguments as %arg0, %arg1, etc. Keep the caller's
            # tensor names separately so importing the text recovers them.
            function.attributes["cent.input_names"] = ir.ArrayAttr.get(
                [ir.StringAttr.get(names[value]) for value in graph.operands]
            )
            block = function.add_entry_block()
            # Keys are IR values (compared by identity), mapped to the native
            # values that stand for them.
            natives = dict(zip(graph.operands, block.arguments, strict=True))
            # An insertion point tells native builders where new operations go.
            # Function order ensures each operand already exists in the table.
            with ir.InsertionPoint(block):
                for operation in graph.operations:
                    # Both native operations require matching tensor shapes;
                    # IR verification has already checked that contract.
                    native_class = (
                        arith.AddFOp if isinstance(operation, AddOp) else arith.MulFOp
                    )
                    native = native_class(*(natives[v] for v in operation.operands))
                    (result,) = operation.results
                    native.attributes["cent.name"] = ir.StringAttr.get(names[result])
                    natives[result] = native.result
                func.ReturnOp([natives[value] for value in graph.results])
        module.operation.verify()
        return str(module)


def from_mlir(text: str) -> Function:
    """Parse an MLIR function and recover its elementwise arithmetic.

    Native MLIR first checks syntax and its own operation rules. The importer
    then rejects features the CENT path cannot represent, rather than silently
    dropping them. The final IR function is verified too.

    Args:
        text: MLIR assembly containing a single function with one body block,
            supported tensor additions/multiplications, and a return.

    Returns:
        IR function with the same calculations, shapes, and result order.
        Original names are recovered when ``cent.*`` attributes are present;
        otherwise the importer generates names.

    Raises:
        RuntimeError: If the optional native MLIR toolchain is unavailable.
        ValueError: If parsing or verification fails, naming attributes are invalid,
            or a function, operation, or tensor type is outside the supported subset.
    """
    ir, _, _ = _bindings()
    with ir.Context():
        try:
            module = ir.Module.parse(text)
            module.operation.verify()
            graph = _import_module(ir, module)
            graph.verify()
            return graph
        except ir.MLIRError as error:
            raise ValueError(f"Native MLIR verification failed: {error}") from error


def _import_module(ir: ModuleType, module: Module) -> Function:
    """Walk a verified MLIR function and rebuild it as an IR function.

    Track the IR value that stands for each native argument or result. When
    an operation reads one of those native values, use that IR value. This
    preserves connections between calculations even if textual names differ.

    Args:
        ir: Loaded MLIR module used to inspect types and attributes.
        module: Parsed, verified native module, whose context remains alive.

    Returns:
        Arguments, ordered arithmetic operations, and results of the function.

    Raises:
        ValueError: If the module has multiple functions/body blocks, unsupported
            operations or types, explicit rounding modes, relaxed floating-point flags,
            or invalid attributes.
    """
    functions = list(module.body.operations)
    if len(functions) != 1 or functions[0].operation.name != "func.func":
        raise ValueError("MLIR bridge requires exactly one func.func")
    function = functions[0]
    blocks = list(function.regions[0].blocks)
    if len(blocks) != 1:
        raise ValueError("MLIR bridge requires one function body block")
    block = blocks[0]
    # Partial provenance is legal. Reserve all explicit names before generating
    # any fallback: a later result called arg0 must not collide with input zero.
    reserved = {
        _string(ir, native, "cent.name", "")
        for native in block.operations
        if "cent.name" in native.attributes
    }
    if "cent.input_names" in function.attributes:
        names = [
            str(ir.StringAttr(value).value)
            for value in ir.ArrayAttr(function.attributes["cent.input_names"])
        ]
        if len(names) != len(block.arguments):
            raise ValueError("cent.input_names must match function arguments")
        reserved.update(names)
    else:
        names = [
            _fresh_name(f"arg{index}", reserved)
            for index in range(len(block.arguments))
        ]
    arguments = tuple(
        Value(type=_tensor_type(ir, argument.type), name=name)
        for name, argument in zip(names, block.arguments, strict=True)
    )
    # Keys are native values, not their printed names. This follows actual
    # operand connections even when the printer chooses different SSA names.
    values = dict(zip(block.arguments, arguments, strict=True))
    operations: list[Operation] = []
    results: tuple[Value, ...] = ()
    for native in block.operations:
        kind = native.operation.name
        if kind == "func.return":
            results = tuple(values[value] for value in native.operands)
        elif kind in ("arith.addf", "arith.mulf"):
            # Fast-math flags permit assumptions such as ignoring NaNs. The IR
            # has no field for these promises, so accepting them would
            # silently lose part of the operation's meaning.
            if (
                "fastmath" in native.attributes
                and str(native.attributes["fastmath"]) != "#arith.fastmath<none>"
            ):
                raise ValueError("MLIR bridge does not support fastmath assumptions")
            # The IR has no explicit rounding-mode operand. Reject one even
            # when a newer MLIR runtime can parse and verify it.
            if "roundingmode" in native.attributes:
                raise ValueError("MLIR bridge does not support explicit rounding modes")
            name = (
                _string(ir, native, "cent.name", "")
                if "cent.name" in native.attributes
                else _fresh_name(f"value{len(operations)}", reserved)
            )
            operation_class = AddOp if kind == "arith.addf" else MulOp
            operation = operation_class(
                operands=tuple(values[value] for value in native.operands),
                name_hint=name,
            )
            # The IR infers the result type; MLIR states it. Both must agree.
            if operation.results[0].type != _tensor_type(ir, native.result.type):
                raise ValueError(f"MLIR result type of {name} differs from the IR's")
            operations.append(operation)
            values[native.result] = operation.results[0]
        else:
            raise ValueError(f"Unsupported MLIR operation: {kind}")
    return Function(
        name="main", operands=arguments, operations=tuple(operations), results=results
    )


def optimize_mlir(text: str) -> str:
    """Run MLIR's standard simplification and repeated-expression removal passes.

    ``canonicalize`` applies registered simplification rules, including removing
    unused work. CSE means common subexpression elimination: equivalent operations
    can share a result instead of repeating the calculation. These are native
    MLIR passes; they do not select CENT instructions or allocate CENT memory.

    Args:
        text: Assembly accepted by the native parser and verifier. Unlike ``from_mlir``,
            this function does not restrict the input to the elementwise arithmetic subset.

    Returns:
        Verified MLIR text after canonicalize and CSE. Importing it into this
        compiler still requires that it fit the supported subset of ``from_mlir``.

    Raises:
        RuntimeError: If the optional native MLIR toolchain is unavailable.
        ValueError: If native parsing, verification, or the pass pipeline fails.
    """
    ir, _, _ = _bindings()
    passmanager = import_module("iree.compiler.passmanager")
    with ir.Context():
        try:
            module = ir.Module.parse(text)
            module.operation.verify()
            manager = passmanager.PassManager.parse("builtin.module(canonicalize,cse)")
            manager.run(module.operation)
            module.operation.verify()
            return str(module)
        except ir.MLIRError as error:
            raise ValueError(f"Native MLIR pass pipeline failed: {error}") from error
