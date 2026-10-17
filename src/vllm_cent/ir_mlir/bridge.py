"""Read and write MLIR for the supported elementwise arithmetic graphs.

MLIR is a compiler framework that represents calculations as typed operations.
A dialect is a family of operations: ``func`` supplies functions and returns, and
``arith`` supplies arithmetic such as floating-point addition. IREE packages the
native MLIR Python bindings used here to construct, parse, and verify them.

This bridge preserves known tensor shapes and F32 types. It does not define
CENT dialects or convert device instructions inside MLIR. After import, the
existing Python passes choose storage and emit CENT commands.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from iree.compiler.ir import Module, Type  # type: ignore[import-not-found]
    from iree.compiler.ir import Operation as NativeOperation

from vllm_cent.pipeline.ir import (
    Operation,
    ScalarType,
    SemanticGraph,
    TensorType,
    Value,
    verify_semantic,
)
from vllm_cent.pipeline.operations import AddOperation, MultiplyOperation

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
        Python ``TensorType`` with the same dimensions and F32 element format.

    Raises:
        ValueError: If the type is not a ranked tensor, has unknown dimensions,
            uses another element format, or contains a zero/negative dimension.

    Examples:
        ``tensor<2x3xf32>`` becomes ``shape=(2, 3)`` with ``dtype=ScalarType.F32``.
    """
    if not isinstance(native_type, ir.RankedTensorType):
        raise ValueError("MLIR bridge requires ranked tensors")
    ranked = ir.RankedTensorType(native_type)
    if not ranked.has_static_shape or not isinstance(ranked.element_type, ir.F32Type):
        raise ValueError("MLIR bridge requires static f32 tensors")
    return TensorType(tuple(ranked.shape), ScalarType.F32)


def _string(ir: ModuleType, operation: NativeOperation, key: str, default: str) -> str:
    """Read an optional name or source-block label attached to an MLIR operation.

    Args:
        ir: Loaded MLIR module used to read string attributes.
        operation: Native operation that may carry the requested attribute.
        key: Attribute name, such as ``cent.name`` or ``cent.block``.
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


def to_mlir(graph: SemanticGraph) -> str:
    """Build an MLIR function for the graph, verify it, and return its text.

    Each graph input becomes a function argument. Addition becomes ``arith.addf``
    and multiplication becomes ``arith.mulf``. The function returns the graph's
    requested outputs. Extra ``cent.*`` attributes
    preserve original tensor names, operation names, and model-block labels when
    that text is later imported back into Python.

    Args:
        graph: Named tensor inputs, additions, and multiplications with known
            shapes and F32 types.
            Operations must read only inputs or previously produced results.

    Returns:
        Text printed by native MLIR for one function named ``main``.

    Raises:
        RuntimeError: If the optional native MLIR toolchain is unavailable.
        ValueError: If graph verification fails or an operation/type is unsupported.
    """
    verify_semantic(graph)
    ir, func, arith = _bindings()
    if any(
        not isinstance(op, (AddOperation, MultiplyOperation)) for op in graph.operations
    ):
        raise ValueError("Native MLIR supports only addition and multiplication")
    all_values = (*graph.inputs, *(op.output for op in graph.operations))
    if any(value.type.dtype is not ScalarType.F32 for value in all_values):
        raise ValueError("Native MLIR proof slice supports only f32 tensors")
    # Native objects belong to a context. Construct and print them while it is
    # alive; return text rather than objects tied to this temporary context.
    with ir.Context(), ir.Location.unknown():
        types = {
            value.name: ir.RankedTensorType.get(value.type.shape, ir.F32Type.get())
            for value in all_values
        }
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp(
                "main",
                (
                    [types[value.name] for value in graph.inputs],
                    [types[name] for name in graph.outputs],
                ),
            )
            # MLIR may print arguments as %arg0, %arg1, etc. Keep the caller's
            # tensor names separately so importing the text recovers them.
            function.attributes["cent.input_names"] = ir.ArrayAttr.get(
                [ir.StringAttr.get(value.name) for value in graph.inputs]
            )
            block = function.add_entry_block()
            values = dict(
                zip(
                    (value.name for value in graph.inputs), block.arguments, strict=True
                )
            )
            # An insertion point tells native builders where new operations go.
            # Graph order ensures each operand already exists in the value table.
            with ir.InsertionPoint(block):
                for operation in graph.operations:
                    # Both native operations require matching tensor shapes;
                    # semantic verification has already checked that contract.
                    native_type = (
                        arith.AddFOp
                        if isinstance(operation, AddOperation)
                        else arith.MulFOp
                    )
                    native = native_type(*(values[name] for name in operation.inputs))
                    for key, value in (
                        ("cent.name", operation.name),
                        ("cent.output", operation.output.name),
                        ("cent.block", operation.block),
                    ):
                        native.attributes[key] = ir.StringAttr.get(value)
                    values[operation.output.name] = native.result
                func.ReturnOp([values[name] for name in graph.outputs])
        module.operation.verify()
        return str(module)


def from_mlir(text: str) -> SemanticGraph:
    """Parse an MLIR function and recover its elementwise arithmetic graph.

    Native MLIR first checks syntax and its own operation rules. The importer
    then rejects features the CENT path cannot represent, rather than silently
    dropping them. The final Python graph is verified too.

    Args:
        text: MLIR assembly containing a single function with one body block,
            supported tensor additions/multiplications, and a return.

    Returns:
        ``SemanticGraph`` with the same calculations, shapes, and result order.
        Original names and model-block labels are recovered when ``cent.*`` attributes
        are present; otherwise the importer generates names.

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
            verify_semantic(graph)
            return graph
        except ir.MLIRError as error:
            raise ValueError(f"Native MLIR verification failed: {error}") from error


def _import_module(ir: ModuleType, module: Module) -> SemanticGraph:
    """Walk a verified MLIR function and rebuild its Python graph objects.

    Track each native argument or result in a name table. When an operation reads
    one of those native values, use the corresponding Python tensor name. This
    preserves connections between calculations even if textual names differ.

    Args:
        ir: Loaded MLIR module used to inspect types and attributes.
        module: Parsed, verified native module, whose context remains alive.

    Returns:
        Input tensors, ordered arithmetic operations, and output names for the Python graph.

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
    value_names_reserved = {
        _string(ir, native, "cent.output", "")
        for native in block.operations
        if "cent.output" in native.attributes
    }
    operation_names_reserved = {
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
        value_names_reserved.update(names)
    else:
        names = [
            _fresh_name(f"arg{index}", value_names_reserved)
            for index in range(len(block.arguments))
        ]
    inputs = tuple(
        Value(name, _tensor_type(ir, argument.type))
        for name, argument in zip(names, block.arguments, strict=True)
    )
    # Keys are native values, not their printed names. This follows actual
    # operand connections even when the printer chooses different SSA names.
    value_names = dict(zip(block.arguments, names, strict=True))
    operations: list[Operation] = []
    outputs: tuple[str, ...] = ()
    for native in block.operations:
        kind = native.operation.name
        if kind == "func.return":
            outputs = tuple(value_names[value] for value in native.operands)
        elif kind in ("arith.addf", "arith.mulf"):
            # Fast-math flags permit assumptions such as ignoring NaNs. Our
            # Python graph has no field for these promises, so accepting them
            # would silently lose part of the operation's meaning.
            if (
                "fastmath" in native.attributes
                and str(native.attributes["fastmath"]) != "#arith.fastmath<none>"
            ):
                raise ValueError("MLIR bridge does not support fastmath assumptions")
            # The semantic graph has no explicit rounding-mode operand. Reject
            # one even when a newer MLIR runtime can parse and verify it.
            if "roundingmode" in native.attributes:
                raise ValueError("MLIR bridge does not support explicit rounding modes")
            output_name = (
                _string(ir, native, "cent.output", "")
                if "cent.output" in native.attributes
                else _fresh_name(f"value{len(operations)}", value_names_reserved)
            )
            operation_type = AddOperation if kind == "arith.addf" else MultiplyOperation
            prefix = "add" if kind == "arith.addf" else "multiply"
            operation = operation_type(
                name=(
                    _string(ir, native, "cent.name", "")
                    if "cent.name" in native.attributes
                    else _fresh_name(
                        f"{prefix}{len(operations)}", operation_names_reserved
                    )
                ),
                inputs=tuple(value_names[value] for value in native.operands),
                output=Value(output_name, _tensor_type(ir, native.result.type)),
                block=_string(ir, native, "cent.block", ""),
            )
            operations.append(operation)
            value_names[native.result] = output_name
        else:
            raise ValueError(f"Unsupported MLIR operation: {kind}")
    return SemanticGraph(inputs=inputs, operations=tuple(operations), outputs=outputs)


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
