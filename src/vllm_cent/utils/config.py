"""Strict reading of YAML configuration documents.

Run specs and device specs are hand-written YAML. A misspelled key would
otherwise be silently ignored and a default used instead, so every helper
here rejects unknown keys and wrong value types, naming the document and the
dotted path of the offending entry (e.g. ``device spec architecture.chanels``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "names",
    "nonnegative_int",
    "positive_int",
    "section",
]


def section(
    document: str, data: Mapping[str, Any], path: str, keys: set[str]
) -> Mapping[str, Any]:
    """Return one section of a document, rejecting unknown keys.

    Args:
        document: Kind of document, for error messages (e.g. ``"run spec"``).
        data: The mapping that contains the section.
        path: Dotted path of the section within the document, e.g.
            ``"parallel"``; its last part is the key looked up in ``data``.
        keys: Allowed keys in the section.

    Returns:
        The section, empty if absent.

    Raises:
        ValueError: If the section is not a mapping or has unknown keys.
    """

    found = data.get(path.rpartition(".")[2]) or {}
    if not isinstance(found, Mapping):
        raise ValueError(f"{document} section {path!r} must be a mapping")
    unknown = sorted(set(found) - keys)
    if unknown:
        raise ValueError(f"unknown keys in {document} section {path!r}: {unknown}")
    return found


def positive_int(document: str, path: str, value: object) -> int:
    """Return a positive int setting.

    Args:
        document: Kind of document, for error messages.
        path: Dotted path of the setting, for error messages.
        value: The value read from the document.

    Returns:
        The value.

    Raises:
        ValueError: If the value is not an int of at least 1 (a bool is not
            accepted, although Python treats it as an int).
    """

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{document} {path} must be a positive int, got {value!r}")
    return value


def nonnegative_int(document: str, path: str, value: object) -> int:
    """Return a nonnegative int setting, such as an index.

    Args:
        document: Kind of document, for error messages.
        path: Dotted path of the setting, for error messages.
        value: The value read from the document.

    Returns:
        The value.

    Raises:
        ValueError: If the value is not an int of at least 0.
    """

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{document} {path} must be an int >= 0, got {value!r}")
    return value


def names(document: str, path: str, value: object) -> tuple[str, ...]:
    """Return a list-of-names setting as a tuple.

    Args:
        document: Kind of document, for error messages.
        path: Dotted path of the setting, for error messages.
        value: The value read from the document.

    Returns:
        The names, in document order.

    Raises:
        ValueError: If the value is not a list of strings.
    """

    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{document} {path} must be a list of names, got {value!r}")
    return tuple(value)
