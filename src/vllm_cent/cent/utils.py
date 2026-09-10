"""Small arithmetic helpers shared by CENT compilation modules."""

__all__ = ["ceil_div", "require_nonnegative", "require_positive"]


def require_nonnegative(name: str, value: int) -> None:
    """Reject an integer below zero.

    Args:
        name: Name to include in the error message.
        value: Integer to check.

    Raises:
        ValueError: If ``value`` is negative.
    """

    if value < 0:
        raise ValueError(f"{name} cannot be negative")


def require_positive(name: str, value: int) -> None:
    """Reject an integer below one.

    Args:
        name: Name to include in the error message.
        value: Integer to check.

    Raises:
        ValueError: If ``value`` is less than one.
    """

    if value < 1:
        raise ValueError(f"{name} must be at least 1")


def ceil_div(dividend: int, divisor: int) -> int:
    """Divide two integers and round up to the next whole number.

    Args:
        dividend: Number of items to divide.
        divisor: Number of items that fit in one group.

    Returns:
        Number of groups needed to hold every item.

    Raises:
        ValueError: If either argument is less than 1.
    """

    if dividend < 1 or divisor < 1:
        raise ValueError("ceil_div arguments must be at least 1")

    # Adding divisor - 1 creates one extra group only when there is a remainder.
    # This avoids converting large tensor sizes to floating point.
    return (dividend + divisor - 1) // divisor
