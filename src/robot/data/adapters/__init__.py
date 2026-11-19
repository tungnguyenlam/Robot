"""Dataset adapters. Importing this package registers every adapter in ``ADAPTERS``."""

from robot.data.adapters.base import ADAPTERS, DatasetAdapter, get_adapter, register
from robot.data.adapters import ego4d, epic_kitchens, hoi4d  # noqa: F401  (registration)

__all__ = ["ADAPTERS", "DatasetAdapter", "get_adapter", "register"]
