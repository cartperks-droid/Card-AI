"""Explicit experimental battle semantics; unsupported cards fail closed."""

from .reference import Battle, Fighter, Options, Result, simulate, simulate_batch

__all__ = ["Battle", "Fighter", "Options", "Result", "simulate", "simulate_batch"]
