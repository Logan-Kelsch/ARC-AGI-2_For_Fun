from __future__ import annotations

from collections.abc import Callable

from .solver import NullSolver, PrimitiveSearchSolver, Solver

SolverFactory = Callable[[], Solver]

_SOLVERS: dict[str, SolverFactory] = {
    "null": NullSolver,
    "primitive": PrimitiveSearchSolver,
}


def available_solvers() -> tuple[str, ...]:
    return tuple(sorted(_SOLVERS))


def create_solver(name: str) -> Solver:
    try:
        return _SOLVERS[name]()
    except KeyError as exc:
        raise KeyError(
            f"Unknown solver {name!r}. Available: {', '.join(available_solvers())}"
        ) from exc


def register_solver(name: str, factory: SolverFactory, *, replace: bool = False) -> None:
    if name in _SOLVERS and not replace:
        raise ValueError(f"Solver {name!r} is already registered.")
    _SOLVERS[name] = factory
