from __future__ import annotations

from collections.abc import Callable

from .policy import NullPolicy, Policy

PolicyFactory = Callable[[], Policy]

_POLICY_FACTORIES: dict[str, PolicyFactory] = {
    "null": NullPolicy,
}


def available_policies() -> tuple[str, ...]:
    """Return registered policy names in stable order."""
    return tuple(sorted(_POLICY_FACTORIES))


def get_policy_factory(name: str) -> PolicyFactory:
    """Resolve a policy factory by its short registry name."""
    try:
        return _POLICY_FACTORIES[name]
    except KeyError as exc:
        choices = ", ".join(available_policies())
        raise KeyError(f"Unknown policy {name!r}. Available: {choices}") from exc


def create_policy(name: str) -> Policy:
    """Create a fresh policy instance."""
    return get_policy_factory(name)()


def register_policy(
    name: str,
    factory: PolicyFactory,
    *,
    replace: bool = False,
) -> None:
    """Register a new policy factory for local evaluation."""
    if not name:
        raise ValueError("Policy name must be non-empty.")
    if name in _POLICY_FACTORIES and not replace:
        raise ValueError(f"Policy {name!r} is already registered.")
    _POLICY_FACTORIES[name] = factory
