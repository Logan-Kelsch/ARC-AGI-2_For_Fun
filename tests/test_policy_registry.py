import pytest

pytest.importorskip("arcengine")

from arc_fun.policy import NullPolicy
from arc_fun.policy_registry import available_policies, create_policy, register_policy


def test_null_policy_is_registered():
    assert "null" in available_policies()
    assert isinstance(create_policy("null"), NullPolicy)


def test_registration_is_replace_protected():
    class DummyPolicy(NullPolicy):
        name = "dummy"

    register_policy("dummy-test-policy", DummyPolicy, replace=True)

    with pytest.raises(ValueError):
        register_policy("dummy-test-policy", DummyPolicy)

    assert isinstance(create_policy("dummy-test-policy"), DummyPolicy)
