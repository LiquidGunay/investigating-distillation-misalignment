from __future__ import annotations

import runpy

from inheritance.config import repository_root

_SCRIPT = runpy.run_path(str(repository_root() / "scripts" / "evaluate_teacher_sources.py"))
steering_condition = _SCRIPT["steering_condition"]


def test_signed_steering_conditions_are_unambiguous() -> None:
    assert steering_condition(17, -2.0) == "steering_negative_l17_alpha2"
    assert steering_condition(17, 0.0) == "steering_zero"
    assert steering_condition(17, 2.0) == "steering_positive_l17_alpha2"
