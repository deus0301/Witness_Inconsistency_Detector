"""Post-NLI checks: what they set aside, and the real conflicts they must leave alone."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.pair_checks import CHECKS, apply_checks, check_pair, quantities, same_quantity

SET_ASIDE = [
    ("same_quantity", "The car was doing about 30 mph.", "The car was doing about 50 km/h."),
    ("same_quantity", "About 20 people were waiting at the stop.", "Around 25 people were waiting at the stop."),
    ("non_observation", "The man had a gun in his right hand.", "I didn't see a gun."),
    ("reported_speech", "The driver told me he had stopped at the crossing.", "The driver did not stop at the crossing."),
    ("different_time", "The man was alive when I got there.", "The man was dead when the ambulance arrived."),
    ("different_referent", "The cyclist was badly injured.", "The pedestrian was not hurt at all."),
    ("different_referent", "My car was parked outside the bank.", "My car was in the garage all day."),
]

REAL_CONFLICTS = [
    ("The car turned left onto Mill Road.", "The car turned right onto Mill Road."),
    ("The man who ran off wore a red jacket.", "The man who ran off wore a blue jacket."),
    ("The car hit the cyclist before the bus arrived.", "The bus arrived before the car hit the cyclist."),
    ("Two men got out of the car.", "Only one person was in the car."),
    ("I heard two shots from the alley.", "I heard three shots from the alley."),
    ("I saw the driver stop at the crossing.", "The driver did not stop at the crossing."),
    ("The light was red when the blue car went through.", "The light was green when the blue car went through."),
    ("It was approx. 10 p.m. when it started.", "It was around 6 p.m. when it started."),
    ("(Up to) 150 people perished in the crash", "There were 148 people on board the plane"),
    ("The gunman has been shot dead", "The suspect is still at large"),
]


@pytest.mark.parametrize("expected,a,b", SET_ASIDE)
def test_known_non_contradictions_are_set_aside(expected, a, b):
    assert check_pair(a, b, "Witness A", "Witness B") == expected
    assert check_pair(b, a, "Witness B", "Witness A") == expected  # order does not matter


@pytest.mark.parametrize("a,b", REAL_CONFLICTS)
def test_real_conflicts_are_kept(a, b):
    assert check_pair(a, b, "Witness A", "Witness B") is None


def test_quantities_and_units():
    assert quantities("about 30 mph") == [("speed", pytest.approx(48.28032), True, "mph")]
    assert quantities("at 10 p.m.") == [] and quantities("at 9:15") == []
    assert same_quantity("It was 2 miles away.", "It was 3.2 km away.")      # 3.22 km, units differ
    assert not same_quantity("Twenty people were hurt.", "Twenty-five people were hurt.")
    assert not same_quantity("Exactly 20 people.", "21 people.")             # exact numbers must match


def test_own_things_only_across_accounts():
    a, b = "My car was parked outside.", "My car was in the garage."
    assert check_pair(a, b, "Witness A", "Witness B") == "different_referent"
    assert check_pair(a, b, "Witness A", "Witness A") is None


def test_apply_checks_moves_mass_to_neutral_and_only_touches_flagged_pairs():
    a = ["The cyclist was badly injured.", "The light was red."]
    b = ["The pedestrian was not hurt at all.", "The light was green."]
    probs = np.array([[0.0, 0.01, 0.99], [0.0, 0.02, 0.98]], dtype=np.float32)
    eff, aside = apply_checks(a, b, ["A", "A"], ["B", "B"], probs, 0.9, CHECKS)
    assert aside == {0: "different_referent"}
    assert eff[0, 2] == 0 and eff[0, 1] == pytest.approx(1.0)
    assert np.allclose(eff[1], probs[1]) and np.allclose(probs[0], [0.0, 0.01, 0.99])  # input untouched
    eff, aside = apply_checks(a, b, ["A", "A"], ["B", "B"], probs, 0.9, [])
    assert aside == {} and np.allclose(eff, probs)
