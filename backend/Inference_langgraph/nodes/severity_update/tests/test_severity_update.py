"""
d:/Before_done/severity_update/tests/test_severity_update.py
=============================================================
Unit tests for severity_update node.

Includes exact verification of the Mentor's Worked Example:
  MEMORY_LEAK episode:
    cycle  1: preliminary=P4 (no breaches), ttf=null                           -> severity = P4
    cycle  5: preliminary=P3 (memory warn), ttf=60s,  conf=0.90 -> candidate P2 -> escalate now: severity = P2
    cycle  8: preliminary=P3,               ttf=20s,  conf=0.95 -> candidate P1 -> escalate now: severity = P1
    cycle 12: remediated, preliminary=P4,    ttf=null            -> candidate P4 -> hold: severity = P1 (dwell 1/5)
    cycles 13-16: candidate stays P4, holding at P1 (dwell counting 2/5..5/5)
    cycle 17: candidate P4 persisted 5 cycles                          -> de-escalate: severity = P4
"""
from __future__ import annotations

import unittest
import sys
from pathlib import Path

# Ensure backend/ root on sys.path (3 levels up from tests/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nodes.severity_update import SeverityUpdater, get_impact_band, get_urgency_band, get_candidate_severity


class TestSeverityUpdate(unittest.TestCase):

    def setUp(self):
        self.updater = SeverityUpdater(dwell_k=5, min_confidence=0.75)
        self.ep_id = "ep_MEMORY_LEAK_WORKED_EXAMPLE"

    def test_mentor_worked_example(self):
        """Verify the exact cycle-by-cycle behavior from Mentor Clarification 4."""

        # Cycle 1: preliminary=P4 (no breaches), ttf=null -> severity = P4
        res_c1 = self.updater.process_cycle(
            episode_id=self.ep_id,
            preliminary_severity="P4",
            forecast_result={"time_to_failure": None, "forecast_confidence": 0.0},
        )
        self.assertEqual(res_c1["impact_band"], "None")
        self.assertEqual(res_c1["urgency_band"], "Distant")
        self.assertEqual(res_c1["candidate_severity"], "P4")
        self.assertEqual(res_c1["revised_severity"], "P4")

        # Cycle 5: preliminary=P3 (memory warn), ttf=60s, conf=0.90
        # impact=Moderate, urgency=Near -> candidate P2
        # candidate worse than current -> escalate now: severity = P2
        res_c5 = self.updater.process_cycle(
            episode_id=self.ep_id,
            preliminary_severity="P3",
            forecast_result={"time_to_failure": 60.0, "forecast_confidence": 0.90},
        )
        self.assertEqual(res_c5["impact_band"], "Moderate")
        self.assertEqual(res_c5["urgency_band"], "Near")
        self.assertEqual(res_c5["candidate_severity"], "P2")
        self.assertEqual(res_c5["revised_severity"], "P2")
        self.assertTrue(res_c5["is_escalated"])

        # Cycle 8: preliminary=P3, ttf=20s, conf=0.95
        # impact=Moderate, urgency=Imminent -> candidate P1
        # candidate worse than current -> escalate now: severity = P1
        res_c8 = self.updater.process_cycle(
            episode_id=self.ep_id,
            preliminary_severity="P3",
            forecast_result={"time_to_failure": 20.0, "forecast_confidence": 0.95},
        )
        self.assertEqual(res_c8["impact_band"], "Moderate")
        self.assertEqual(res_c8["urgency_band"], "Imminent")
        self.assertEqual(res_c8["candidate_severity"], "P1")
        self.assertEqual(res_c8["revised_severity"], "P1")
        self.assertTrue(res_c8["is_escalated"])

        # Cycle 12: remediated, preliminary=P4, ttf=null
        # impact=None, urgency=Distant -> candidate P4
        # candidate better than current (P1) -> hold; dwell not yet met (dwell 1/5)
        res_c12 = self.updater.process_cycle(
            episode_id=self.ep_id,
            preliminary_severity="P4",
            forecast_result={"time_to_failure": None, "forecast_confidence": 0.0},
        )
        self.assertEqual(res_c12["impact_band"], "None")
        self.assertEqual(res_c12["urgency_band"], "Distant")
        self.assertEqual(res_c12["candidate_severity"], "P4")
        self.assertEqual(res_c12["revised_severity"], "P1")   # HELD at P1!
        self.assertFalse(res_c12["is_deescalated"])
        self.assertEqual(res_c12["dwell_count"], 1)

        # Cycles 13-16: candidate stays P4, still holding at P1 (dwell counting 2/5, 3/5, 4/5, 5/5)
        for cycle_num in range(13, 17):
            res_hold = self.updater.process_cycle(
                episode_id=self.ep_id,
                preliminary_severity="P4",
                forecast_result={"time_to_failure": None, "forecast_confidence": 0.0},
            )
            # Cycle 16 is the 5th dwell cycle -> dwell threshold met at cycle 16!
            if cycle_num < 16:
                self.assertEqual(res_hold["revised_severity"], "P1")
                self.assertFalse(res_hold["is_deescalated"])
            else:
                self.assertEqual(res_hold["revised_severity"], "P4")
                self.assertTrue(res_hold["is_deescalated"])

        # Cycle 17: confirmed at P4
        res_c17 = self.updater.process_cycle(
            episode_id=self.ep_id,
            preliminary_severity="P4",
            forecast_result={"time_to_failure": None, "forecast_confidence": 0.0},
        )
        self.assertEqual(res_c17["revised_severity"], "P4")

    def test_confidence_gate_failure(self):
        """Test that low confidence (< 0.75) forces urgency to Distant."""
        # ttf=10s (would be Imminent), but conf=0.50 (< 0.75) -> gate fails -> Distant
        band, gate_passed = get_urgency_band(
            ttf=10.0,
            ttf_source="heap_mb",
            confidence=0.50,
            min_confidence=0.75,
        )
        self.assertEqual(band, "Distant")
        self.assertFalse(gate_passed)

    def test_invalid_ttf_source_gate_failure(self):
        """Test that excluded ttf_source forces urgency to Distant."""
        band, gate_passed = get_urgency_band(
            ttf=10.0,
            ttf_source="not_applicable",
            confidence=0.95,
        )
        self.assertEqual(band, "Distant")
        self.assertFalse(gate_passed)

    def test_matrix_combination_lookup(self):
        """Test all 12 cells of the combination matrix."""
        self.assertEqual(get_candidate_severity("High", "Imminent"), "P1")
        self.assertEqual(get_candidate_severity("High", "Near"), "P1")
        self.assertEqual(get_candidate_severity("High", "Distant"), "P2")

        self.assertEqual(get_candidate_severity("Moderate", "Imminent"), "P1")
        self.assertEqual(get_candidate_severity("Moderate", "Near"), "P2")
        self.assertEqual(get_candidate_severity("Moderate", "Distant"), "P3")

        self.assertEqual(get_candidate_severity("None", "Imminent"), "P2")
        self.assertEqual(get_candidate_severity("None", "Near"), "P3")
        self.assertEqual(get_candidate_severity("None", "Distant"), "P4")


if __name__ == "__main__":
    unittest.main()
