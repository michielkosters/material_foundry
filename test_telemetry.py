import json
import tempfile
import unittest
from pathlib import Path
from telemetry import Telemetry


class TelemetryTests(unittest.TestCase):
    def test_known_training_match_cannot_inflate_discovery_upper_bound(self):
        with tempfile.TemporaryDirectory() as root:
            tracker = Telemetry(Path(root)/'run')
            tracker.record(dict(proposal_id='1', status='missing_reference',
                                converged=True, unique=True, novel=False))
            self.assertEqual(tracker.summary()['un_at_0p1_yield_bounds'], [0, 0])
            self.assertEqual(tracker.summary()['un_yield_at_0p1'], 0)

    def test_partial_yield_and_resume_and_duplicate(self):
        with tempfile.TemporaryDirectory() as root:
            directory = Path(root)/'run'
            tracker = Telemetry(directory)
            tracker.record(dict(proposal_id='1', status='rejected'))
            tracker.record(dict(proposal_id='2', status='evaluated', converged=True,
                                above_hull=.05, unique=True, novel=True))
            self.assertEqual(tracker.summary()['un_yield_at_0p1'], .5)
            tracker.record(dict(proposal_id='3', status='missing_reference'))
            self.assertIsNone(tracker.summary()['un_yield_at_0p1'])
            self.assertEqual(tracker.summary()['un_at_0p1_yield_bounds'], [1/3, 2/3])
            tracker.checkpoint()
            resumed = Telemetry(directory, resume=True)
            self.assertEqual(resumed.summary(), tracker.summary())
            self.assertEqual(json.loads((directory/'summary.json').read_text()), tracker.summary())
            with self.assertRaises(ValueError):
                resumed.record(dict(proposal_id='1', status='rejected'))

    def test_failures_in_denominator_and_threshold_labels(self):
        with tempfile.TemporaryDirectory() as root:
            tracker = Telemetry(Path(root)/'run')
            tracker.record(dict(proposal_id='1', status='generation_failed'))
            tracker.record(dict(proposal_id='2', status='relaxation_failed'))
            tracker.record(dict(proposal_id='3', status='evaluated', converged=True,
                                above_hull=0, unique=True, novel=True))
            self.assertEqual(tracker.summary()['un_yield_at_0'], 1/3)
            self.assertEqual(tracker.summary()['early_rejection_rate'], 0)
            self.assertFalse(tracker.summary()['record_verified'])

    def test_invalid_event_does_not_pollute_state(self):
        with tempfile.TemporaryDirectory() as root:
            tracker = Telemetry(Path(root)/'run')
            for extra in [dict(oracle_ms=float('nan')), dict(relax_seconds=-1), dict(unique='false')]:
                with self.assertRaises(ValueError):
                    tracker.record(dict(proposal_id='x', status='evaluated', **extra))
            self.assertEqual(tracker.summary()['outcomes_recorded'], 0)


if __name__ == '__main__':
    unittest.main()
