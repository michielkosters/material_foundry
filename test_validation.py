import unittest
from validate_candidates import build_summary, joint_pass, compare_and_match


class ValidationTests(unittest.TestCase):
    def test_failures_and_pending_keep_original_denominator(self):
        good = dict(proposal_id='1', status='validated', converged=True,
                    within_0p1=True, unique=True, novel=True)
        bad = dict(proposal_id='2', status='failed')
        result = build_summary([good, bad], 100, 3, 1, 1, 'RUNNING')
        self.assertIsNone(result['diagnostic_joint_yield'])
        self.assertEqual(result['joint_yield_bounds'], [.01, .02])
        result = build_summary([good, bad], 100, 2, 1, 1, 'COMPLETE')
        self.assertEqual(result['diagnostic_joint_yield'], .01)

    def test_joint_requires_convergence_and_recomputed_novelty(self):
        row = dict(status='validated', converged=True, within_0p1=True, unique=True, novel=True)
        self.assertTrue(joint_pass(row))
        row['novel'] = False
        self.assertFalse(joint_pass(row))
        row['novel'], row['converged'] = True, False
        self.assertFalse(joint_pass(row))

    def test_relaxed_endpoint_can_lose_novelty_and_uniqueness(self):
        from pymatgen.core import Lattice, Structure
        from pymatgen.analysis.structure_matcher import StructureMatcher
        known = Structure(Lattice.cubic(3), ['Si'], [[0, 0, 0]])
        self.assertEqual(compare_and_match(known.copy(), [known], [known], StructureMatcher()), (False, False))


if __name__ == '__main__':
    unittest.main()
