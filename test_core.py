"""Correctness regressions. Run with the WSL project Python (Sage + ASE + pymatgen)."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from sage.all import QQ
from algebraic_oracle import (
    AlgebraicOracle,
    expand_exact,
    optimize_wyckoff_clearance_qq,
    periodic_connectivity,
    reduce_cell_lll_zz,
)
from evaluator_and_hull import ReferenceHull
from telemetry import Telemetry


class GeometryTests(unittest.TestCase):
    def test_exact_orbit_and_collision(self):
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        operations = [(identity, [0, 0, 0]), (identity, [QQ(1)/2, 0, 0])]
        result = expand_exact(operations, [('Si', [0, 0, 0])])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[1][1][0], QQ(1)/2)
        with self.assertRaises(ValueError):
            expand_exact(operations, [('Si', [0, 0, 0]), ('O', [0, 0, 0])])

    def test_self_image_clash(self):
        self.assertEqual(AlgebraicOracle().inspect([[0, 0, 0]], ['Si'], np.eye(3))[1], 'ERR_CLASH')

    def test_skewed_cell_clash_missed_by_fractional_rounding(self):
        cell = [[4, 0, 0], [3.9, .2, 0], [0, 0, 5]]
        passed, reason, _ = AlgebraicOracle().inspect([[0, 0, 0], [.49, .49, 0]], ['C', 'C'], cell)
        self.assertFalse(passed)
        self.assertEqual(reason, 'ERR_CLASH')

    def test_single_atom_cubic_network_and_basis_invariance(self):
        oracle = AlgebraicOracle(require_3d=True)
        cell = np.eye(3)*2.1
        first = oracle.inspect([[0, 0, 0]], ['Si'], cell)
        second = oracle.inspect([[0, 0, 0]], ['Si'], np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]]) @ cell)
        self.assertTrue(first[0])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[2]['degrees'], [6])

    def test_rank_does_not_imply_single_network(self):
        result = periodic_connectivity([[(0, (2, 0, 0)), (0, (0, 1, 0)), (0, (0, 0, 1))]])
        self.assertEqual(result['periodic_rank'], 3)
        self.assertEqual(result['translation_index'], 2)
        self.assertFalse(result['single_3d_network'])

    def test_dimensions(self):
        for rank in range(4):
            edges = [(0, tuple(int(k == axis) for k in range(3))) for axis in range(rank)]
            self.assertEqual(periodic_connectivity([edges])['periodic_rank'], rank)
        self.assertFalse(periodic_connectivity([[], []])['quotient_connected'])

    def test_bad_inputs_and_scales(self):
        oracle = AlgebraicOracle()
        self.assertEqual(oracle.inspect([[0, 0, 0]], ['Qq'], np.eye(3))[1], 'ERR_SPECIES')
        self.assertEqual(oracle.inspect([[float('nan'), 0, 0]], ['Si'], np.eye(3))[1], 'ERR_INPUT')
        self.assertEqual(oracle.inspect([], [], np.eye(3))[1], 'ERR_INPUT')
        with self.assertRaises(ValueError):
            AlgebraicOracle(clash_scale=2, bond_scale=1)

    def test_sage_rational_wyckoff_clearance_heals_1d_clash(self):
        zero_proj = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        ops_fixed = [(zero_proj, [0, 0, 0])]
        # Free z-axis Wyckoff site (x=0, y=0, z=z)
        ops_free = [([[0, 0, 0], [0, 0, 0], [0, 0, 1]], [0, 0, 0])]
        cell = np.diag([4.0, 4.0, 6.0])
        healed = optimize_wyckoff_clearance_qq(
            [ops_fixed, ops_free], ['Si', 'Si'], [[0, 0, 0], [0, 0, 0.08]], cell, steps=10)
        self.assertEqual(healed[0], [0.0, 0.0, 0.0])
        self.assertGreater(healed[1][2], 0.25)
        passed, reason, _ = AlgebraicOracle().inspect(healed, ['Si', 'Si'], cell)
        self.assertTrue(passed, reason)

    def test_lll_lattice_reduction_preserves_volume_and_reduces_skew(self):
        base_cell = np.diag([4.0, 4.0, 4.0])
        skew_u = np.array([[1, 3, 2], [0, 1, 1], [0, 0, 1]], dtype=float)
        skew_cell = skew_u @ base_cell
        frac = np.array([[0.0, 0.0, 0.0], [0.25, 0.5, 0.75]])
        red_cell, red_frac = reduce_cell_lll_zz(skew_cell, frac)
        self.assertAlmostEqual(abs(np.linalg.det(red_cell)), abs(np.linalg.det(skew_cell)), places=6)
        self.assertLess(np.linalg.cond(red_cell), np.linalg.cond(skew_cell))
        self.assertEqual(red_frac.shape, frac.shape)
        self.assertTrue(np.all((red_frac >= 0.0) & (red_frac < 1.0)))


class HullTests(unittest.TestCase):
    def setUp(self):
        self.entries = [dict(formula=f, energy_per_atom=e, energy_id='test')
                        for f, e in [('Li', 0), ('O', 0), ('LiO', -1)]]
        self.hull = ReferenceHull(self.entries, 'test')

    def energy(self, formula, energy):
        return self.hull.above_hull(formula, energy, energy_id='test')

    def test_hull_known_mixtures_and_cache(self):
        self.assertAlmostEqual(self.energy('LiO', -.8), .2)
        self.assertAlmostEqual(self.energy('Li3O', -.4), .1)
        self.assertAlmostEqual(self.energy('LiO', -1.1), 0)
        self.assertEqual(len(self.hull._facets), 1)
        self.assertAlmostEqual(self.energy('Li', .2), .2)

    def test_energy_scale_missing_element_and_nan(self):
        with self.assertRaises(ValueError):
            self.hull.above_hull('LiO', -.8, energy_id='different')
        with self.assertRaises(ValueError):
            self.energy('LiS', -.8)
        with self.assertRaises(ValueError):
            self.energy('LiO', float('nan'))
        with self.assertRaises(ValueError):
            ReferenceHull(self.entries, 'different')

    def test_matches_independent_pymatgen_phase_diagram(self):
        from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
        entries = self.entries + [dict(formula=f, energy_per_atom=e, energy_id='test')
                                  for f, e in [('Na', 0), ('NaO', -.7), ('LiNaO2', -1.1)]]
        hull = ReferenceHull(entries, 'test')
        reference = PhaseDiagram([PDEntry(e['formula'], e['energy_per_atom'] *
                                  __import__('pymatgen.core', fromlist=['Composition']).Composition(e['formula']).num_atoms)
                                  for e in entries])
        for formula in ['LiNaO2', 'Li2NaO3', 'LiNa2O', 'NaO', 'Li3O']:
            from pymatgen.core import Composition
            composition = Composition(formula)
            # Deliberately above the hull so the nonnegative convention agrees.
            energy = .2
            candidate = PDEntry(composition, energy*composition.num_atoms)
            expected = reference.get_e_above_hull(candidate)
            actual = hull.above_hull(formula, energy, energy_id='test')
            self.assertAlmostEqual(actual, expected, places=10)


if __name__ == '__main__':
    unittest.main()
