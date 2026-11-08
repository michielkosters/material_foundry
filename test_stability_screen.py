import unittest
from pymatgen.core import Composition
from stability_screen import TabulatedHull


class StabilityHullTests(unittest.TestCase):
    def setUp(self):
        self.hull = TabulatedHull([
            dict(immutable_id='Li', nsites=1, species_at_sites=['Li'], mace_mp_energy=0),
            dict(immutable_id='O', nsites=2, species_at_sites=['O', 'O'], mace_mp_energy=0),
            dict(immutable_id='LiO', nsites=2, species_at_sites=['Li', 'O'], mace_mp_energy=-2)])

    def test_total_to_per_atom_and_competing_mixtures(self):
        row = self.hull.evaluate(Composition('LiO'), -.8)
        self.assertAlmostEqual(row['energy_above_reference_hull_eV_atom'], .2)
        self.assertFalse(row['within_0p1'])
        self.assertAlmostEqual(self.hull.evaluate(Composition('Li3O'), -.4)['energy_above_reference_hull_eV_atom'], .1)

    def test_signed_energy_and_missing_coverage(self):
        row = self.hull.evaluate(Composition('LiO'), -1.1)
        self.assertAlmostEqual(row['signed_decomposition_energy_eV_atom'], -.1)
        self.assertTrue(row['on_or_below_reference_hull'])
        with self.assertRaises(ValueError):
            self.hull.evaluate(Composition('LiS'), -.8)


if __name__ == '__main__':
    unittest.main()
