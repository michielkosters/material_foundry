import csv
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from generator_bridge import GeneratorBridge
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


class GeneratorTests(unittest.TestCase):
    def test_seed_reproducibility_stoichiometry_symmetry_and_serialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'train.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['pretty_formula', 'spacegroup.number', 'cif', 'material_id'])
                writer.writeheader()
                writer.writerow({'pretty_formula': 'NaCl', 'spacegroup.number': 225,
                                 'cif': '', 'material_id': 'synthetic-fixture'})
            generator = GeneratorBridge(path, ['Na', 'Cl'])
            first, metadata, reason = generator.propose(123)
            second, _, _ = generator.propose(123)
            self.assertEqual(reason, 'OK')
            self.assertLessEqual(len(first), 20)
            self.assertEqual(first.composition.reduced_formula, 'NaCl')
            self.assertTrue(np.allclose(first.lattice.matrix, second.lattice.matrix))
            self.assertTrue(np.allclose(first.frac_coords, second.frac_coords))
            self.assertEqual(SpacegroupAnalyzer(first, symprec=.01).get_space_group_number(), 225)
            json.dumps(metadata, allow_nan=False)
            self.assertEqual(sum(w['multiplicity'] for w in metadata['wyckoff']), metadata['conventional_atoms'])

    def test_discover_mode_isovalent_substitution_creates_novel_composition(self):
        from ase.build import bulk
        from pymatgen.io.ase import AseAtomsAdaptor
        rocksalt = AseAtomsAdaptor.get_structure(bulk('NaCl', 'rocksalt', a=5.64, cubic=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'train.csv'
            with path.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['pretty_formula', 'spacegroup.number', 'cif', 'material_id'])
                writer.writeheader()
                writer.writerow({'pretty_formula': 'NaCl', 'spacegroup.number': 225,
                                 'cif': rocksalt.to(fmt='cif'), 'material_id': 'mp-nacl'})
            generator = GeneratorBridge(path, ['Na', 'K', 'Cl', 'Br'])
            structure, metadata, reason = generator.propose(42, prior='discover')
            self.assertEqual(reason, 'OK')
            self.assertIsNotNone(structure)
            self.assertNotEqual(structure.composition.reduced_formula, 'NaCl')
            self.assertTrue(metadata.get('formula_novel_vs_mp20'))


if __name__ == '__main__':
    unittest.main()
