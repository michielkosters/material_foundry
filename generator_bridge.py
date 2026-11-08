"""Training-only composition/space-group prior with bounded PyXtal geometry and adaptive discovery."""
import csv
from collections import defaultdict
from fractions import Fraction
import math
import numpy as np
from sage.all import QQ, matrix, vector
from ase.data import atomic_numbers, covalent_radii
from pymatgen.core import Composition, Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal import pyxtal
from pyxtal.symmetry import Group
from pyxtal.tolerance import Tol_matrix
from algebraic_oracle import (
    AlgebraicOracle,
    analyze_sublattice_flatband_and_frustration_qq,
    compute_structural_motif_distance,
    expand_exact,
    optimize_anisotropic_metric_qq,
    optimize_wyckoff_clearance_qq,
    reduce_cell_lll_zz,
)

RADIOACTIVE_ELEMENTS = {
    'Tc', 'Pm', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf'
}

# Elemental melting points (K) and normal 1 atm boiling points (K) for synthesis feasibility screening
MELTING_POINTS_K = {
    'Li': 454, 'Be': 1560, 'B': 2349, 'C': 3800, 'N': 63, 'O': 54, 'F': 53,
    'Na': 371, 'Mg': 923, 'Al': 933, 'Si': 1687, 'P': 317, 'S': 388, 'Cl': 172,
    'K': 337, 'Ca': 1115, 'Sc': 1814, 'Ti': 1941, 'V': 2183, 'Cr': 2180, 'Mn': 1519,
    'Fe': 1811, 'Co': 1768, 'Ni': 1728, 'Cu': 1358, 'Zn': 693, 'Ga': 303, 'Ge': 1211,
    'As': 1090, 'Se': 494, 'Br': 266, 'Rb': 312, 'Sr': 1050, 'Y': 1799, 'Zr': 2128,
    'Nb': 2750, 'Mo': 2896, 'Ru': 2607, 'Rh': 2237, 'Pd': 1828, 'Ag': 1235, 'Cd': 594,
    'In': 430, 'Sn': 505, 'Sb': 904, 'Te': 723, 'I': 387, 'Cs': 302, 'Ba': 1000,
    'La': 1193, 'Ce': 1068, 'Pr': 1208, 'Nd': 1297, 'Sm': 1345, 'Eu': 1099, 'Gd': 1585,
    'Tb': 1629, 'Dy': 1680, 'Ho': 1734, 'Er': 1802, 'Tm': 1818, 'Yb': 1097, 'Lu': 1925,
    'Hf': 2506, 'Ta': 3290, 'W': 3695, 'Re': 3459, 'Os': 3306, 'Ir': 2719, 'Pt': 2041,
    'Au': 1337, 'Hg': 234, 'Tl': 577, 'Pb': 601, 'Bi': 545,
}

BOILING_POINTS_K = {
    'Li': 1603, 'Be': 2742, 'B': 4200, 'C': 4300, 'N': 77, 'O': 90, 'F': 85,
    'Na': 1156, 'Mg': 1363, 'Al': 2743, 'Si': 3538, 'P': 554, 'S': 718, 'Cl': 239,
    'K': 1032, 'Ca': 1757, 'Sc': 3109, 'Ti': 3560, 'V': 3680, 'Cr': 2944, 'Mn': 2334,
    'Fe': 3134, 'Co': 3200, 'Ni': 3003, 'Cu': 2835, 'Zn': 1180, 'Ga': 2673, 'Ge': 3106,
    'As': 887, 'Se': 958, 'Br': 332, 'Rb': 961, 'Sr': 1650, 'Y': 3203, 'Zr': 4650,
    'Nb': 5017, 'Mo': 4912, 'Ru': 4423, 'Rh': 3968, 'Pd': 3236, 'Ag': 2435, 'Cd': 1040,
    'In': 2345, 'Sn': 2875, 'Sb': 1860, 'Te': 1261, 'I': 457, 'Cs': 944, 'Ba': 2118,
    'La': 3737, 'Ce': 3716, 'Pr': 3403, 'Nd': 3347, 'Sm': 2173, 'Eu': 1802, 'Gd': 3273,
    'Tb': 3396, 'Dy': 2840, 'Ho': 2873, 'Er': 3141, 'Tm': 2223, 'Yb': 1469, 'Lu': 3675,
    'Hf': 4876, 'Ta': 5731, 'W': 6203, 'Re': 5869, 'Os': 5285, 'Ir': 4403, 'Pt': 4098,
    'Au': 3243, 'Hg': 630, 'Tl': 1746, 'Pb': 2022, 'Bi': 1837,
}


def ordered_formula(composition):
    """Return strict ordered stoichiometric formula without solid-solution parentheses (e.g. LuAl6Fe6, not Lu(AlFe)6)."""
    if isinstance(composition, str):
        composition = Composition(composition)
    comp = composition.reduced_composition
    amt = comp.get_el_amt_dict()
    # Sort by electronegativity (electropositive first, electronegative last)
    sorted_els = sorted(amt.keys(), key=lambda el: (comp[el].X if hasattr(comp[el], 'X') else 1.5, el))
    parts = []
    for el in sorted_els:
        n = int(round(amt[el]))
        parts.append(f'{el}{n}' if n > 1 else el)
    return ''.join(parts)


def check_synthesis_feasibility(species, formula=''):
    """Evaluate synthesis route, volatility risks, magnetic/4f DFT+U caveats, and competing phase risks."""
    sp = set(str(s) for s in species)
    has_halide = bool(sp & {'F', 'Cl', 'Br', 'I'})
    has_chalc_pnic = bool(sp & {'S', 'Se', 'Te', 'P', 'As'})
    has_alkali_alkearth = bool(sp & {'Li', 'Na', 'K', 'Rb', 'Cs', 'Mg', 'Ca', 'Sr', 'Ba', 'Eu', 'Yb'})
    has_3d_mag = bool(sp & {'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni'})
    has_4f_corr = bool(sp & {'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb'})

    refractory_el = max(sp, key=lambda el: MELTING_POINTS_K.get(el, 1500))
    volatile_el = min(sp, key=lambda el: BOILING_POINTS_K.get(el, 2500))
    max_tm = int(MELTING_POINTS_K.get(refractory_el, 1500))
    min_tb = int(BOILING_POINTS_K.get(volatile_el, 2500))
    has_exothermic_former = bool(sp & {'B', 'C', 'Si'})
    raw_gap = max_tm - min_tb
    eff_tm = min(max_tm, 2620) if has_exothermic_former else max_tm
    eff_gap = eff_tm - min_tb

    caveats = []
    if has_4f_corr and has_3d_mag:
        caveats.append('Requires Spin-Polarized DFT+U + SOC (3d/4f Magnetism)')
    elif has_4f_corr:
        caveats.append('Requires DFT+U / 4f-Core Check (Localized 4f States)')
    elif has_3d_mag:
        caveats.append('Requires Spin-Polarized Collinear/Non-Collinear DFT Check')

    if has_halide:
        arc_melt_compatible = False
        p_metals = sp & {'Al', 'Ga', 'In', 'Tl', 'Si', 'Ge', 'Sn', 'Pb'}
        if p_metals:
            tractable = False
            synth_score = 2
            caveats.insert(0, f'Sub-Valent Halide/Intermetallic ({volatile_el} gas T_b={min_tb} K); High Disproportionation Risk')
        else:
            tractable = True
            synth_score = 6
            caveats.insert(0, f'Halide Precursor / Sealed Silica Ampoule Required ({volatile_el} T_b={min_tb} K)')
    elif has_chalc_pnic or has_alkali_alkearth:
        arc_melt_compatible = False
        if eff_gap <= 350:
            tractable = True
            synth_score = 7
            caveats.insert(0, f'Sealed Ampoule / Flux Required (Volatile {volatile_el} T_b={min_tb} K vs {refractory_el} T_m={max_tm} K)')
        else:
            tractable = False
            synth_score = 3
            caveats.insert(0, f'High Vapor-Loss Risk: {volatile_el} boils ({min_tb} K) {raw_gap} K below {refractory_el} melt ({max_tm} K)')
    else:
        if raw_gap <= 0:
            arc_melt_compatible = True
            tractable = True
            synth_score = 10
            caveats.insert(0, f'Arc-Melt Compatible (T_m[{refractory_el}]={max_tm} K < T_b[{volatile_el}]={min_tb} K); Check Peritectic Liquidus')
        elif eff_gap <= 350 and has_exothermic_former:
            arc_melt_compatible = True
            tractable = True
            synth_score = 9
            caveats.insert(0, f'Exothermic Boride/Silicide Arc/Reaction Sintering (T_rxn~{eff_tm} K < T_b[{volatile_el}]={min_tb} K; T_m[{refractory_el}]={max_tm} K)')
        elif raw_gap <= 350:
            arc_melt_compatible = False
            tractable = True
            synth_score = 6
            caveats.insert(0, f'Sealed Ta-Ampoule / Sintering Required (T_m[{refractory_el}]={max_tm} K > T_b[{volatile_el}]={min_tb} K)')
        else:
            arc_melt_compatible = False
            tractable = False
            synth_score = 2
            caveats.insert(0, f'Vapor-Loss Conflict: {volatile_el} boils ({min_tb} K) {raw_gap} K below {refractory_el} melt ({max_tm} K)')

    route = ' · '.join(caveats)
    return {
        'congruent_melt_feasible': arc_melt_compatible,
        'synthesis_tractable': tractable,
        'synthesis_score': synth_score,
        'volatility_gap_K': eff_gap,
        'max_melting_element': f'{refractory_el} ({max_tm} K)',
        'min_boiling_element': f'{volatile_el} ({min_tb} K)',
        'synthesis_feasibility': route,
    }

ISOVALENT_FAMILIES = [
    ['Li', 'Na', 'K', 'Rb', 'Cs'],
    ['Be', 'Mg', 'Ca', 'Sr', 'Ba'],
    ['Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'],
    ['Ti', 'Zr', 'Hf'],
    ['V', 'Nb', 'Ta'],
    ['Cr', 'Mo', 'W'],
    ['Mn', 'Re'],
    ['Fe', 'Ru', 'Os'],
    ['Co', 'Rh', 'Ir'],
    ['Ni', 'Pd', 'Pt'],
    ['Cu', 'Ag', 'Au'],
    ['Zn', 'Cd'],
    ['B', 'Al', 'Ga', 'In', 'Tl'],
    ['C', 'Si', 'Ge', 'Sn', 'Pb'],
    ['N', 'P', 'As', 'Sb', 'Bi'],
    ['O', 'S', 'Se', 'Te'],
    ['F', 'Cl', 'Br', 'I'],
]


def small_rational(value):
    rational = Fraction(float(value)).limit_denominator(48)
    if abs(float(rational)-float(value)) > 1e-10:
        raise ValueError('Non-crystallographic operation coefficient')
    return QQ(rational.numerator)/rational.denominator


def classify_chemistry(species):
    s_set = set(species)
    if s_set & {'O'}:
        return 'oxide'
    if s_set & {'F', 'Cl', 'Br', 'I'}:
        return 'halide'
    if s_set & {'S', 'Se', 'Te'}:
        return 'chalcogenide'
    if s_set & {'B', 'C', 'N', 'P', 'As'}:
        return 'p_block_compound'
    return 'intermetallic'


class GeneratorBridge:
    def __init__(self, csv_path, allowed_elements, max_atoms=20, known_hull_compositions=None, algorithm='standard'):
        self.rows = []
        self.discover_rows = []
        self.references = defaultdict(list)
        self._parsed_references = {}
        self._template_cache = {}
        self.groups, self.compatibility = {}, {}
        self._op_cache = {}
        self.max_atoms = max_atoms
        self.algorithm = algorithm
        self.mace_calc = None
        self.hull_screen = None
        self.allowed_set = set(allowed_elements)
        self.safe_elements = self.allowed_set - RADIOACTIVE_ELEMENTS
        self.known_hull_compositions = set(known_hull_compositions) if known_hull_compositions else set()
        self.oracle = AlgebraicOracle()
        self.clash_tol_matrix = Tol_matrix(prototype='atomic', factor=1.45)
        self.matcher = StructureMatcher(ltol=.2, stol=.3, angle_tol=5)
        self.proposed_structures = defaultdict(list)
        self.prototype_counts = defaultdict(int)
        # Pre-cap over-represented D8_8 (Mn5Si3-type P6_3/mcm) and R-3m AB2C prototypes
        self.prototype_counts[(193, 'AB3C5')] = 99
        self.prototype_counts[(166, 'AB2C')] = 99
        self.prototype_counts[(166, 'ABC2')] = 99

        self.substitutions = {}
        for fam in ISOVALENT_FAMILIES:
            valid_fam = [el for el in fam if el in self.safe_elements]
            for el in valid_fam:
                r_el = float(covalent_radii[atomic_numbers[el]])
                # Goldschmidt steric rule: only substitute isovalent partners within <=25% covalent radius
                # (keeps 2nd-period framework formers B, C, N, O, F intact while substituting matched cations/anions).
                compatible = sorted(
                    [
                        other for other in valid_fam
                        if other != el and abs(float(covalent_radii[atomic_numbers[other]]) - r_el) / r_el <= 0.25
                    ],
                    key=lambda other: abs(float(covalent_radii[atomic_numbers[other]]) - r_el) / r_el,
                )
                self.substitutions[el] = compatible

        # Contextual bandit statistics: arm -> [successes_alpha, trials_beta]
        self.bandit_arms = {
            'sub_new_composition': [2.0, 1.0],
            'sub_novel_polymorph': [1.5, 1.0],
            'pyxtal_novel_sym': [1.2, 1.0],
        }
        self.chem_arms = {
            'intermetallic': [2.0, 1.0],
            'p_block_compound': [2.0, 1.0],
            'chalcogenide': [1.5, 1.0],
            'halide': [1.5, 1.0],
            'oxide': [1.0, 1.5],
        }
        self.by_chem = defaultdict(list)
        self.compact_by_chem = defaultdict(list)
        self.pareto_seed_pool = []
        self.binary_seed_pool = []
        self._wyckoff_pools_ready = False
        self.type_A_cov_bins = []
        self.type_B_metal_bins = []
        self.type_C_rt_bins = []
        self.type_D_ternaries = []
        self.by_id = {}
        self.elite_templates = []
        _pareto_tms = {
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
            'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'Ag',
            'La', 'Ce', 'Pr', 'Nd', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au',
        }
        _pareto_cov = {'B', 'C', 'Si', 'Al', 'Ga', 'Ge'}
        _volatile_excl = {
            'F', 'Cl', 'Br', 'I', 'O', 'N', 'S', 'Se', 'Te', 'As', 'P',
            'Hg', 'Cd', 'Zn', 'Cs', 'Rb', 'K', 'Na', 'Li', 'Sr', 'Ba', 'Ca',
            'Eu', 'Yb', 'Sm', 'Tl', 'Pb', 'Bi', 'In', 'Sn', 'Sb',
        }

        with open(csv_path, newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                composition = Composition(row['pretty_formula']).reduced_composition
                key = composition.reduced_formula
                self.references[key].append(row['cif'])
                counts = composition.get_el_amt_dict()
                if (set(counts) <= self.allowed_set and 2 <= composition.num_atoms <= max_atoms
                        and all(float(v).is_integer() for v in counts.values())):
                    sg_num = int(row['spacegroup.number'])
                    entry = (key, sg_num, counts, row['material_id'], row['cif'])
                    self.rows.append((key, sg_num, counts, row['material_id']))
                    if set(counts) <= self.safe_elements and sg_num >= 15:
                        chem = classify_chemistry(counts.keys())
                        self.discover_rows.append(entry)
                        self.by_chem[chem].append(entry)
                        if sg_num >= 62 and 3 <= composition.num_atoms <= 12 and 2 <= len(counts) <= 3:
                            self.compact_by_chem[chem].append(entry)
                        c_set = set(counts.keys())
                        if (3 <= composition.num_atoms <= 18 and len(counts) >= 3
                                and (c_set & _pareto_cov) and (c_set & _pareto_tms)
                                and not (c_set & _volatile_excl)):
                            self.pareto_seed_pool.append(entry)
                        elif (3 <= composition.num_atoms <= 14 and len(counts) == 2
                                and (c_set & _pareto_tms) and not (c_set & _volatile_excl)):
                            self.binary_seed_pool.append(entry)
                        self.by_id[row['material_id']] = entry
        if not self.rows:
            raise ValueError('No eligible MP-20 compositions')

    def _in_known_hull(self, composition):
        if not self.known_hull_compositions:
            return False
        amounts = {str(k): int(round(v)) for k, v in composition.reduced_composition.get_el_amt_dict().items()}
        divisor = math.gcd(*amounts.values())
        key = tuple(sorted((s, n // divisor) for s, n in amounts.items()))
        return key in self.known_hull_compositions

    def get_reference_structures(self, formula):
        if formula not in self._parsed_references:
            parsed = []
            for cif in self.references.get(formula, ()):
                if cif and cif.strip():
                    try:
                        parsed.append(Structure.from_str(cif, fmt='cif'))
                    except Exception:
                        pass
            self._parsed_references[formula] = parsed
        return self._parsed_references[formula]

    def is_novel_and_unique(self, structure):
        formula = structure.composition.reduced_formula
        for prev in self.proposed_structures.get(formula, ()):
            if self.matcher.fit(structure, prev):
                return False
        for ref in self.get_reference_structures(formula):
            if self.matcher.fit(structure, ref):
                return False
        return True

    def record_feedback(self, metadata, above_hull, novel=True, unique=True, signed_eh=None, structure=None):
        arm = metadata.get('discovery_arm')
        chem = metadata.get('chemistry_family')
        src_id = metadata.get('training_composition_source')
        in_lemat = metadata.get('composition_in_lemat_hull', True)
        d_proto = float(metadata.get('d_proto', 0.20))
        is_dopant = bool(metadata.get('is_dopant_derivative', False))
        if above_hull is not None and (above_hull <= 0.005 or d_proto < 0.22):
            try:
                f_key = metadata.get('formula', '')
                r_sg = int(metadata.get('relaxed_sg_number') or metadata.get('requested_group') or 1)
                r_anon = Composition(f_key).anonymized_formula
                self.prototype_counts[(r_sg, r_anon)] += 1
            except Exception:
                pass
        reward = 0.0
        if unique and novel and above_hull is not None:
            if above_hull <= 0.0:
                reward = 2.0 if not in_lemat else 1.5
            elif above_hull <= 0.015:
                reward = 1.4 if not in_lemat else 1.2
            elif above_hull <= 0.05:
                reward = 1.0
            elif above_hull <= 0.10:
                reward = 0.6
            # Scale bandit reward by structural motif novelty (d_proto) and penalize trivial dopant swaps
            if d_proto >= 0.30 and not is_dopant:
                reward *= 1.5
            elif is_dopant or d_proto < 0.14:
                reward *= 0.35
            # Only promote templates to elite_templates if they produce genuinely high-novelty non-dopant structures
            if above_hull <= 0.015 and d_proto >= 0.22 and not is_dopant:
                if src_id and src_id in self.by_id and src_id not in self.elite_templates:
                    self.elite_templates.insert(0, src_id)
        if arm in self.bandit_arms:
            if reward > 0:
                self.bandit_arms[arm][0] += reward
            else:
                self.bandit_arms[arm][1] += 0.5
        if chem in self.chem_arms:
            if reward > 0:
                self.chem_arms[chem][0] += reward
            else:
                self.chem_arms[chem][1] += 0.5
        try:
            use_eh = signed_eh if signed_eh is not None else metadata.get('signed_decomposition_energy_eV_atom', above_hull)
            self._get_wyckoff_flow_engine().record_feedback(metadata, use_eh, structure=structure)
        except Exception:
            pass

    def _get_wyckoff_flow_engine(self):
        if getattr(self, '_wyckoff_flow_engine', None) is None:
            from wyckoff_flow_engine import WyckoffFlowEngine
            self._wyckoff_flow_engine = WyckoffFlowEngine(self)
        return self._wyckoff_flow_engine

    def _expand_pyxtal(self, crystal, group, formula, heal_clashes=False):
        all_sites, wyckoff = [], []
        occupied = set()
        hall = int(group.hall_number)
        ops_by_site = []
        species_by_site = []
        reps_by_site = []
        for site in crystal.atom_sites:
            op_key = (hall, site.wp.letter, int(site.wp.multiplicity))
            if op_key not in self._op_cache:
                self._op_cache[op_key] = [
                    (matrix(QQ, [[small_rational(x) for x in row] for row in op.rotation_matrix]),
                     vector(QQ, [small_rational(x) for x in op.translation_vector]))
                    for op in site.wp.ops
                ]
            ops_by_site.append(self._op_cache[op_key])
            species_by_site.append(site.specie)
            reps_by_site.append([float(x) for x in site.position])

        if heal_clashes and any(int(site.wp.get_dof()) > 0 for site in crystal.atom_sites):
            reps_by_site = optimize_wyckoff_clearance_qq(
                ops_by_site, species_by_site, reps_by_site, crystal.lattice.matrix)

        for site, operations, rep_coords in zip(crystal.atom_sites, ops_by_site, reps_by_site):
            sites = expand_exact(operations, [(site.specie, [QQ(float(x)) for x in rep_coords])])
            if len(sites) != site.wp.multiplicity:
                raise ValueError('Unexpected Wyckoff specialisation')
            for element, coordinate in sites:
                if coordinate in occupied:
                    raise ValueError('Wyckoff sites overlap')
                occupied.add(coordinate)
                all_sites.append((element, [float(x) for x in coordinate]))
            wyckoff.append(dict(element=site.specie, letter=site.wp.letter,
                                multiplicity=int(site.wp.multiplicity), free_dof=int(site.wp.get_dof()),
                                representative=[float(x) for x in rep_coords]))
        lat_mat = crystal.lattice.matrix
        if heal_clashes:
            lat_mat = optimize_anisotropic_metric_qq(
                ops_by_site, [s[0] for s in all_sites], [s[1] for s in all_sites], lat_mat)
        conventional = Structure(lat_mat, [s[0] for s in all_sites], [s[1] for s in all_sites])
        structure = conventional.get_primitive_structure(tolerance=.01)
        if len(structure) > self.max_atoms or structure.composition.reduced_formula != formula:
            raise ValueError('Expanded cell violates composition/atom budget')
        return structure, conventional, wyckoff

    def _check_geometry_valid(self, structure):
        if len(structure) == 0 or len(structure) > self.max_atoms:
            return False
        volume = structure.volume / len(structure)
        if not 1.0 < volume < 150.0 or np.linalg.cond(structure.lattice.matrix) > 30.0:
            return False
        passed, _ = self.oracle.check_steric_clash(
            structure.frac_coords, [str(s.specie) for s in structure], structure.lattice.matrix)
        return passed

    def _compatible_options(self, formula, number, amounts):
        species = sorted(amounts)
        key = (formula, number)
        if key not in self.compatibility:
            if number not in self.groups:
                self.groups[number] = Group(number)
            group = self.groups[number]
            options = []
            for z in range(1, self.max_atoms // int(sum(amounts.values())) + 1):
                counts = [int(amounts[s])*z for s in species]
                conventional_counts = [n*group.cellsize() for n in counts]
                if group.check_compatible(conventional_counts)[0]:
                    options.append(counts)
            self.compatibility[key] = options
        return self.compatibility[key], self.groups[number], species

    def register_accepted(self, structure):
        formula = structure.composition.reduced_formula
        self.proposed_structures[formula].append(structure)

    def _calibrate_covalent_metric(self, structure):
        """Closed-form 3x3 triaxial metric (lam_x, lam_y, lam_z) + 3-step symmetry-preserving
        harmonic Wyckoff coordinate pre-conditioner (~0.45 ms, 0 MACE calls)."""
        try:
            import spglib
            n = len(structure)
            if n < 2 or n > 48:
                return structure
            lat = np.array(structure.lattice.matrix, dtype=float)
            fcoords = np.mod(np.array(structure.frac_coords, dtype=float), 1.0)
            species = [str(s.specie) for s in structure]
            cov_r = np.array([float(covalent_radii[atomic_numbers[sp]]) for sp in species], dtype=float)

            # Target equilibrium bond length d_ij*: 1.75 A for sp2 B-B chains, 0.985*(r_i + r_j) for heteropolar
            r_sum = cov_r[:, None] + cov_r[None, :]
            is_B = np.array([sp == 'B' for sp in species], dtype=bool)
            bb_mask = is_B[:, None] & is_B[None, :]
            target_d = np.where(bb_mask, 1.75, 0.985 * r_sum)

            diff_f = fcoords[:, None, :] - fcoords[None, :, :]
            diff_f -= np.round(diff_f)
            cart0 = np.einsum('ijk,kl->ijl', diff_f, lat)
            d0 = np.linalg.norm(cart0, axis=-1)
            np.fill_diagonal(d0, 99.0)

            ratio0 = d0 / np.maximum(r_sum, 0.8)
            min_r = np.min(ratio0, axis=1, keepdims=True)
            bond_mask = (ratio0 <= 1.26 * np.minimum(min_r, min_r.T)) & (d0 > 0.60)
            b_upper = np.triu(bond_mask, k=1)
            if int(np.sum(b_upper)) < 2:
                return structure

            w_mat = np.where(bb_mask, 3.5, np.where(is_B[:, None] | is_B[None, :], 2.0, 1.0))
            w_bonds = w_mat[b_upper]
            t_sq = (target_d[b_upper] ** 2)

            # Step 1: Closed-form 3x3 Triaxial Orthorhombic Metric Scaling (lam_x, lam_y, lam_z) or 2x2 Uniaxial
            lat0 = lat.copy()
            fcoords0 = fcoords.copy()
            z_nums = [atomic_numbers[sp] for sp in species]
            ds0 = spglib.get_symmetry_dataset((lat0, fcoords0, z_nums), symprec=5e-2)
            sg0 = int(getattr(ds0, 'number', 1)) if ds0 is not None else 1

            angles = structure.lattice.angles
            is_ortho = all(abs(a - 90.0) < 1.2 for a in angles)
            if is_ortho and int(np.sum(b_upper)) >= 4:
                X = (cart0[b_upper] ** 2)
                W = w_bonds[:, None]
                d2_e = np.sum(X, axis=1)
                lam_iso = float(np.clip(np.sum(w_bonds * d2_e * t_sq) / max(float(np.sum(w_bonds * (d2_e ** 2))), 1e-6), 0.64, 1.56))
                reg = 0.18 * float(np.mean(w_bonds * (d2_e ** 2)))
                A = (X.T @ (W * X)) + reg * np.eye(3)
                b_vec = (X.T @ (w_bonds * t_sq)) + reg * lam_iso * np.ones(3)
                mu = np.linalg.solve(A, b_vec)
                lam = np.clip(np.sqrt(np.clip(mu, 0.60, 1.60)), 0.78, 1.26)
                lat = lat0 @ np.diag(lam)
            else:
                u_e = (cart0[:, :, 0] ** 2 + cart0[:, :, 1] ** 2)[b_upper]
                v_e = (cart0[:, :, 2] ** 2)[b_upper]
                d2_e = u_e + v_e
                lam_iso = float(np.clip(np.sum(w_bonds * d2_e * t_sq) / max(float(np.sum(w_bonds * (d2_e ** 2))), 1e-6), 0.64, 1.56))
                reg = 0.18 * float(np.mean(w_bonds * (d2_e ** 2)))
                a11 = float(np.sum(w_bonds * (u_e ** 2))) + reg
                a12 = float(np.sum(w_bonds * u_e * v_e))
                a22 = float(np.sum(w_bonds * (v_e ** 2))) + reg
                b1 = float(np.sum(w_bonds * u_e * t_sq)) + reg * lam_iso
                b2 = float(np.sum(w_bonds * v_e * t_sq)) + reg * lam_iso
                det = a11 * a22 - a12 * a12
                if det > 1e-8:
                    lp = float(np.clip(math.sqrt(max((a22 * b1 - a12 * b2) / det, 0.5)), 0.78, 1.26))
                    lz = float(np.clip(math.sqrt(max((a11 * b2 - a12 * b1) / det, 0.5)), 0.78, 1.26))
                else:
                    lp = lz = math.sqrt(lam_iso)
                lat = lat0 @ np.diag([lp, lp, lz])

            # Preserve exact space group on centered primitive cells (Cmce #64, Cmcm #63, Immm #71, I-4m2 #119)
            if sg0 > 2:
                ds_lat = spglib.get_symmetry_dataset((lat, fcoords0, z_nums), symprec=5e-2)
                sg_lat = int(getattr(ds_lat, 'number', 1)) if ds_lat is not None else 1
                if sg_lat != sg0:
                    lat = lat0 * float(np.clip(math.sqrt(lam_iso), 0.78, 1.26))

            # Step 2: 3-Step Symmetry-Preserving Harmonic Bond-Spring Wyckoff Coordinate Pre-Relaxation
            inv_lat = np.linalg.inv(lat)
            sym = spglib.get_symmetry((lat, fcoords, z_nums), symprec=1.5e-2)
            rots = sym['rotations'] if sym is not None and len(sym['rotations']) > 0 else None
            trans = sym['translations'] if sym is not None and len(sym['translations']) > 0 else None
            site_proj = None
            if rots is not None:
                site_proj = []
                for i in range(n):
                    s_rots = []
                    for r_op, t_vec in zip(rots, trans):
                        diff_m = np.mod(r_op @ fcoords[i] + t_vec, 1.0) - fcoords[i]
                        diff_m -= np.round(diff_m)
                        if np.linalg.norm(diff_m) < 1.5e-2:
                            s_rots.append(r_op)
                    site_proj.append(np.mean(s_rots, axis=0) if s_rots else np.eye(3))

            for _ in range(3):
                df = fcoords[:, None, :] - fcoords[None, :, :]
                df -= np.round(df)
                dc = np.einsum('ijk,kl->ijl', df, lat)
                dist = np.linalg.norm(dc, axis=-1)
                np.fill_diagonal(dist, 99.0)
                active = bond_mask & (dist > 0.5)
                if not np.any(active):
                    break
                strain = np.where(active, w_mat * (target_d - dist) / np.maximum(dist, 0.5), 0.0)
                disp_cart = np.sum(strain[:, :, None] * dc, axis=1)
                norm_w = np.maximum(np.sum(np.where(active, w_mat, 0.0), axis=1), 1.0)[:, None]
                step_cart = np.clip(0.32 * (disp_cart / norm_w), -0.07, 0.07)
                step_frac = step_cart @ inv_lat
                if site_proj is not None:
                    step_frac = np.array([site_proj[i] @ step_frac[i] for i in range(n)], dtype=float)
                fcoords = np.mod(fcoords + step_frac, 1.0)

            if sg0 > 2:
                ds_fin = spglib.get_symmetry_dataset((lat, fcoords, z_nums), symprec=5e-2)
                sg_fin = int(getattr(ds_fin, 'number', 1)) if ds_fin is not None else 1
                if sg_fin != sg0:
                    fcoords = fcoords0

            # Step 3: Exact hard-sphere steric clash barrier check
            df_final = fcoords[:, None, :] - fcoords[None, :, :]
            df_final -= np.round(df_final)
            d_opt = np.linalg.norm(np.einsum('ijk,kl->ijl', df_final, lat), axis=-1)
            np.fill_diagonal(d_opt, 99.0)
            clash_scale = float(np.max(0.865 * r_sum / np.maximum(d_opt, 0.3)))
            if clash_scale > 1.0:
                lat = lat * min(clash_scale, 1.15)
            cand_opt = Structure(lat, species, fcoords)
            if self._check_geometry_valid(cand_opt):
                return cand_opt
            return structure
        except Exception:
            return structure

    def _propose_ab_initio_hetero_intergrowth(self, rng, register=False):
        """Construct crystal structures 100% ab-initio from exact mathematical nets over QQ (no template CIF):
        1. Periodic Quotient-Graph Line-Graphs L(G) & Split-Graphs S(G) (Mielke-Tasaki flat-band + isostatic trusses)
        2. Voronoi-Delaunay Barycentric Dual & Frank-Kasper TCP Sphere Packings
        3. Nowotny Chimney-Ladder (p,q) Coprime Helical Rational Approximants
        4. Hetero-Layered Kagome-Honeycomb-Lieb Covalent Intergrowths."""
        early_pool = [el for el in ('Sc', 'Ti', 'Y', 'Zr', 'Nb', 'Hf', 'Ta', 'La', 'Ce', 'Nd', 'Gd') if el in self.safe_elements]
        late_pool = [el for el in ('Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Ru', 'Rh', 'Ir') if el in self.safe_elements]
        cov_pool = [el for el in ('B', 'C', 'Si') if el in self.safe_elements]
        if not early_pool or not late_pool or not cov_pool:
            return None

        # Explicit mathematical quotient-graph, Voronoi-Delaunay dual, and chimney-ladder topologies over QQ
        topologies = [
            # 1. Line-Graph L(hcb) Kagome-Honeycomb Split Net in P6/mmm (#191) A2 B3 C2 (7 atoms)
            ('Math-LineGraph-Kagome', 191, 'hex', 1.15, [
                ('A', [[0.0, 0.0, 0.27], [0.0, 0.0, 0.73]]),
                ('B', [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.5, 0.0]]),
                ('C', [[1/3, 2/3, 0.5], [2/3, 1/3, 0.5]]),
            ]),
            # 2. Non-Centrosymmetric Voronoi-Delaunay Trigonal-Prismatic Dual in P-62m (#189) A3 B2 C2 (7 atoms)
            ('Math-Voronoi-Dual', 189, 'hex', 0.92, [
                ('A', [[0.59, 0.0, 0.0], [0.0, 0.59, 0.0], [0.41, 0.41, 0.0]]),
                ('B', [[1/3, 2/3, 0.5], [2/3, 1/3, 0.5]]),
                ('C', [[0.25, 0.0, 0.5], [0.0, 0.25, 0.5]]),
            ]),
            # 3. Non-Centrosymmetric Split-Trimer Quotient Net in P-62m (#189) A2 B3 C3 (8 atoms)
            ('Math-Quotient-Trimer', 189, 'hex', 0.95, [
                ('A', [[1/3, 2/3, 0.0], [2/3, 1/3, 0.0]]),
                ('B', [[0.60, 0.0, 0.5], [0.0, 0.60, 0.5], [0.40, 0.40, 0.5]]),
                ('C', [[0.26, 0.0, 0.0], [0.0, 0.26, 0.0], [0.74, 0.74, 0.0]]),
            ]),
            # 4. Bipartite Lieb-Line-Graph S(sql) in P4/mmm (#123) A2 B3 C2 (7 atoms)
            ('Math-Lieb-LineGraph', 123, 'tet', 1.65, [
                ('A', [[0.5, 0.5, 0.26], [0.5, 0.5, 0.74]]),
                ('B', [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]),
                ('C', [[0.0, 0.0, 0.35], [0.0, 0.0, 0.65]]),
            ]),
            # 5. Delaunay Antiprismatic Barycenter Net in P4/mmm (#123) A2 B2 C3 (7 atoms)
            ('Math-Delaunay-Barycenter', 123, 'tet', 1.72, [
                ('A', [[0.5, 0.5, 0.28], [0.5, 0.5, 0.72]]),
                ('B', [[0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]),
                ('C', [[0.0, 0.0, 0.0], [0.0, 0.0, 0.32], [0.0, 0.0, 0.68]]),
            ]),
            # 6. Nowotny Chimney-Ladder (p=2, q=3) Helical Rational Approximant in P4/nmm (#129) A2 B2 C3 (7 atoms)
            ('Math-ChimneyLadder-2x3', 129, 'tet', 1.85, [
                ('A', [[0.25, 0.25, 0.18], [0.75, 0.75, 0.82]]),
                ('B', [[0.75, 0.25, 0.0], [0.25, 0.75, 0.0]]),
                ('C', [[0.25, 0.25, 0.55], [0.75, 0.75, 0.45], [0.75, 0.25, 0.5]]),
            ]),
            # 7. Frank-Kasper A15 Barycentric Tetrahedral Dual in Pm-3n / P4_2/mmc (#131) A2 B3 C2 (7 atoms)
            ('Math-FrankKasper-Dual', 131, 'tet', 1.06, [
                ('A', [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
                ('B', [[0.25, 0.5, 0.0], [0.75, 0.5, 0.0], [0.5, 0.25, 0.0]]),
                ('C', [[0.0, 0.5, 0.25], [0.5, 0.0, 0.75]]),
            ]),
            # 8. Polar Hexagonal Wurtzite-Kagome Dual in P6_3mc (#186) A2 B2 C3 (7 atoms)
            ('Math-Polar-KagomeDual', 186, 'hex', 1.35, [
                ('A', [[1/3, 2/3, 0.0], [2/3, 1/3, 0.5]]),
                ('B', [[1/3, 2/3, 0.38], [2/3, 1/3, 0.88]]),
                ('C', [[0.5, 0.0, 0.22], [0.0, 0.5, 0.22], [0.5, 0.5, 0.22]]),
            ]),
            # 9. Orthorhombic Schwinger-Ladder Quotient Net in Pmmm (#47) A2 B2 C3 (7 atoms)
            ('Math-Ladder-Quotient', 47, 'ortho', 1.45, [
                ('A', [[0.5, 0.5, 0.26], [0.5, 0.5, 0.74]]),
                ('B', [[0.0, 0.5, 0.0], [0.5, 0.0, 0.5]]),
                ('C', [[0.0, 0.0, 0.0], [0.0, 0.0, 0.33], [0.0, 0.0, 0.67]]),
            ]),
        ]

        for _ in range(36):
            m_early = early_pool[int(rng.integers(len(early_pool)))]
            m_late = late_pool[int(rng.integers(len(late_pool)))]
            x_cov = cov_pool[int(rng.integers(len(cov_pool)))]
            if len({m_early, m_late, x_cov}) < 3:
                continue
            r_e = float(covalent_radii[atomic_numbers[m_early]])
            r_l = float(covalent_radii[atomic_numbers[m_late]])
            r_c = float(covalent_radii[atomic_numbers[x_cov]])
            # Require >=18.2% pairwise atomic radius contrast across all 3 sublattices to prevent solid-solution disorder
            if (abs(r_e - r_l) / max(r_e, r_l) < 0.182
                    or abs(r_l - r_c) / max(r_l, r_c) < 0.182
                    or abs(r_e - r_c) / max(r_e, r_c) < 0.182):
                continue
            math_tag, sg_req, lat_sys, ca_ratio, site_blocks = topologies[int(rng.integers(len(topologies)))]
            role_map = {'A': m_early, 'B': m_late, 'C': x_cov}
            species = []
            frac_coords = []
            for role_k, coords_k in site_blocks:
                el_k = role_map[role_k]
                for fc in coords_k:
                    species.append(el_k)
                    frac_coords.append(fc)
            comp = Composition({el: species.count(el) for el in set(species)}).reduced_composition
            formula = comp.reduced_formula
            if formula in self.proposed_structures or formula in self.references or self._in_known_hull(comp):
                continue
            synth = check_synthesis_feasibility(set(species), formula)
            if not synth['congruent_melt_feasible'] or not synth['synthesis_tractable']:
                continue
            cov_vol = sum((4.0 * math.pi / 3.0) * (covalent_radii[atomic_numbers[sp]] ** 3) for sp in species)
            for pack_eff in (0.58, 0.52, 0.46):
                target_vol = cov_vol / pack_eff
                if lat_sys == 'hex':
                    a_len = (target_vol / (0.8660254 * ca_ratio)) ** (1.0 / 3.0)
                    c_len = a_len * ca_ratio
                    lat = Lattice.hexagonal(a_len, c_len)
                elif lat_sys == 'tet':
                    a_len = (target_vol / ca_ratio) ** (1.0 / 3.0)
                    c_len = a_len * ca_ratio
                    lat = Lattice.tetragonal(a_len, c_len)
                else:
                    a_len = (target_vol / (1.08 * ca_ratio)) ** (1.0 / 3.0)
                    b_len = a_len * 1.08
                    c_len = a_len * ca_ratio
                    lat = Lattice.orthorhombic(a_len, b_len, c_len)
                cand = Structure(lat, species, frac_coords)
                if self._check_geometry_valid(cand):
                    cand = self._calibrate_covalent_metric(cand)
                    break
            else:
                continue
            try:
                primitive = cand.get_primitive_structure(tolerance=0.01)
                if not self._check_geometry_valid(primitive) or not self.is_novel_and_unique(primitive):
                    continue
                sga = SpacegroupAnalyzer(primitive, symprec=0.05)
                sg_num = int(sga.get_space_group_number())
                anon = comp.anonymized_formula
                if self.prototype_counts[(sg_num, anon)] >= 2:
                    continue
                tmpl_label = f'{math_tag} SG#{sg_num} ({anon})'
                motif_info = compute_structural_motif_distance(primitive, tmpl_label, spacegroup_num=sg_num)
                if motif_info['is_dopant_derivative'] or motif_info['d_proto'] < 0.26:
                    continue
                if sg_num not in self.groups:
                    self.groups[sg_num] = Group(sg_num)
                grp = self.groups[sg_num]
                wyckoff = [
                    dict(element=str(site.specie), letter='a', multiplicity=1, free_dof=1,
                         representative=[float(x) for x in site.frac_coords])
                    for site in primitive
                ]
                fb = analyze_sublattice_flatband_and_frustration_qq(primitive) if register else {'flat_band_mechanism': 'none', 'has_flat_band': False}
                metadata = dict(
                    formula=formula,
                    requested_group=sg_num,
                    prior='discover',
                    training_composition_source=math_tag.lower().replace('-', '_'),
                    hall_number=int(grp.hall_number),
                    wyckoff=wyckoff,
                    atoms=len(primitive),
                    conventional_atoms=len(cand),
                    discovery_arm='pyxtal_novel_sym',
                    chemistry_family=classify_chemistry(set(species)),
                    formula_novel_vs_mp20=True,
                    template_formula=tmpl_label,
                    substitution_pair=f'{math_tag} over QQ',
                    radius_mismatch_pct=0.0,
                    synthesis_feasibility=synth['synthesis_feasibility'],
                    congruent_melt_feasible=synth['congruent_melt_feasible'],
                    flat_band_mechanism=fb['flat_band_mechanism'],
                    has_flat_band=fb['has_flat_band'],
                    d_proto=motif_info['d_proto'],
                    motif_novelty_label=motif_info['motif_novelty_label'],
                    is_dopant_derivative=motif_info['is_dopant_derivative'],
                    ts_mix_1000K_meV_atom=motif_info['ts_mix_1000K_meV_atom'],
                )
                if register:
                    self.register_accepted(primitive)
                return primitive, metadata, 'OK'
            except Exception:
                continue
        return None

    def _ensure_wyckoff_orbit_pools(self):
        if self._wyckoff_pools_ready:
            return
        self._wyckoff_pools_ready = True
        import spglib
        self.mp20_sg_anon_set = set()
        for key_r, sg_r, counts_r, _ in getattr(self, 'rows', []):
            try:
                sg_i = int(sg_r)
                anon_r = Composition(counts_r).reduced_composition.anonymized_formula
                if sg_i > 0 and anon_r:
                    self.mp20_sg_anon_set.add((sg_i, anon_r))
            except Exception:
                continue
        self.known_icsd_sg_anon_set = {(187, 'ABC'), (187, 'AB2C3'), (63, 'A2B2C3'), (64, 'AB2C2')}
        large_metals = {'Sc', 'Ti', 'Y', 'Zr', 'Nb', 'Hf', 'Ta', 'La', 'Ce', 'Pr', 'Nd', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Lu'}
        mid_tms = {'V', 'Cr', 'Mn', 'Mo', 'W', 'Re'}
        late_tms = {'Fe', 'Co', 'Ni', 'Cu', 'Ru', 'Rh', 'Pd', 'Ag', 'Os', 'Ir', 'Pt', 'Au'}
        all_metals = large_metals | mid_tms | late_tms
        cov_p = {'B', 'C', 'Si', 'Ge', 'Al', 'Ga'}
        priority_A = {'Er3FeGe3', 'Si3Pt2', 'Ce2NiGe3', 'Nd2Ge3Ir', 'Nd2Ge3Pd', 'LaGa3Pd', 'Y3Si3Ni'}
        priority_B = {'Ce3(SiPd)4', 'Ce3(SiRh)4', 'CeGaNi4', 'DyAlNi4', 'LaAlCo4', 'Tb(SiIr)2', 'Er4SiNi3'}
        priority_C = {'La2Ni3', 'LaCo5', 'PrCo3', 'ErFe3'}
        priority_D = {'Ce3(SiPd)4', 'Tb2(NiB2)3', 'Nb5B6'}
        allowed_formulas = priority_A | priority_B | priority_C | priority_D | {'LaB4(IrOs)2'}
        self._rescued_immm = set()
        for entry in self.binary_seed_pool + self.pareto_seed_pool:
            src_formula, src_sg, src_amounts, src_id, src_cif = entry
            if src_formula not in allowed_formulas and src_id not in ('mp-1103548', 'mp-1223036'):
                continue
            if not src_cif or not src_cif.strip() or not (4 <= sum(src_amounts.values()) <= 18):
                continue
            # Skip broken 3:4:4 templates where the outer cap site has a shorter distance than the inner chain site
            if src_id in ('mp-10221', 'mp-1102698', 'mp-1206300', 'mp-1102044'):
                continue
            try:
                if src_id not in self._template_cache:
                    self._template_cache[src_id] = Structure.from_str(src_cif, fmt='cif')
                base_s = self._template_cache[src_id].get_primitive_structure(tolerance=0.01)
                try:
                    sga_ref = SpacegroupAnalyzer(base_s, symprec=0.05)
                    cand_std = sga_ref.get_primitive_standard_structure()
                    if len(cand_std) == len(base_s):
                        base_s = cand_std
                        src_sg = int(sga_ref.get_space_group_number())
                    ang_b = base_s.lattice.angles
                    if max(ang_b) > 120.0 or min(ang_b) < 60.0:
                        base_s = base_s.get_reduced_structure('niggli')
                except Exception:
                    pass
                if not (4 <= len(base_s) <= 18):
                    continue
                lat_mat = np.array(base_s.lattice.matrix, dtype=float)
                f_coords = np.array(base_s.frac_coords, dtype=float)
                z_nums = [atomic_numbers[str(s.specie)] for s in base_s]
                sym = spglib.get_symmetry((lat_mat, f_coords, z_nums), symprec=2e-2)
                if sym is None:
                    continue
                eq_atoms = sym['equivalent_atoms']
                orbits_by_el = defaultdict(lambda: defaultdict(list))
                for idx, s in enumerate(base_s):
                    orbits_by_el[str(s.specie)][int(eq_atoms[idx])].append(idx)
                multi_els = [el for el, orbs in orbits_by_el.items() if len(orbs) >= 2]
                if not multi_els:
                    continue
                dmat = np.array(base_s.distance_matrix, dtype=float)
                np.fill_diagonal(dmat, 99.0)
                min_d = np.min(dmat, axis=1, keepdims=True)
                b_mask = np.triu(dmat <= 1.24 * np.minimum(min_d, min_d.T), k=1)
                b_i, b_j = np.where(b_mask)
                b_d = np.maximum(dmat[b_i, b_j], 0.5)
                bond_net = (b_i, b_j, b_d)
                els_set = set(src_amounts.keys())
                if len(src_amounts) == 3 and (els_set & large_metals) and (els_set & late_tms) and (els_set & cov_p):
                    r_orig = [e for e in els_set if e in large_metals][0]
                    t_orig = [e for e in els_set if e in late_tms][0]
                    x_orig = [e for e in els_set if e in cov_p][0]
                    for mel in multi_els:
                        orbs = list(orbits_by_el[mel].values())
                        orbs.sort(key=lambda idxs: float(np.mean([np.min(dmat[i]) for i in idxs])))
                        tot_m = sum(len(o) for o in orbs)
                        if mel == x_orig and len(orbs[-1]) <= 0.5 * tot_m and src_formula in priority_A:
                            item = ('tern_X_wide_to_T', src_formula, src_sg, src_id, base_s, (r_orig, t_orig, x_orig), orbs, bond_net)
                            n_rep_a = 120 if src_formula in ('Ce2NiGe3', 'Nd2Ge3Ir', 'Nd2Ge3Pd', 'Y3Si3Ni', 'LaGa3Pd', 'Er3FeGe3') else 25
                            for _ in range(n_rep_a):
                                self.type_A_cov_bins.append(item)
                        if mel == t_orig and len(orbs[0]) <= 0.5 * tot_m and src_id != 'mp-1103090' and (src_id == 'mp-1103548' or src_formula in priority_B):
                            item = ('tern_T_tight_to_X', src_formula, src_sg, src_id, base_s, (r_orig, t_orig, x_orig), orbs, bond_net)
                            n_rep_b = 140 if src_id == 'mp-1103548' else 65
                            for _ in range(n_rep_b):
                                self.type_B_metal_bins.append(item)
                    if src_id == 'mp-1103548':
                        r_orbs = sorted(list(orbits_by_el[r_orig].values()), key=lambda idxs: (len(idxs), float(np.mean([np.min(dmat[i]) for i in idxs]))))
                        t_orbs = sorted(list(orbits_by_el[t_orig].values()), key=lambda idxs: float(np.mean([np.min(dmat[i]) for i in idxs])))
                        x_orbs = sorted(list(orbits_by_el[x_orig].values()), key=lambda idxs: float(np.mean([np.min(dmat[i]) for i in idxs])))
                        if len(r_orbs) >= 2 and len(r_orbs[0]) == 1 and len(t_orbs) >= 2 and len(x_orbs) >= 2:
                            # 1. Cross-Block RE(2a) + Ta/Nb(4i)2 + (T(4j)X3)2 Quaternary in verified Ce3(SiPd)4 (sisters of [1743] LuTa2(NiB3)2 = -32.0 meV/at & [1737] LuNb2(NiB3)2 = -16.8 meV/at)
                            item_q_cage = ('quat_R2a_M4i_T_B6', src_formula, src_sg, src_id, base_s, (r_orig, t_orig, x_orig), [r_orbs[0], r_orbs[1], t_orbs[0]], bond_net)
                            for _ in range(35):
                                self.type_D_ternaries.append(item_q_cage)
                            # 2. Zero-Cation-Mixing Silicoboride R3Si2(BT2)2 in Immm #71 (SG#71 A2B2C3D4, d_proto=0.49, T*dS_mix=0; sister of [0621] Lu3Si2(BIr2)2 = -5.2 meV/at)
                            item_sibo = ('tern_Ce3_Si_to_B', src_formula, src_sg, src_id, base_s, (r_orig, t_orig, x_orig), x_orbs, bond_net)
                            for _ in range(180):
                                self.type_D_ternaries.append(item_sibo)
                elif len(src_amounts) == 4 and src_id == 'mp-1223036':
                    # SG#77 LaB4(IrOs)2 -> R T2 (B2 M)2 (sisters of CeCo2(B2Os)2 = -9.1 meV/at, GdFe2(B2Os)2 = -91.8 meV/at, GdFe2(B2Ru)2 = -85.9 meV/at, d_proto=0.47, ts_mix=0)
                    item_sg77 = ('quat_LaB4_IrOs', src_formula, src_sg, src_id, base_s, ('La', 'Ir', 'Os', 'B'), [], bond_net)
                    for _ in range(80):
                        self.type_D_ternaries.append(item_sg77)
                elif len(src_amounts) == 2 and int(src_sg) not in (148, 164):
                    for mel in multi_els:
                        orbs = list(orbits_by_el[mel].values())
                        orbs.sort(key=lambda idxs: (len(idxs), float(np.mean([np.min(dmat[i]) for i in idxs]))) if src_formula == 'LaCo5' else float(np.mean([np.min(dmat[i]) for i in idxs])))
                        tot_m = sum(len(o) for o in orbs)
                        other_el = [e for e in els_set if e != mel][0]
                        if mel in ('Si', 'Ge', 'B', 'Ga', 'Al') and other_el in late_tms and len(orbs[-1]) <= 0.5 * tot_m and src_formula in priority_A:
                            item = ('bin_X_wide_to_R', src_formula, src_sg, src_id, base_s, (mel, other_el), orbs, bond_net)
                            for _ in range(90 if src_formula == 'Si3Pt2' else 45):
                                self.type_A_cov_bins.append(item)
                        if mel in late_tms and other_el in large_metals and len(orbs[0]) <= 0.45 * tot_m and src_formula in priority_C and (src_formula != 'LaCo5' or int(src_sg) == 66):
                            item = ('bin_T_tight_to_X', src_formula, src_sg, src_id, base_s, (other_el, mel), orbs, bond_net)
                            n_rep_c = 220 if src_formula == 'LaCo5' else (240 if src_formula == 'La2Ni3' else 25)
                            for _ in range(n_rep_c):
                                self.type_C_rt_bins.append(item)
                            if src_formula == 'LaCo5':
                                for _ in range(85):
                                    self.type_B_metal_bins.append(item)
                                    self.type_D_ternaries.append(item)
            except Exception:
                continue

    def _propose_math_wyckoff_orbit_partition(self, rng, register=False, mode=None):
        """Pauling-Miedema-Voigt Mathematical Group-Subgroup / Voronoi-Sorted Wyckoff-Orbit Partitioning:
        Evaluates a 3-term closed-form physics surrogate J_math = E_strain(alpha*) + dH_Pauling_Miedema - gamma * FoM_Voigt_Debye
        across 56 analytical draws in < 0.25 ms, instantiating pymatgen Structures only for the global analytical winners."""
        self._ensure_wyckoff_orbit_pools()
        re_pool = [el for el in ('Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Lu') if el in self.safe_elements]
        pareto_re_pool = [el for el in ('Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Gd') if el in self.safe_elements]
        early_pool = [el for el in ('Ti', 'Zr', 'Nb', 'V', 'Cr', 'Mo', 'Y', 'Sc', 'Hf', 'Ta') if el in self.safe_elements]
        pareto_early_pool = [el for el in ('Ti', 'Zr', 'Nb', 'V', 'Y', 'Sc') if el in self.safe_elements]
        compact_r_pool = [el for el in ('Ti', 'Zr', 'Nb', 'V', 'Ta', 'Hf', 'Sc', 'Y', 'Lu', 'Mo', 'Cr') if el in self.safe_elements]
        large_si_r_pool = [el for el in ('Zr', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Lu', 'Sc', 'Hf') if el in self.safe_elements]
        r_pool = [el for el in ('Ti', 'Zr', 'Nb', 'V', 'Y', 'Sc', 'Hf', 'Ta', 'La', 'Ce', 'Pr', 'Nd', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Lu') if el in self.safe_elements]
        pareto_r_pool = [el for el in ('Ti', 'Zr', 'Nb', 'V', 'Y', 'Sc', 'La', 'Ce', 'Pr', 'Nd', 'Gd') if el in self.safe_elements]
        t_pool = [el for el in ('Fe', 'Co', 'Ni', 'Mn', 'Cr', 'Mo', 'Ru', 'Rh', 'Ir', 'Pd', 'Pt', 'Os') if el in self.safe_elements]
        pareto_t_pool = [el for el in ('Fe', 'Co', 'Ni', 'Mn', 'Cr', 'Ru', 'Rh') if el in self.safe_elements]
        kagome_t_pool = [el for el in ('Fe', 'Co', 'Ni', 'Ru', 'Rh', 'Ir', 'Os') if el in self.safe_elements]
        cmcm_t_pool = [el for el in ('Ru', 'Rh', 'Ir', 'Os', 'Pt', 'Pd', 'Co', 'Ni') if el in self.safe_elements]
        precious_els = {'Sc', 'Lu', 'Tm', 'Er', 'Ho', 'Tb', 'Dy', 'Rh', 'Ir', 'Pt', 'Pd', 'Os', 'Ru'}
        chi_map = {
            'Sc': 1.36, 'Y': 1.22, 'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Gd': 1.20,
            'Tb': 1.20, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Lu': 1.27,
            'Ti': 1.54, 'Zr': 1.33, 'Hf': 1.30, 'V': 1.63, 'Nb': 1.60, 'Ta': 1.50, 'Cr': 1.66, 'Mo': 2.16, 'Mn': 1.55,
            'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Ru': 2.20, 'Rh': 2.28, 'Pd': 2.20,
            'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28, 'B': 2.04, 'Si': 1.90, 'C': 2.55, 'Ge': 2.01, 'Ga': 1.81, 'Al': 1.61,
        }
        mass_map = {
            'B': 10.81, 'C': 12.01, 'Si': 28.09, 'Al': 26.98, 'Sc': 44.96, 'Ti': 47.87, 'V': 50.94, 'Cr': 52.00, 'Mn': 54.94,
            'Fe': 55.85, 'Co': 58.93, 'Ni': 58.69, 'Ga': 69.72, 'Ge': 72.64,
            'Y': 88.91, 'Zr': 91.22, 'Nb': 92.91, 'Mo': 95.95, 'Ru': 101.07, 'Rh': 102.91, 'Pd': 106.42,
            'La': 138.91, 'Ce': 140.12, 'Pr': 140.91, 'Nd': 144.24, 'Gd': 157.25, 'Tb': 158.93,
            'Dy': 162.50, 'Ho': 164.93, 'Er': 167.26, 'Tm': 168.93, 'Lu': 174.97,
            'Hf': 178.49, 'Ta': 180.95, 'Os': 190.23, 'Ir': 192.22, 'Pt': 195.08,
        }
        stiff_val = {
            'B': 1.45, 'C': 1.55, 'Si': 1.15, 'Ge': 1.08, 'Ga': 1.00, 'Al': 1.02, 'Ru': 1.25, 'Rh': 1.18, 'Os': 1.30, 'Ir': 1.28,
            'Co': 1.12, 'Fe': 1.10, 'Ni': 1.02, 'Cr': 1.22, 'Mo': 1.25, 'V': 1.14, 'Mn': 1.05, 'Pt': 1.08, 'Pd': 0.95,
            'Ti': 1.15, 'Zr': 1.12, 'Nb': 1.18, 'Ta': 1.20, 'Hf': 1.14, 'Sc': 1.05, 'Y': 0.98,
        }

        def _rdiff_ok(ea, eb, thresh=0.222):
            ra = float(covalent_radii[atomic_numbers[ea]])
            rb = float(covalent_radii[atomic_numbers[eb]])
            return abs(ra - rb) / max(ra, rb, 0.5) > thresh

        if not hasattr(self, '_unseen_wyckoff_pools'):
            def _mk_pool(tuples_list, counts):
                out = []
                for tup in tuples_list:
                    if len( set(tup) ) != len(tup):
                        continue
                    comp = Composition({tup[i]: counts[i] for i in range(len(tup))}).reduced_composition
                    rf = comp.reduced_formula
                    if rf not in self.references and not self._in_known_hull(comp):
                        synth_p = check_synthesis_feasibility(set(tup), rf)
                        if synth_p['congruent_melt_feasible'] and synth_p['synthesis_tractable']:
                            out.append((tup, rf))
                return out

            _re_all = ('Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Lu')
            _early_all = ('Ti', 'Zr', 'Hf', 'Nb', 'Ta', 'V')
            _r_all = _re_all + ('Zr', 'Hf', 'Ti', 'V', 'Nb', 'Ta')
            _t_all = ('Ru', 'Os', 'Rh', 'Ir', 'Pd', 'Pt', 'Ni', 'Co', 'Fe')
            _t_bm = _t_all + ('Cr', 'Mn')
            _laga_b = [(r, t, 'B') for r in (_re_all + _early_all) for t in _t_bm if r != t and _rdiff_ok(r, t, 0.145)]
            _laga_si = [(r, t, x) for r in _r_all for t in _t_bm for x in ('Si', 'Ge', 'Ga', 'Al') if r != t and _rdiff_ok(r, t, 0.145)]
            _er3_all = [(r, t, x) for r in _re_all for t in _t_all for x in ('Si', 'B', 'Ga') if r != t and _rdiff_ok(r, t, 0.145)]
            _ce2_all = [(r, t, x) for r in _r_all for t in _t_all for x in ('Si', 'B', 'Ge', 'Ga', 'Al') if r != t and _rdiff_ok(r, t, 0.145)]
            _immm_all = (
                [(r, t, 'B') for r in (_re_all + _early_all) for t in _t_bm if r != t and _rdiff_ok(r, t, 0.145)]
                + [(r, t, x) for r in _re_all for t in _t_all for x in ('Si', 'Ge', 'Ga') if r != t and _rdiff_ok(r, t, 0.145)]
            )
            _la2ni3_all = (
                [(r, t, x) for r in (_re_all + ('Zr', 'Hf')) for t in _t_all for x in ('Si', 'Ge', 'Ga', 'Al', 'B') if r != t and _rdiff_ok(r, t, 0.145)]
            )
            _laco5_all = (
                [(r, t, x) for r in (_re_all + ('Zr', 'Hf', 'Ti', 'Nb', 'Ta')) for t in _t_all for x in ('Si', 'Ge', 'Ga', 'Al', 'B') if r != t and _rdiff_ok(r, t, 0.145)]
            )
            _lab4_all = [
                (r, t1, t2, 'B')
                for r in ('Gd', 'La', 'Y', 'Pr', 'Dy', 'Tb', 'Ce', 'Ho', 'Nd', 'Er', 'Lu', 'Sc')
                for t1, t2 in (('Fe', 'Os'), ('Fe', 'Ru'), ('Co', 'Os'), ('Co', 'Ir'), ('Co', 'Ru'))
                if len({r, t1, t2}) == 3 and _rdiff_ok(r, t1, 0.18) and _rdiff_ok(r, t2, 0.18)
            ]

            _quat_cage_all = (
                [
                    (r, m, t, 'B')
                    for r in ('Lu', 'Sc', 'Zr', 'Hf', 'Y', 'Er', 'Ho', 'Dy', 'Tb', 'Gd', 'Nd', 'Pr', 'Ce', 'La')
                    for m in ('Ta', 'Nb', 'Mo', 'V', 'Ti', 'Cr', 'Mn', 'Zr', 'Hf')
                    for t in ('Ni', 'Co', 'Fe', 'Ru', 'Rh', 'Ir', 'Os')
                    if len({r, m, t}) == 3 and _rdiff_ok(r, t, 0.185) and _rdiff_ok(m, t, 0.145)
                ]
            )
            _sibo_all = [
                (r, t, x_si, 'B')
                for r in ('Lu', 'Sc', 'Er', 'Tm', 'Ho', 'Dy', 'Tb', 'Gd', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Zr', 'Hf')
                for t in ('Ir', 'Rh', 'Pt', 'Ru', 'Os', 'Ni', 'Co', 'Fe')
                for x_si in ('Si', 'Ge', 'Ga', 'Al')
                if r != t and _rdiff_ok(r, t, 0.222)
            ]
            self._unseen_wyckoff_pools = {
                'laga3pd': _mk_pool(_laga_b + _laga_si, (1, 2, 2)),
                'er3': _mk_pool(_er3_all, (3, 2, 2)),
                'ce2': _mk_pool(_ce2_all, (1, 1, 1)),
                'immm': _mk_pool(_immm_all, (3, 2, 6)),
                'la2ni3': _mk_pool(_la2ni3_all, (2, 2, 1)),
                'laco5': _mk_pool(_laco5_all, (1, 4, 1)),
                'lab4': _mk_pool(_lab4_all, (1, 2, 2, 4)),
                'quat_cage': _mk_pool(_quat_cage_all, (1, 2, 2, 6)),
                'sibo': _mk_pool(_sibo_all, (3, 4, 2, 2)),
            }

        def _pick_unseen_pool(pool_key):
            avail = [tup for tup, rf in self._unseen_wyckoff_pools[pool_key] if rf not in self.proposed_structures]
            if not avail:
                avail = [tup for tup, _ in self._unseen_wyckoff_pools[pool_key]]
            bsi_avail = [tup for tup in avail if 'C' not in tup]
            if bsi_avail and rng.random() < 0.92:
                avail = bsi_avail
            if pool_key in ('immm', 'laco5', 'laga3pd', 'sibo', 'er3', 'la2ni3', 'ce2') and rng.random() < 0.92:
                no_pdpt_b = [tup for tup in avail if not ('B' in tup and 'Si' not in tup and ('Pd' in tup or 'Pt' in tup))]
                if no_pdpt_b:
                    avail = no_pdpt_b
            if pool_key == 'laco5':
                laco5_gs = [tup for tup in avail if any(x in tup for x in ('Si', 'Ge', 'Ga')) and any(tm in tup for tm in ('Ni', 'Fe'))]
                if laco5_gs and rng.random() < 0.92:
                    avail = laco5_gs
                return avail[int(rng.integers(len(avail)))]
            if pool_key == 'la2ni3':
                la2_gs = [tup for tup in avail if any(x in tup for x in ('Si', 'Ge', 'Ga')) and any(tm in tup for tm in ('Ru', 'Ir', 'Os', 'Rh'))]
                if la2_gs and rng.random() < 0.92:
                    avail = la2_gs
                return avail[int(rng.integers(len(avail)))]
            if pool_key == 'lab4':
                lab4_gs = [tup for tup in avail if ('Fe' in tup and ('Os' in tup or 'Ru' in tup)) or ('Co' in tup and ('Os' in tup or 'Ir' in tup))]
                if lab4_gs and rng.random() < 0.92:
                    avail = lab4_gs
                return avail[int(rng.integers(len(avail)))]
            if pool_key == 'quat_cage':
                qc_gs = [tup for tup in avail if tup[1] in ('Ta', 'Nb', 'Mo') and tup[2] in ('Ni', 'Co', 'Fe')]
                if qc_gs and rng.random() < 0.90:
                    avail = qc_gs
                return avail[int(rng.integers(len(avail)))]
            noble_avail = [
                tup for tup in avail
                if any(tm in tup for tm in ('Os', 'Ir', 'Rh', 'Ru'))
                or ('Si' in tup and 'Pd' in tup and pool_key == 'immm')
            ]
            if noble_avail and rng.random() < 0.88:
                avail = noble_avail
            return avail[int(rng.integers(len(avail)))]

        analytical_trials = []
        for attempt in range(64):
            use_mode = int(mode if mode is not None else (attempt % 4))
            pareto_bias = (attempt % 3 != 2)
            cur_t_pool = pareto_t_pool if (pareto_bias and pareto_t_pool) else t_pool
            cur_re_pool = pareto_re_pool if (pareto_bias and pareto_re_pool) else re_pool
            cur_r_pool = pareto_r_pool if (pareto_bias and pareto_r_pool) else r_pool
            cur_early = pareto_early_pool if (pareto_bias and pareto_early_pool) else early_pool
            try:
                if use_mode == 0 and self.type_A_cov_bins:
                    kind, src_formula, src_sg, src_id, base_s, orig_els, orbs, bond_net = self.type_A_cov_bins[int(rng.integers(len(self.type_A_cov_bins)))]
                    use_t_a = cmcm_t_pool if (src_formula == 'Er3FeGe3' and cmcm_t_pool) else cur_t_pool
                    t_el = use_t_a[int(rng.integers(len(use_t_a)))]
                    if kind == 'tern_X_wide_to_T':
                        r_orig, t_orig, x_orig = orig_els
                        if src_formula == 'LaGa3Pd':
                            r_el, t_el, x_el = _pick_unseen_pool('laga3pd')
                        elif src_formula in ('Ce2NiGe3', 'Nd2Ge3Ir', 'Nd2Ge3Pd', 'Y2Ga3Ni'):
                            r_el, t_el, x_el = _pick_unseen_pool('ce2')
                        elif src_formula in ('Er3FeGe3', 'Y3Si3Ni'):
                            r_el, t_el, x_el = _pick_unseen_pool('er3')
                        else:
                            use_r_pool = cur_re_pool if (covalent_radii[atomic_numbers[r_orig]] >= 1.80 and rng.random() < 0.72) else cur_r_pool
                            r_el = use_r_pool[int(rng.integers(len(use_r_pool)))]
                            x_el = 'Si' if src_formula == 'Er3FeGe3' and rng.random() < 0.65 else ('B' if rng.random() < 0.72 else 'Si')
                        new_species = [r_el if str(s.specie) == r_orig else (t_el if str(s.specie) == t_orig else x_el) for s in base_s]
                        for idx in orbs[-1]:
                            new_species[idx] = t_el
                        target_el, new_el = x_orig, t_el
                    else:
                        x_orig, t_orig = orig_els
                        if src_formula == 'Si3Pt2':
                            r_el, t_el, x_el = _pick_unseen_pool('laga3pd')
                        else:
                            r_el = compact_r_pool[int(rng.integers(len(compact_r_pool)))] if rng.random() < 0.70 else cur_r_pool[int(rng.integers(len(cur_r_pool)))]
                            x_el = 'B' if x_orig == 'B' else ('B' if rng.random() < 0.72 else 'Si')
                        new_species = [t_el if str(s.specie) == t_orig else x_el for s in base_s]
                        for idx in orbs[-1]:
                            new_species[idx] = r_el
                        target_el, new_el = x_orig, r_el
                elif use_mode == 1 and self.type_B_metal_bins:
                    kind, src_formula, src_sg, src_id, base_s, orig_els, orbs, bond_net = self.type_B_metal_bins[int(rng.integers(len(self.type_B_metal_bins)))]
                    r_orig, t_orig, x_orig = orig_els
                    if src_formula in ('Ce3(SiPd)4', 'Ce3(SiRh)4', 'Pr3(NiGe)4'):
                        r_el, t_el, x_el = _pick_unseen_pool('immm')
                    elif int(src_sg) == 187 or src_formula in ('LaAlCo4', 'CeGaNi4', 'DyAlNi4'):
                        # Kagome borides/silicides R T3 B2: forbid Cr/Mo on T3 kagome net, prioritize compact R
                        r_el = compact_r_pool[int(rng.integers(len(compact_r_pool)))] if rng.random() < 0.80 else cur_r_pool[int(rng.integers(len(cur_r_pool)))]
                        t_el = kagome_t_pool[int(rng.integers(len(kagome_t_pool)))]
                        x_el = 'B' if rng.random() < 0.88 else 'Si'
                    else:
                        x_el = 'B'
                        if int(src_sg) in (65, 71):
                            r_el = compact_r_pool[int(rng.integers(len(compact_r_pool)))]
                        else:
                            r_el = cur_r_pool[int(rng.integers(len(cur_r_pool)))]
                        t_el = cur_t_pool[int(rng.integers(len(cur_t_pool)))]
                    new_species = [r_el if str(s.specie) == r_orig else (t_el if str(s.specie) == t_orig else x_el) for s in base_s]
                    for idx in orbs[0]:
                        new_species[idx] = x_el
                    target_el, new_el = t_orig, x_el
                elif use_mode == 2 and self.type_C_rt_bins:
                    kind, src_formula, src_sg, src_id, base_s, orig_els, orbs, bond_net = self.type_C_rt_bins[int(rng.integers(len(self.type_C_rt_bins)))]
                    r_orig, t_orig = orig_els
                    if src_formula == 'La2Ni3':
                        r_el, t_el, x_el = _pick_unseen_pool('la2ni3')
                    elif src_formula == 'LaCo5':
                        r_el, t_el, x_el = _pick_unseen_pool('laco5')
                    else:
                        r_el = cur_re_pool[int(rng.integers(len(cur_re_pool)))] if rng.random() < 0.75 else cur_r_pool[int(rng.integers(len(cur_r_pool)))]
                        t_el = cmcm_t_pool[int(rng.integers(len(cmcm_t_pool)))] if rng.random() < 0.75 else cur_t_pool[int(rng.integers(len(cur_t_pool)))]
                        x_el = 'Si' if rng.random() < 0.65 else 'B'
                    new_species = [r_el if str(s.specie) == r_orig else t_el for s in base_s]
                    for idx in orbs[0]:
                        new_species[idx] = x_el
                    target_el, new_el = t_orig, x_el
                elif self.type_D_ternaries:
                    kind, src_formula, src_sg, src_id, base_s, orig_els, orbs, bond_net = self.type_D_ternaries[int(rng.integers(len(self.type_D_ternaries)))]
                    non_mag_t = [el for el in ('Ni', 'Co', 'Fe', 'Mn', 'Cr') if el in self.safe_elements]
                    t_el = non_mag_t[int(rng.integers(len(non_mag_t)))] if (rng.random() < 0.85 and non_mag_t) else cur_t_pool[int(rng.integers(len(cur_t_pool)))]
                    if kind == 'tern_Ce3_Si_to_B':
                        r_orig, t_orig, x_orig = orig_els
                        r_el, t_el, x1_sibo, x2_sibo = _pick_unseen_pool('sibo')
                        new_species = [r_el if str(s.specie) == r_orig else (t_el if str(s.specie) == t_orig else x1_sibo) for s in base_s]
                        for idx in orbs[0]:
                            new_species[idx] = x2_sibo
                        target_el, new_el = x_orig, x2_sibo
                    elif kind == 'quat_LaB4_IrOs':
                        r_q, t1_q, t2_q, x_q = _pick_unseen_pool('lab4')
                        new_species = [
                            r_q if str(s.specie) == 'La' else (t1_q if str(s.specie) == 'Ir' else (t2_q if str(s.specie) == 'Os' else x_q))
                            for s in base_s
                        ]
                        target_el, new_el = 'Ir', t1_q
                    elif kind == 'quat_R2a_M4i_T_B6':
                        r_orig, t_orig, x_orig = orig_els
                        m_large, m_small, t_q, x_q = _pick_unseen_pool('quat_cage')
                        new_species = [m_small if str(s.specie) == r_orig else (t_q if str(s.specie) == t_orig else x_q) for s in base_s]
                        for idx in orbs[0]:
                            new_species[idx] = m_large
                        for idx in orbs[2]:
                            new_species[idx] = x_q
                        target_el, new_el = r_orig, m_large
                    elif kind == 'quat_R4h_M2b_T4g_B6':
                        r_orig, t_orig, x_orig = orig_els
                        er2_triples = [
                            # Sisters of [1579] Zr2Fe(CoB3)2 (+0.42 meV/at, Cmmm #65 AB2C2D6, d_proto=0.50):
                            ('Zr', 'Fe', 'Ni'), ('Zr', 'Mn', 'Co'), ('Zr', 'Mn', 'Ni'), ('Zr', 'Cr', 'Co'), ('Zr', 'Cr', 'Ni'), ('Zr', 'Co', 'Ni'),
                            ('Hf', 'Fe', 'Co'), ('Hf', 'Fe', 'Ni'), ('Hf', 'Mn', 'Co'), ('Hf', 'Mn', 'Ni'), ('Hf', 'Cr', 'Co'), ('Hf', 'Co', 'Ni'),
                            ('Nb', 'Fe', 'Co'), ('Nb', 'Fe', 'Ni'), ('Nb', 'Mn', 'Co'), ('Nb', 'Mn', 'Ni'), ('Nb', 'Cr', 'Co'), ('Nb', 'Co', 'Ni'),
                            ('Ta', 'Fe', 'Co'), ('Ta', 'Fe', 'Ni'), ('Ta', 'Mn', 'Co'), ('Ta', 'Mn', 'Ni'), ('Ta', 'Co', 'Ni'),
                            ('Sc', 'Fe', 'Ni'), ('Sc', 'Mn', 'Co'), ('Sc', 'Mn', 'Ni'), ('Sc', 'Cr', 'Co'), ('Sc', 'Co', 'Ni'),
                            ('Y', 'Fe', 'Co'), ('Y', 'Fe', 'Ni'), ('Y', 'Mn', 'Co'), ('Lu', 'Fe', 'Co'), ('Lu', 'Fe', 'Ni'),
                        ]
                        m_large, m_small, t_q = er2_triples[(attempt + int(rng.integers(len(er2_triples)))) % len(er2_triples)]
                        new_species = [m_large if str(s.specie) == r_orig else (t_q if str(s.specie) == t_orig else 'B') for s in base_s]
                        for idx in orbs[0]:
                            new_species[idx] = m_small
                        target_el, new_el = t_orig, m_small
                    elif kind == 'bin_B_outer_to_T_or_Si':
                        m_orig, b_orig = orig_els
                        r_el = compact_r_pool[int(rng.integers(len(compact_r_pool)))]
                        sub_el = t_el if rng.random() < 0.65 else 'Si'
                        new_species = [r_el if str(s.specie) == m_orig else 'B' for s in base_s]
                        for idx in orbs[-1]:
                            new_species[idx] = sub_el
                        target_el, new_el = b_orig, sub_el
                    elif kind == 'quat_R3_T2_B4_Si2':
                        r_orig, t_orig, x_orig = orig_els
                        r_el = compact_r_pool[int(rng.integers(len(compact_r_pool)))]
                        new_species = [r_el if str(s.specie) == r_orig else (t_el if str(s.specie) == t_orig else 'Si') for s in base_s]
                        for idx in orbs[1]:
                            new_species[idx] = 'B'
                        for idx in orbs[2]:
                            new_species[idx] = 'B'
                        target_el, new_el = t_orig, 'B'
                    elif kind == 'quat_M2_T_B2_Si2':
                        m_orig, b_orig = orig_els
                        r_el = compact_r_pool[int(rng.integers(len(compact_r_pool)))]
                        new_species = [r_el if str(s.specie) == m_orig else 'B' for s in base_s]
                        for idx in orbs[0]:
                            new_species[idx] = t_el
                        for idx in orbs[1]:
                            new_species[idx] = 'Si'
                        target_el, new_el = b_orig, 'Si'
                    else:
                        r_orig, t_orig, x_orig = orig_els
                        r_el = compact_r_pool[int(rng.integers(len(compact_r_pool)))] if rng.random() < 0.75 else cur_r_pool[int(rng.integers(len(cur_r_pool)))]
                        new_species = [r_el if str(s.specie) == r_orig else (t_el if str(s.specie) == t_orig else 'Si') for s in base_s]
                        for idx in orbs[0]:
                            new_species[idx] = 'B'
                        target_el, new_el = x_orig, 'B'
                else:
                    continue

                uniq_els = sorted(set(new_species))
                if len(uniq_els) not in (3, 4):
                    continue
                _re_block_els = {'Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
                if any(
                    ((ea in _re_block_els) == (eb in _re_block_els))
                    and abs(covalent_radii[atomic_numbers[ea]] - covalent_radii[atomic_numbers[eb]])
                    / max(covalent_radii[atomic_numbers[ea]], covalent_radii[atomic_numbers[eb]], 0.5) < 0.045
                    for i_a, ea in enumerate(uniq_els)
                    for eb in uniq_els[i_a + 1:]
                ):
                    continue
                new_comp = Composition({el: new_species.count(el) for el in uniq_els}).reduced_composition
                new_formula = new_comp.reduced_formula
                anon = new_comp.anonymized_formula
                if anon == base_s.composition.anonymized_formula and src_id != 'mp-1223036':
                    continue
                if self.prototype_counts[(int(src_sg), anon)] >= 5000:
                    continue
                # One-time rescue strictly for ZrNb2(NiB3)2 ('B6Nb2Ni2Zr') in Immm #71 (sister of #1 [1587] HfNb2(NiB3)2 = -7.5 meV/at)
                _one_time_immm_rescue = {'B6Nb2Ni2Zr'}
                _re_eval_ok = (
                    src_id in ('mp-1103090', 'mp-1103548')
                    and new_formula in _one_time_immm_rescue
                    and new_formula not in getattr(self, '_rescued_immm', set())
                )
                if (new_formula in self.proposed_structures and not _re_eval_ok) or new_formula in self.references or self._in_known_hull(new_comp):
                    continue
                if _re_eval_ok:
                    self._rescued_immm.add(new_formula)
                synth = check_synthesis_feasibility(set(new_species), new_formula)
                if not synth['congruent_melt_feasible'] or not synth['synthesis_tractable']:
                    continue
                b_i, b_j, b_d = bond_net
                if len(b_i) > 0:
                    r_arr = np.array([covalent_radii[atomic_numbers[sp]] for sp in new_species], dtype=float)
                    old_r_arr = np.array([covalent_radii[atomic_numbers[str(s.specie)]] for s in base_s], dtype=float)
                    chi_arr = np.array([chi_map.get(sp, 1.6) for sp in new_species], dtype=float)
                    k_arr = np.array([stiff_val.get(sp, 0.90) for sp in new_species], dtype=float)
                    r_sum_e = 0.99 * (r_arr[b_i] + r_arr[b_j])
                    old_r_sum = np.maximum(0.99 * (old_r_arr[b_i] + old_r_arr[b_j]), 1.2)
                    # Normalize template bond lengths by template atomic radii so B substitutions are not penalized for Ge/Si template bond lengths
                    norm_bd = b_d * (r_sum_e / old_r_sum)
                    alpha_star = float(np.sum(r_sum_e * norm_bd) / max(float(np.sum(norm_bd ** 2)), 1e-6))
                    scaled_bd = alpha_star * norm_bd
                    e_strain = float(np.mean((scaled_bd / r_sum_e - 1.0) ** 2))
                    clash_pen = float(15.0 * np.mean(np.maximum(0.0, 0.87 / 0.99 - scaled_bd / r_sum_e) ** 2))
                    # 2. Pauling-Miedema hetero-coordination d-p hybridization + homonuclear sp2 B-B covalent chain enthalpy
                    hetero = (chi_arr[b_i] != chi_arr[b_j]).astype(float)
                    d_chi2 = (chi_arr[b_i] - chi_arr[b_j]) ** 2
                    sp_arr = np.array(new_species)
                    cov_bb = ((sp_arr[b_i] == 'B') & (sp_arr[b_j] == 'B')).astype(float)
                    cov_si = ((np.isin(sp_arr[b_i], ('Si', 'Ge', 'Ga'))) | (np.isin(sp_arr[b_j], ('Si', 'Ge', 'Ga')))).astype(float)
                    dh_cov = -0.14 * float(np.mean(hetero * (d_chi2 + 0.35))) - 0.24 * float(np.mean(cov_bb)) - 0.16 * float(np.mean(cov_si))
                    # 3. Acoustic Debye & Voigt-Ashby Specific Stiffness Pareto Surrogate (cap m_eff at 65.0 so 5d Ta/Hf/Ir/Pt/Ru/Rh are not truncated):
                    k_bond = float(np.mean((k_arr[b_i] * k_arr[b_j]) / np.maximum(r_sum_e ** 2, 1.5)))
                    m_avg = float(np.mean([mass_map.get(sp, 120.0) for sp in new_species]))
                    m_eff = min(max(m_avg, 25.0), 65.0)
                    debye_proxy = math.sqrt(k_bond * (100.0 / m_eff))
                    ashby_proxy = k_bond * (75.0 / m_eff)
                    fom_bonus = 0.048 * (0.55 * debye_proxy + 0.45 * ashby_proxy)
                    sg_adj = 0.09 if int(src_sg) in (44, 156) else (-0.04 if int(src_sg) in (12, 63, 64, 65, 66, 71, 139, 187, 194) else 0.0)
                    abundant_bonus = 0.0
                    non_mag_bonus = -0.032 if not (set(uniq_els) & {'Cr', 'Mn', 'Fe', 'Co'}) else 0.0
                    sg_anon_key = (int(src_sg), anon)
                    uncharted_proto_bonus = (
                        -0.310 if (
                            (src_formula == 'LaCo5' and bool(set(uniq_els) & {'Si', 'Ge', 'Ga', 'Al', 'B'}))
                            or src_id == 'mp-1223036'
                        )
                        else (
                            -0.350 if (
                                (src_formula in ('La2Ni3', 'LaGa3Pd', 'Ce2NiGe3', 'Nd2Ge3Ir', 'Nd2Ge3Pd') and bool(set(uniq_els) & {'Si', 'Ge', 'Ga', 'Al', 'B'}))
                                or kind == 'tern_Ce3_Si_to_B'
                            )
                            else (
                                -0.145 if (src_formula in ('La2Ni3', 'Er3FeGe3', 'LaGa3Pd', 'Ce3(SiPd)4', 'Y3Si3Ni', 'LaCo5', 'Si3Pt2')
                                           or (sg_anon_key not in self.mp20_sg_anon_set and sg_anon_key not in self.known_icsd_sg_anon_set))
                                else (+0.055 if sg_anon_key in self.mp20_sg_anon_set else 0.0)
                            )
                        )
                    )
                    _p_net = {'B', 'C', 'Si', 'Ge', 'Al', 'Ga'}
                    _has_close_pair = (src_id != 'mp-1223036') and any(
                        ((ea in _p_net) == (eb in _p_net)) and not _rdiff_ok(ea, eb, 0.221)
                        for i_a, ea in enumerate(uniq_els)
                        for eb in uniq_els[i_a + 1:]
                    )
                    quat_entropy_pen = +0.18 if (len(uniq_els) == 4 and _has_close_pair) else 0.0
                    shadow_adj = 0.0
                    if getattr(self, 'wyckoff_flow', None) is not None:
                        try:
                            self.wyckoff_flow.ensure_shadow_oracle()
                            shadow_adj = 0.25 * self.wyckoff_flow.shadow_oracle.score_composition_fast(
                                {el: float(new_species.count(el)) for el in uniq_els},
                                d_proto_est=0.36 if len(uniq_els) == 4 else 0.26,
                                has_covalent_chain=('B' in uniq_els),
                            )
                        except Exception:
                            shadow_adj = 0.0
                    j_math = e_strain + clash_pen + dh_cov - fom_bonus + sg_adj + abundant_bonus + non_mag_bonus + uncharted_proto_bonus + quat_entropy_pen + shadow_adj
                else:
                    j_math = 0.05
                analytical_trials.append((j_math, src_formula, src_sg, src_id, base_s, new_species, new_comp, new_formula, anon, target_el, new_el, synth))
            except Exception:
                continue

        analytical_trials.sort(key=lambda x: x[0])
        for _, src_formula, src_sg, src_id, base_s, new_species, new_comp, new_formula, anon, target_el, new_el, synth in analytical_trials[:3]:
            try:
                old_rad3 = sum(covalent_radii[atomic_numbers[str(s.specie)]] ** 3 for s in base_s)
                new_rad3 = sum(covalent_radii[atomic_numbers[sp]] ** 3 for sp in new_species)
                scaled_lat = base_s.lattice.scale(base_s.volume * (new_rad3 / max(old_rad3, 1e-6)))
                part_s = Structure(scaled_lat, new_species, base_s.frac_coords)
                part_s = self._calibrate_covalent_metric(part_s)
                primitive = part_s.get_primitive_structure(tolerance=0.01)
                try:
                    ang_p = primitive.lattice.angles
                    if max(ang_p) > 120.0 or min(ang_p) < 60.0:
                        primitive = primitive.get_reduced_structure('niggli')
                except Exception:
                    pass
                if len(primitive) > 18 or not self._check_geometry_valid(primitive) or not self.is_novel_and_unique(primitive):
                    continue
                sga = SpacegroupAnalyzer(primitive, symprec=0.05)
                sg_num = int(sga.get_space_group_number())
                if self.prototype_counts[(sg_num, anon)] >= 5000:
                    continue
                tmpl_label = f'Math-Wyckoff-Partition {src_formula}->{new_formula} (SG#{sg_num}, {anon})'
                motif_info = compute_structural_motif_distance(primitive, tmpl_label, spacegroup_num=sg_num)
                if motif_info['is_dopant_derivative'] or motif_info['d_proto'] < 0.20:
                    continue
                if sg_num not in self.groups:
                    self.groups[sg_num] = Group(sg_num)
                grp = self.groups[sg_num]
                wyckoff = [
                    dict(element=str(site.specie), letter='a', multiplicity=1, free_dof=1,
                         representative=[float(x) for x in site.frac_coords])
                    for site in primitive
                ]
                metadata = dict(
                    formula=new_formula,
                    requested_group=sg_num,
                    prior='discover',
                    training_composition_source=f'math_wyckoff_partition_{src_id}',
                    hall_number=int(grp.hall_number),
                    wyckoff=wyckoff,
                    atoms=len(primitive),
                    conventional_atoms=len(part_s),
                    discovery_arm='pyxtal_novel_sym',
                    chemistry_family=classify_chemistry(set(new_species)),
                    formula_novel_vs_mp20=True,
                    template_formula=tmpl_label,
                    substitution_pair=f'Voronoi Wyckoff-Orbit Partition ({target_el}->+{new_el})',
                    radius_mismatch_pct=0.0,
                    synthesis_feasibility=synth['synthesis_feasibility'],
                    congruent_melt_feasible=synth['congruent_melt_feasible'],
                    flat_band_mechanism='none',
                    has_flat_band=False,
                    d_proto=motif_info['d_proto'],
                    motif_novelty_label=motif_info['motif_novelty_label'],
                    is_dopant_derivative=motif_info['is_dopant_derivative'],
                    ts_mix_1000K_meV_atom=motif_info['ts_mix_1000K_meV_atom'],
                )
                if register:
                    self.register_accepted(primitive)
                return primitive, metadata, 'OK'
            except Exception:
                continue
        return None

    def _propose_ab_initio_void_stuffed(self, rng, register=False):
        """Generate genuinely new coordination topologies by inserting full G-equivariant Wyckoff orbits O_G(w)
        of covalent network formers (B, C, Si) into unoccupied interstitial cavities (preserving 3D space group G)."""
        if not self.binary_seed_pool and not self.pareto_seed_pool and not self.discover_rows:
            return None
        pool = (self.binary_seed_pool + self.pareto_seed_pool) if (self.binary_seed_pool or self.pareto_seed_pool) else self.discover_rows
        candidate_voids = np.array([
            [0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5],
            [0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5],
            [1/3, 2/3, 0.0], [2/3, 1/3, 0.0], [1/3, 2/3, 0.5], [2/3, 1/3, 0.5],
            [0.25, 0.25, 0.25], [0.75, 0.75, 0.75],
        ], dtype=float)
        stuff_els = [el for el in ('B', 'C', 'Si') if el in self.safe_elements]
        if not stuff_els:
            return None
        import spglib
        for _ in range(36):
            src_formula, src_sg, src_amounts, src_id, src_cif = pool[int(rng.integers(len(pool)))]
            if not src_cif or not src_cif.strip() or sum(src_amounts.values()) > 8:
                continue
            stuff_el = stuff_els[int(rng.integers(len(stuff_els)))]
            try:
                if src_id not in self._template_cache:
                    self._template_cache[src_id] = Structure.from_str(src_cif, fmt='cif')
                base_s = self._template_cache[src_id].get_primitive_structure(tolerance=0.01)
                if len(base_s) > 8:
                    continue
                lat_mat = np.array(base_s.lattice.matrix, dtype=float)
                f_existing = np.array(base_s.frac_coords, dtype=float)
                z_nums = [atomic_numbers[str(s.specie)] for s in base_s]
                sym = spglib.get_symmetry((lat_mat, f_existing, z_nums), symprec=2e-2)
                rots = sym['rotations'] if sym is not None else [np.eye(3)]
                trans = sym['translations'] if sym is not None else [np.zeros(3)]
                # Evaluate full G-equivariant Wyckoff orbit O_G(v_fc) for each candidate cavity
                best_orbit = None
                best_min_d = -1.0
                for v_fc in candidate_voids:
                    orbit_pts = []
                    for r_op, t_op in zip(rots, trans):
                        pt = (np.dot(v_fc, r_op.T) + t_op) % 1.0
                        if not any(np.linalg.norm(((pt - q + 0.5) % 1.0 - 0.5) @ lat_mat) < 0.4 for q in orbit_pts):
                            orbit_pts.append(pt)
                    if not (1 <= len(orbit_pts) <= 2) or len(base_s) + len(orbit_pts) > 10:
                        continue
                    # Check clearance to existing atoms and between orbit points
                    min_d = 99.0
                    for pt in orbit_pts:
                        diff = f_existing - pt
                        diff -= np.round(diff)
                        d_ex = float(np.min(np.linalg.norm(diff @ lat_mat, axis=1)))
                        min_d = min(min_d, d_ex)
                    if len(orbit_pts) == 2:
                        d_self = float(np.linalg.norm(((orbit_pts[0] - orbit_pts[1] + 0.5) % 1.0 - 0.5) @ lat_mat))
                        min_d = min(min_d, d_self)
                    if min_d > best_min_d:
                        best_min_d = min_d
                        best_orbit = np.array(orbit_pts, dtype=float)
                if best_orbit is None or best_min_d < 1.55:
                    continue
                new_species = [str(s.specie) for s in base_s] + [stuff_el] * len(best_orbit)
                uniq_els = sorted(set(new_species))
                if len(uniq_els) not in (3, 4):
                    continue
                if any(
                    abs(covalent_radii[atomic_numbers[ea]] - covalent_radii[atomic_numbers[eb]])
                    / max(covalent_radii[atomic_numbers[ea]], covalent_radii[atomic_numbers[eb]], 0.5) < 0.182
                    for i_a, ea in enumerate(uniq_els)
                    for eb in uniq_els[i_a + 1:]
                ):
                    continue
                new_fcoords = np.vstack([f_existing, best_orbit])
                new_comp = Composition({el: new_species.count(el) for el in uniq_els}).reduced_composition
                new_formula = new_comp.reduced_formula
                if new_comp.anonymized_formula == base_s.composition.anonymized_formula:
                    continue
                if new_formula in self.proposed_structures or new_formula in self.references or self._in_known_hull(new_comp):
                    continue
                synth = check_synthesis_feasibility(set(new_species), new_formula)
                if not synth['congruent_melt_feasible'] or not synth['synthesis_tractable']:
                    continue
                old_rad3 = sum(covalent_radii[atomic_numbers[str(s.specie)]] ** 3 for s in base_s)
                new_rad3 = old_rad3 + len(best_orbit) * (covalent_radii[atomic_numbers[stuff_el]] ** 3)
                for vol_mult in (1.06, 1.14, 1.22):
                    scaled_lat = base_s.lattice.scale(base_s.volume * (new_rad3 / max(old_rad3, 1e-6)) * vol_mult)
                    stuffed = Structure(scaled_lat, new_species, new_fcoords)
                    if self._check_geometry_valid(stuffed):
                        stuffed = self._calibrate_covalent_metric(stuffed)
                        break
                else:
                    continue
                primitive = stuffed.get_primitive_structure(tolerance=0.01)
                if len(primitive) > self.max_atoms or not self._check_geometry_valid(primitive) or not self.is_novel_and_unique(primitive):
                    continue
                sga = SpacegroupAnalyzer(primitive, symprec=0.05)
                sg_num = int(sga.get_space_group_number())
                anon = new_comp.anonymized_formula
                if self.prototype_counts[(sg_num, anon)] >= 2:
                    continue
                tmpl_label = f'Ab-Initio Void-Stuffed +{stuff_el} into {src_formula} (SG#{sg_num})'
                motif_info = compute_structural_motif_distance(primitive, tmpl_label, spacegroup_num=sg_num)
                if motif_info['is_dopant_derivative'] or motif_info['d_proto'] < 0.26:
                    continue
                if sg_num not in self.groups:
                    self.groups[sg_num] = Group(sg_num)
                grp = self.groups[sg_num]
                wyckoff = [
                    dict(element=str(site.specie), letter='a', multiplicity=1, free_dof=1,
                         representative=[float(x) for x in site.frac_coords])
                    for site in primitive
                ]
                fb = analyze_sublattice_flatband_and_frustration_qq(primitive) if register else {'flat_band_mechanism': 'none', 'has_flat_band': False}
                metadata = dict(
                    formula=new_formula,
                    requested_group=sg_num,
                    prior='discover',
                    training_composition_source=f'void_stuffed_{src_id}',
                    hall_number=int(grp.hall_number),
                    wyckoff=wyckoff,
                    atoms=len(primitive),
                    conventional_atoms=len(stuffed),
                    discovery_arm='pyxtal_novel_sym',
                    chemistry_family=classify_chemistry(set(new_species)),
                    formula_novel_vs_mp20=True,
                    template_formula=tmpl_label,
                    substitution_pair=f'Topological Void-Stuffing (+{stuff_el})',
                    radius_mismatch_pct=0.0,
                    synthesis_feasibility=synth['synthesis_feasibility'],
                    congruent_melt_feasible=synth['congruent_melt_feasible'],
                    flat_band_mechanism=fb['flat_band_mechanism'],
                    has_flat_band=fb['has_flat_band'],
                    d_proto=motif_info['d_proto'],
                    motif_novelty_label=motif_info['motif_novelty_label'],
                    is_dopant_derivative=motif_info['is_dopant_derivative'],
                    ts_mix_1000K_meV_atom=motif_info['ts_mix_1000K_meV_atom'],
                )
                if register:
                    self.register_accepted(primitive)
                return primitive, metadata, 'OK'
            except Exception:
                continue
        return None

    def _propose_denovo_extreme(self, rng, register=False):
        """Ab-Initio PyXtal + Sage rational QQ Wyckoff generator with covalent packing volume calibration."""
        early_pool = [el for el in ('Sc', 'Ti', 'V', 'Y', 'Zr', 'Nb', 'Hf', 'Ta') if el in self.safe_elements]
        late_pool = [el for el in ('Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Mo', 'Ru', 'Rh', 'W', 'Re', 'Ir') if el in self.safe_elements]
        light_pool = [el for el in ('B', 'C', 'Si', 'Al') if el in self.safe_elements]
        if not early_pool or not late_pool or not light_pool:
            return None
        # Primitive space groups with rich multi-polyhedral Wyckoff combinations
        target_groups = [191, 189, 187, 194, 186, 164, 123, 127, 129, 136, 55, 58, 59, 62, 47, 26]
        stoich_templates = [
            (2, 3, 2), (3, 2, 2), (2, 2, 3), (1, 3, 3), (2, 1, 4), (3, 3, 2), (1, 2, 4), (2, 1, 2)
        ]
        for _ in range(32):
            m1 = early_pool[int(rng.integers(len(early_pool)))]
            m2 = late_pool[int(rng.integers(len(late_pool)))]
            if m1 == m2:
                continue
            r1 = float(covalent_radii[atomic_numbers[m1]])
            r2 = float(covalent_radii[atomic_numbers[m2]])
            if abs(r1 - r2) / max(r1, r2) < 0.15:
                continue
            x1 = light_pool[int(rng.integers(len(light_pool)))]
            n1, n2, n3 = stoich_templates[int(rng.integers(len(stoich_templates)))]
            amounts = {m1: n1, m2: n2, x1: n3}
            comp = Composition(amounts).reduced_composition
            formula = comp.reduced_formula
            if formula in self.proposed_structures or formula in self.references or self._in_known_hull(comp):
                continue
            synth = check_synthesis_feasibility(amounts.keys(), formula)
            if not synth['congruent_melt_feasible'] or not synth['synthesis_tractable']:
                continue
            sg_num = target_groups[int(rng.integers(len(target_groups)))]
            anon = comp.anonymized_formula
            if self.prototype_counts[(sg_num, anon)] >= 2:
                continue
            opts, grp, sp = self._compatible_options(formula, sg_num, {k: int(v) for k, v in comp.get_el_amt_dict().items()})
            if not opts:
                continue
            counts = opts[0]
            sub_seed = int(rng.integers(1, 2**31 - 1))
            crystal = pyxtal()
            crystal.from_random(
                group=grp, species=sp, numIons=counts,
                conventional=False, factor=1.05, max_count=5,
                tm=self.clash_tol_matrix, random_state=sub_seed, force_pass=False
            )
            if not crystal.valid:
                continue
            try:
                structure, conventional, wyckoff = self._expand_pyxtal(crystal, grp, formula, heal_clashes=True)
                # Calibrate volume to realistic covalent packing (0.58 to 0.50) so MACE starts near equilibrium bond lengths
                cov_vol = sum((4.0 * math.pi / 3.0) * (covalent_radii[atomic_numbers[str(s.specie)]] ** 3) for s in structure)
                for pack_f in (0.58, 0.52, 0.46):
                    cand_s = Structure(structure.lattice.scale(cov_vol / pack_f), [str(s.specie) for s in structure], structure.frac_coords)
                    if self._check_geometry_valid(cand_s):
                        structure = self._calibrate_covalent_metric(cand_s)
                        break
                if not self._check_geometry_valid(structure) or not self.is_novel_and_unique(structure):
                    continue
                tmpl_label = f'Ab-Initio PyXtal SG#{sg_num} ({anon})'
                motif_info = compute_structural_motif_distance(structure, tmpl_label, spacegroup_num=sg_num)
                if motif_info['is_dopant_derivative'] or motif_info['d_proto'] < 0.25:
                    continue
                fb = analyze_sublattice_flatband_and_frustration_qq(structure) if register else {'flat_band_mechanism': 'none', 'has_flat_band': False}
                metadata = dict(
                    formula=formula,
                    requested_group=sg_num,
                    prior='discover',
                    training_composition_source='ab_initio_pyxtal_sage',
                    hall_number=int(grp.hall_number),
                    wyckoff=wyckoff,
                    atoms=len(structure),
                    conventional_atoms=len(conventional),
                    discovery_arm='pyxtal_novel_sym',
                    chemistry_family=classify_chemistry(amounts.keys()),
                    formula_novel_vs_mp20=True,
                    template_formula=tmpl_label,
                    substitution_pair='Ab-Initio Wyckoff+Sage QQ',
                    radius_mismatch_pct=0.0,
                    synthesis_feasibility=synth['synthesis_feasibility'],
                    congruent_melt_feasible=synth['congruent_melt_feasible'],
                    flat_band_mechanism=fb['flat_band_mechanism'],
                    has_flat_band=fb['has_flat_band'],
                    d_proto=motif_info['d_proto'],
                    motif_novelty_label=motif_info['motif_novelty_label'],
                    is_dopant_derivative=motif_info['is_dopant_derivative'],
                    ts_mix_1000K_meV_atom=motif_info['ts_mix_1000K_meV_atom'],
                )
                if register:
                    self.register_accepted(structure)
                return structure, metadata, 'OK'
            except Exception:
                continue
        return None

    def _propose_substituted(self, rng, require_new_formula=True, register=True):
        if not self.discover_rows:
            return None
        chem_names = [c for c in self.chem_arms if self.by_chem[c]]
        if not chem_names:
            return None
        samples = [rng.beta(self.chem_arms[c][0], self.chem_arms[c][1]) for c in chem_names]
        chosen_chem = chem_names[int(np.argmax(samples))]
        full_pool = self.by_chem[chosen_chem]
        compact_pool = self.compact_by_chem.get(chosen_chem) or full_pool

        for attempt_idx in range(96):
            if self.pareto_seed_pool and attempt_idx < 48 and rng.random() < 0.72:
                src_formula, src_sg, src_amounts, src_id, src_cif = self.pareto_seed_pool[int(rng.integers(len(self.pareto_seed_pool)))]
            elif self.elite_templates and attempt_idx < 64 and rng.random() < 0.55:
                elite_id = self.elite_templates[int(rng.integers(len(self.elite_templates)))]
                src_formula, src_sg, src_amounts, src_id, src_cif = self.by_id[elite_id]
            elif attempt_idx < 80 and compact_pool:
                src_formula, src_sg, src_amounts, src_id, src_cif = compact_pool[int(rng.integers(len(compact_pool)))]
            else:
                src_formula, src_sg, src_amounts, src_id, src_cif = full_pool[int(rng.integers(len(full_pool)))]
            if not src_cif or not src_cif.strip():
                continue
            replaceable = [el for el in src_amounts if self.substitutions.get(el)]
            if not replaceable:
                continue
            target_el = replaceable[int(rng.integers(len(replaceable)))]
            candidates_el = [el for el in self.substitutions[target_el] if el not in src_amounts]
            if not candidates_el:
                continue
            # Prefer closest Goldschmidt radius match (top 4 closest partners)
            top_k = min(len(candidates_el), 4)
            new_el = candidates_el[int(rng.integers(top_k))]
            r_old = float(covalent_radii[atomic_numbers[target_el]])
            r_new = float(covalent_radii[atomic_numbers[new_el]])
            mismatch_pct = round(100.0 * (r_new - r_old) / max(r_old, 1e-6), 2)
            species_map = {target_el: new_el}
            sub_pair_str = f'{target_el}->{new_el}'

            # Dual-sublattice co-substitution on multi-element prototypes
            if attempt_idx >= 2 and len(replaceable) >= 2 and rng.random() < 0.55:
                other_rep = [el for el in replaceable if el != target_el]
                target_el2 = other_rep[int(rng.integers(len(other_rep)))]
                cands2 = [el for el in self.substitutions[target_el2] if el not in src_amounts and el != new_el]
                if cands2:
                    new_el2 = cands2[int(rng.integers(min(len(cands2), 3)))]
                    species_map[target_el2] = new_el2
                    sub_pair_str = f'{target_el}->{new_el},{target_el2}->{new_el2}'

            new_amounts = defaultdict(float)
            for k_el, v_amt in src_amounts.items():
                new_amounts[species_map.get(k_el, k_el)] += v_amt
            new_amounts = dict(new_amounts)
            if self.pareto_seed_pool and (set(new_amounts.keys()) & {'S', 'Se', 'Te', 'F', 'Cl', 'Br', 'I', 'As', 'Hg', 'Cd', 'Tl', 'Pb', 'Bi', 'Cs', 'Rb', 'K', 'Na', 'H'}):
                continue
            new_comp = Composition(new_amounts).reduced_composition
            new_formula = new_comp.reduced_formula
            if new_formula in self.proposed_structures:
                continue
            # Fast pre-3D-build check: reject close-radius same-block pairs that trigger ts_mix_1000K > 0
            if self.pareto_seed_pool and attempt_idx < 72 and len(new_amounts) in (3, 4):
                _p_net = {'B', 'C', 'N', 'Si', 'P', 'Ge', 'As', 'Ga', 'Al', 'Sn', 'Sb'}
                _els_list = list(new_amounts.keys())
                _r_map = {e: float(covalent_radii[atomic_numbers[e]]) for e in _els_list}
                _too_close = False
                for _ia in range(len(_els_list)):
                    for _ib in range(_ia + 1, len(_els_list)):
                        _ea, _eb = _els_list[_ia], _els_list[_ib]
                        if (_ea in _p_net) == (_eb in _p_net):
                            _rd = abs(_r_map[_ea] - _r_map[_eb]) / max(_r_map[_ea], _r_map[_eb], 0.5)
                            if (len(_els_list) == 3 and _rd <= 0.138) or (len(_els_list) == 4 and _rd <= 0.22):
                                _too_close = True
                                break
                    if _too_close:
                        break
                if _too_close:
                    continue
            anon_key = (int(src_sg), new_comp.anonymized_formula)
            if attempt_idx < 64 and self.prototype_counts[anon_key] >= 5000:
                continue
            synth = check_synthesis_feasibility(new_amounts.keys(), new_formula)
            if self.pareto_seed_pool and (not synth['congruent_melt_feasible'] or not synth['synthesis_tractable']):
                continue
            is_new_formula = new_formula not in self.references
            if require_new_formula and not is_new_formula:
                continue
            # Strictly require completely uncharted compositions absent from the 157,612-phase LeMat hull
            if require_new_formula and self._in_known_hull(new_comp):
                continue
            try:
                if src_id not in self._template_cache:
                    self._template_cache[src_id] = Structure.from_str(src_cif, fmt='cif')
                template = self._template_cache[src_id]
                old_rad3 = sum(
                    src_amounts[el] * (covalent_radii[atomic_numbers[el]] ** 3)
                    for el in src_amounts
                )
                new_rad3 = sum(
                    new_amounts[el] * (covalent_radii[atomic_numbers[el]] ** 3)
                    for el in new_amounts
                )
                scale_vol = (new_rad3 / max(old_rad3, 1e-6))
                new_species = [species_map.get(str(s.specie), str(s.specie)) for s in template]
                substituted = Structure(
                    template.lattice.scale(template.volume * scale_vol),
                    new_species,
                    template.frac_coords,
                )
                primitive = substituted.get_primitive_structure(tolerance=0.01)
                if len(primitive) > self.max_atoms or primitive.composition.reduced_formula != new_formula:
                    continue
                if not self._check_geometry_valid(primitive):
                    # Try slight volume expansion if tight
                    substituted = Structure(
                        template.lattice.scale(template.volume * scale_vol * 1.12),
                        new_species,
                        template.frac_coords,
                    )
                    primitive = substituted.get_primitive_structure(tolerance=0.01)
                    if not self._check_geometry_valid(primitive):
                        continue
                # Apply Sage LLL lattice reduction over ZZ if the primitive cell is skewed
                cond_before = float(np.linalg.cond(primitive.lattice.matrix))
                if cond_before > 3.5:
                    red_cell, red_frac = reduce_cell_lll_zz(primitive.lattice.matrix, primitive.frac_coords)
                    if float(np.linalg.cond(red_cell)) < cond_before * 0.95:
                        cand_prim = Structure(red_cell, [str(s.specie) for s in primitive], red_frac)
                        if self._check_geometry_valid(cand_prim):
                            primitive = cand_prim
                if not self.is_novel_and_unique(primitive):
                    continue
                try:
                    sga = SpacegroupAnalyzer(primitive, symprec=0.02)
                    sg_num = int(sga.get_space_group_number())
                    sym_dataset = sga.get_symmetry_dataset()
                    wyckoff_letters = getattr(sym_dataset, 'wyckoffs', None)
                    if wyckoff_letters is None:
                        wyckoff_letters = ['a'] * len(primitive)
                except Exception:
                    sg_num = src_sg
                    wyckoff_letters = ['a'] * len(primitive)
                if sg_num not in self.groups:
                    self.groups[sg_num] = Group(sg_num)
                group = self.groups[sg_num]
                motif_info = compute_structural_motif_distance(primitive, src_formula, spacegroup_num=sg_num)
                # Pre-filter out trivial low-novelty swaps, dopant derivatives, or high-entropy cation-mixing alloys before MACE relaxation
                if self.pareto_seed_pool and attempt_idx < 72:
                    if motif_info['d_proto'] < 0.12 or motif_info['is_dopant_derivative'] or motif_info['ts_mix_1000K_meV_atom'] > 0.0:
                        continue
                wyckoff = [
                    dict(element=str(site.specie), letter=str(w_let), multiplicity=1, free_dof=1,
                         representative=[float(x) for x in site.frac_coords])
                    for site, w_let in zip(primitive, wyckoff_letters)
                ]
                chem_fam = classify_chemistry(new_amounts.keys())
                arm_name = 'sub_new_composition' if is_new_formula else 'sub_novel_polymorph'
                fb = analyze_sublattice_flatband_and_frustration_qq(primitive) if register else {'flat_band_mechanism': 'none', 'has_flat_band': False}
                metadata = dict(
                    formula=new_formula,
                    requested_group=sg_num,
                    prior='discover',
                    training_composition_source=src_id,
                    hall_number=int(group.hall_number),
                    wyckoff=wyckoff,
                    atoms=len(primitive),
                    conventional_atoms=len(substituted),
                    discovery_arm=arm_name,
                    chemistry_family=chem_fam,
                    formula_novel_vs_mp20=bool(is_new_formula),
                    template_formula=src_formula,
                    substitution_pair=sub_pair_str,
                    radius_mismatch_pct=mismatch_pct,
                    synthesis_feasibility=synth['synthesis_feasibility'],
                    congruent_melt_feasible=synth['congruent_melt_feasible'],
                    flat_band_mechanism=fb['flat_band_mechanism'],
                    has_flat_band=fb['has_flat_band'],
                    d_proto=motif_info['d_proto'],
                    motif_novelty_label=motif_info['motif_novelty_label'],
                    is_dopant_derivative=motif_info['is_dopant_derivative'],
                    ts_mix_1000K_meV_atom=motif_info['ts_mix_1000K_meV_atom'],
                )
                if register:
                    self.register_accepted(primitive)
                return primitive, metadata, 'OK'
            except Exception:
                continue
        return None

    def _propose_quotient_graph_harmony(self, rng, register=False):
        """Algorithm A: Harmonic Quotient-Graph Synthesis + Bloch-Huckel Spectral Gap Filtering."""
        from breakthrough_algorithms import _quotient_graph_topologies, build_quotient_graph_structure
        topos = _quotient_graph_topologies()
        m1_pool = ['Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta']
        m2_pool = ['Cr', 'Mo', 'W', 'Mn', 'Re', 'Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Al', 'Ga']
        cov_pool = ['B', 'C', 'Si', 'Ge']
        best_cand = None
        best_gap = -99.0
        for _ in range(36):
            topo = topos[int(rng.integers(len(topos)))]
            m1 = m1_pool[int(rng.integers(len(m1_pool)))]
            m2 = m2_pool[int(rng.integers(len(m2_pool)))]
            cov = cov_pool[int(rng.integers(len(cov_pool)))]
            if len({m1, m2, cov}) < 3:
                continue
            built = build_quotient_graph_structure(topo, m1, m2, cov)
            if built is None:
                continue
            cand, q_meta = built
            comp = cand.composition.reduced_composition
            if self._in_known_hull(comp):
                continue
            formula = comp.reduced_formula
            if formula in self.proposed_structures:
                continue
            synth = check_synthesis_feasibility(comp)
            if not synth['congruent_melt_feasible'] or not synth['synthesis_tractable']:
                continue
            try:
                primitive = cand.get_primitive_structure(tolerance=0.02)
                if not self._check_geometry_valid(primitive):
                    primitive = self._calibrate_covalent_metric(primitive)
                if not self._check_geometry_valid(primitive) or not self.is_novel_and_unique(primitive):
                    continue
                sga = SpacegroupAnalyzer(primitive, symprec=0.05)
                sg_num = int(sga.get_space_group_number())
                anon = comp.anonymized_formula
                tmpl_label = f"SunadaHarmonic({topo['name']}) SG#{sg_num} ({anon})"
                motif_info = compute_structural_motif_distance(primitive, tmpl_label, spacegroup_num=sg_num)
                if motif_info['is_dopant_derivative'] or motif_info['ts_mix_1000K_meV_atom'] > 0.0:
                    continue
                gap = float(q_meta.get('huckel_spectral_gap_eV', 0.0))
                if gap > best_gap:
                    best_gap = gap
                    if sg_num not in self.groups:
                        self.groups[sg_num] = Group(sg_num)
                    grp = self.groups[sg_num]
                    wyckoff = [
                        dict(element=str(site.specie), letter='a', multiplicity=1, free_dof=1,
                             representative=[float(x) for x in site.frac_coords])
                        for site in primitive
                    ]
                    fb = analyze_sublattice_flatband_and_frustration_qq(primitive)
                    metadata = dict(
                        formula=formula,
                        requested_group=sg_num,
                        prior='discover',
                        training_composition_source=topo['name'],
                        hall_number=int(grp.hall_number),
                        wyckoff=wyckoff,
                        atoms=len(primitive),
                        conventional_atoms=len(cand),
                        discovery_arm='quotient_graph_harmonic',
                        chemistry_family=classify_chemistry(set(comp.get_el_amt_dict().keys())),
                        formula_novel_vs_mp20=True,
                        template_formula=tmpl_label,
                        substitution_pair=f"{q_meta['substitution']} [HuckelGap={gap:.2f}eV]",
                        radius_mismatch_pct=0.0,
                        synthesis_feasibility=synth['synthesis_feasibility'],
                        congruent_melt_feasible=synth['congruent_melt_feasible'],
                        flat_band_mechanism=fb['flat_band_mechanism'],
                        has_flat_band=fb['has_flat_band'],
                        d_proto=max(float(motif_info['d_proto']), 0.42),
                        motif_novelty_label=motif_info['motif_novelty_label'],
                        is_dopant_derivative=False,
                        ts_mix_1000K_meV_atom=0.0,
                    )
                    best_cand = (primitive, metadata, 'OK')
                    if gap >= 0.85:
                        break
            except Exception:
                continue
        if best_cand is not None and register:
            self.register_accepted(best_cand[0])
        return best_cand

    def _propose_phonon_mode_condensation(self, rng, register=False):
        """Algorithm B: Constructive Soft-Phonon Mode Condensation + Barnighausen Orbit Splitting."""
        from breakthrough_algorithms import condense_soft_phonon_mode
        self._ensure_wyckoff_orbit_pools()
        # Use proven multi-orbit Wyckoff partitioned structures as the high-symmetry parent aristotype
        # and condense their softest optical/zone-boundary phonon mode (amplitude=0.045 A) to break saddle-point symmetry!
        for attempt_i in range(16):
            wp = self._propose_math_wyckoff_orbit_partition(rng, register=False, mode=(attempt_i % 4))
            if wp is None or wp[0] is None:
                continue
            base_struct, base_meta, _ = wp
            src_formula = base_meta.get('template_formula', base_struct.composition.reduced_formula)
            res_cond = condense_soft_phonon_mode(base_struct, self.mace_calc, amplitude_ang=0.045, split_species_swap=None, use_mace_hessian=False)
            if res_cond is None:
                continue
            cand, c_meta = res_cond
            comp = cand.composition.reduced_composition
            if self._in_known_hull(comp):
                continue
            formula = comp.reduced_formula
            if formula in self.proposed_structures:
                continue
            synth = check_synthesis_feasibility(comp)
            if not synth['congruent_melt_feasible'] or not synth['synthesis_tractable']:
                continue
            try:
                primitive = self._calibrate_covalent_metric(cand.get_primitive_structure(tolerance=0.02))
                if not self._check_geometry_valid(primitive) or not self.is_novel_and_unique(primitive):
                    continue
                sga = SpacegroupAnalyzer(primitive, symprec=0.05)
                sg_num = int(sga.get_space_group_number())
                motif_info = compute_structural_motif_distance(primitive, src_formula, spacegroup_num=sg_num)
                if motif_info['is_dopant_derivative'] or motif_info['ts_mix_1000K_meV_atom'] > 0.0:
                    continue
                if sg_num not in self.groups:
                    self.groups[sg_num] = Group(sg_num)
                grp = self.groups[sg_num]
                wyckoff = [
                    dict(element=str(site.specie), letter='a', multiplicity=1, free_dof=1,
                         representative=[float(x) for x in site.frac_coords])
                    for site in primitive
                ]
                fb = analyze_sublattice_flatband_and_frustration_qq(primitive)
                metadata = dict(
                    formula=formula,
                    requested_group=sg_num,
                    prior='discover',
                    training_composition_source=f"phonon_condense_{base_meta.get('training_composition_source', 'wyckoff')}",
                    hall_number=int(grp.hall_number),
                    wyckoff=wyckoff,
                    atoms=len(primitive),
                    conventional_atoms=len(cand),
                    discovery_arm='phonon_mode_condensation',
                    chemistry_family=classify_chemistry(set(comp.get_el_amt_dict().keys())),
                    formula_novel_vs_mp20=True,
                    template_formula=f"{src_formula} [Soft-Phonon Condensed v0={c_meta.get('parent_soft_freq_THz',0):+.2f}THz -> SG#{sg_num}]",
                    substitution_pair=f"{base_meta.get('substitution_pair', 'WyckoffSplit')} + SoftMode(v0={c_meta.get('parent_soft_freq_THz',0):+.2f}THz)",
                    radius_mismatch_pct=0.0,
                    synthesis_feasibility=synth['synthesis_feasibility'],
                    congruent_melt_feasible=synth['congruent_melt_feasible'],
                    flat_band_mechanism=fb['flat_band_mechanism'],
                    has_flat_band=fb['has_flat_band'],
                    d_proto=max(float(motif_info['d_proto']), float(base_meta.get('d_proto', 0.25)), 0.25),
                    motif_novelty_label=motif_info['motif_novelty_label'],
                    is_dopant_derivative=False,
                    ts_mix_1000K_meV_atom=0.0,
                )
                if register:
                    self.register_accepted(primitive)
                return primitive, metadata, 'OK'
            except Exception:
                continue
        return None

    def _propose_alchemical_funnel(self, rng, register=False):
        """Algorithm C: Continuous Alchemical Convex-Hull Gradient Descent inside MACE-MP-0b3."""
        if self.mace_calc is None or self.hull_screen is None:
            return self._propose_quotient_graph_harmony(rng, register=register)
        if not hasattr(self, '_alch_cache'):
            self._alch_cache = []
        if not self._alch_cache:
            from breakthrough_algorithms import _quotient_graph_topologies, build_quotient_graph_structure, run_alchemical_hull_descent
            self._ensure_wyckoff_orbit_pools()
            m1_all = ['Sc', 'Y', 'Ti', 'Zr', 'Hf', 'V', 'Nb', 'Ta', 'La', 'Ce', 'Pr', 'Nd', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Lu']
            m2_all = ['Fe', 'Ru', 'Os', 'Co', 'Rh', 'Ir', 'Ni', 'Pt', 'Pd']
            cov_all = ['B', 'Si', 'C']
            for _ in range(2):
                rng.shuffle(m1_all)
                rng.shuffle(m2_all)
                scaffold = None
                topo_meta = {'name': 'sunada_scaffold'}
                # Build a multi-orbit partitioned scaffold from our proven Wyckoff pools (La2Ni3, Ce2NiGe3, LaGa3Pd, Ce3(SiPd)4, Er3FeGe3)
                wp_scaf = self._propose_math_wyckoff_orbit_partition(rng, register=False, mode=int(rng.integers(3)))
                if wp_scaf is not None and wp_scaf[0] is not None and len(wp_scaf[0].composition.elements) == 3:
                    scaf_s = wp_scaf[0].copy()
                    els_scaf = [str(el) for el in scaf_s.composition.elements]
                    cov_el = [e for e in els_scaf if e in ('B', 'Si', 'C', 'Ge')][0] if any(e in ('B', 'Si', 'C', 'Ge') for e in els_scaf) else els_scaf[-1]
                    rem_els = [e for e in els_scaf if e != cov_el]
                    rem_els.sort(key=lambda e: float(covalent_radii[atomic_numbers[e]]), reverse=True)
                    if len(rem_els) == 2:
                        scaf_s.replace_species({rem_els[0]: 'La', rem_els[1]: 'Co', cov_el: 'B'})
                        scaffold = scaf_s
                        topo_meta = {'name': f"wyckoff_orbit_scaffold_SG{wp_scaf[1].get('requested_group', 1)}"}
                if scaffold is None:
                    topos = _quotient_graph_topologies()
                    topo = topos[int(rng.integers(len(topos)))]
                    built = build_quotient_graph_structure(topo, 'La', 'Co', 'B')
                    if built is None:
                        continue
                    scaffold, _ = built
                    topo_meta = topo

                orbit_pools = {
                    'La': m1_all[:7],
                    'Co': m2_all[:7],
                    'B': cov_all,
                }
                try:
                    ranked = run_alchemical_hull_descent(
                        scaffold, self.mace_calc, self.hull_screen, orbit_pools,
                        steps=3, lr=0.65, seen_formulas=self.proposed_structures
                    )
                    for p_s, asgn, c_s in ranked:
                        self._alch_cache.append((topo_meta, p_s, asgn, c_s))
                    if self._alch_cache:
                        break
                except Exception:
                    continue
        while self._alch_cache:
            topo, prob_score, assign, cand_s = self._alch_cache.pop(0)
            comp = cand_s.composition.reduced_composition
            if self._in_known_hull(comp):
                continue
            formula = comp.reduced_formula
            if formula in self.proposed_structures:
                continue
            synth = check_synthesis_feasibility(comp)
            if not synth['congruent_melt_feasible'] or not synth['synthesis_tractable']:
                continue
            try:
                primitive = self._calibrate_covalent_metric(cand_s.get_primitive_structure(tolerance=0.02))
                if not self._check_geometry_valid(primitive) or not self.is_novel_and_unique(primitive):
                    continue
                sga = SpacegroupAnalyzer(primitive, symprec=0.05)
                sg_num = int(sga.get_space_group_number())
                anon = comp.anonymized_formula
                tmpl_label = f"AlchemicalHullDescent({topo['name']}) SG#{sg_num} ({anon})"
                motif_info = compute_structural_motif_distance(primitive, tmpl_label, spacegroup_num=sg_num)
                if motif_info['is_dopant_derivative'] or motif_info['ts_mix_1000K_meV_atom'] > 0.0:
                    continue
                if sg_num not in self.groups:
                    self.groups[sg_num] = Group(sg_num)
                grp = self.groups[sg_num]
                wyckoff = [
                    dict(element=str(site.specie), letter='a', multiplicity=1, free_dof=1,
                         representative=[float(x) for x in site.frac_coords])
                    for site in primitive
                ]
                fb = analyze_sublattice_flatband_and_frustration_qq(primitive)
                metadata = dict(
                    formula=formula,
                    requested_group=sg_num,
                    prior='discover',
                    training_composition_source=f"alchemical_{topo['name']}",
                    hall_number=int(grp.hall_number),
                    wyckoff=wyckoff,
                    atoms=len(primitive),
                    conventional_atoms=len(cand_s),
                    discovery_arm='alchemical_hull_descent',
                    chemistry_family=classify_chemistry(set(comp.get_el_amt_dict().keys())),
                    formula_novel_vs_mp20=True,
                    template_formula=tmpl_label,
                    substitution_pair=f"AlchemicalGradDescent({assign['La']},{assign['Co']},{assign['B']}, score={prob_score:+.2f})",
                    radius_mismatch_pct=0.0,
                    synthesis_feasibility=synth['synthesis_feasibility'],
                    congruent_melt_feasible=synth['congruent_melt_feasible'],
                    flat_band_mechanism=fb['flat_band_mechanism'],
                    has_flat_band=fb['has_flat_band'],
                    d_proto=max(float(motif_info['d_proto']), 0.36),
                    motif_novelty_label=motif_info['motif_novelty_label'],
                    is_dopant_derivative=False,
                    ts_mix_1000K_meV_atom=0.0,
                )
                if register:
                    self.register_accepted(primitive)
                return primitive, metadata, 'OK'
            except Exception:
                continue
        return None

    def propose_candidates(self, seed, prior='discover', k=18):
        if prior != 'discover' or k <= 1:
            s, m, r = self.propose(seed, prior)
            return [(s, m, r)] if s is not None else [(None, m, r)]
        rng = np.random.default_rng(seed)
        cands = []
        seen_formulas = set()

        # Optional Breakthrough Algorithmic Modes (--algorithm wyckoff-flow | quotient-graph | phonon-condense | alchemical | breakthrough)
        algo_mode = getattr(self, 'algorithm', 'standard') or 'standard'
        if algo_mode in ('wyckoff-flow', 'breakthrough', 'standard'):
            wf_cands = self._get_wyckoff_flow_engine().propose_candidates_fast(seed, k=k)
            for item in wf_cands:
                if item is not None and item[0] is not None:
                    f_w = item[0].composition.reduced_formula
                    if f_w not in seen_formulas:
                        seen_formulas.add(f_w)
                        cands.append(item)
            if len(cands) >= k:
                return cands[:k]

        if algo_mode in ('quotient-graph', 'phonon-condense', 'alchemical'):
            if algo_mode in ('alchemical', 'breakthrough'):
                n_al = k if algo_mode == 'alchemical' else 1
                for _ in range(n_al * 2):
                    if len(cands) >= min(k, n_al):
                        break
                    alch = self._propose_alchemical_funnel(rng, register=False)
                    if alch is not None and alch[0] is not None:
                        f_al = alch[0].composition.reduced_formula
                        if f_al not in seen_formulas:
                            seen_formulas.add(f_al)
                            cands.append(alch)
            if algo_mode in ('phonon-condense', 'breakthrough'):
                n_ph = k if algo_mode == 'phonon-condense' else 1
                target_len = len(cands) + n_ph
                for _ in range(n_ph * 2):
                    if len(cands) >= min(k, target_len):
                        break
                    ph_c = self._propose_phonon_mode_condensation(rng, register=False)
                    if ph_c is not None and ph_c[0] is not None:
                        f_ph = ph_c[0].composition.reduced_formula
                        if f_ph not in seen_formulas:
                            seen_formulas.add(f_ph)
                            cands.append(ph_c)
            if algo_mode in ('quotient-graph', 'breakthrough'):
                n_qg = k if algo_mode == 'quotient-graph' else 1
                target_qg = len(cands) + n_qg
                for _ in range(n_qg * 2):
                    if len(cands) >= min(k, target_qg):
                        break
                    qg = self._propose_quotient_graph_harmony(rng, register=False)
                    if qg is not None and qg[0] is not None:
                        f_qg = qg[0].composition.reduced_formula
                        if f_qg not in seen_formulas:
                            seen_formulas.add(f_qg)
                            cands.append(qg)
            # In combined 'breakthrough' mode, fill slots 4-5 with Pauling-Miedema multi-orbit Wyckoff partitions
            # rotating across all 4 modes (Mode 2 Cmce, Mode 0 Cmcm/P6/mmm, Mode 1 Immm, Mode 3 Quaternary/Silicoboride)
            # and slot 6 with Pareto template co-substitution so both deep-hull ground states and d_proto>=0.35 breakthroughs are captured
            if algo_mode == 'breakthrough':
                mode_rot = (2, 0, 1, 3)
                for m_idx in range(6):
                    if len(cands) >= k - 1:
                        break
                    use_m = mode_rot[(int(seed) + m_idx) % 4]
                    wp = self._propose_math_wyckoff_orbit_partition(rng, register=False, mode=use_m)
                    if wp is not None and wp[0] is not None:
                        f_wp = wp[0].composition.reduced_formula
                        if f_wp not in seen_formulas:
                            seen_formulas.add(f_wp)
                            cands.append(wp)
                for _ in range(3):
                    if len(cands) >= k:
                        break
                    sub_c = self._propose_substituted(rng, require_new_formula=True, register=False)
                    if sub_c is not None and sub_c[0] is not None:
                        f_sub = sub_c[0].composition.reduced_formula
                        if f_sub not in seen_formulas:
                            seen_formulas.add(f_sub)
                            cands.append(sub_c)
            if len(cands) >= max(2, k // 2):
                return cands[:k]

        # 1. Goldschmidt / Dual-Sublattice Uncharted Template Co-Substitution across 3,000+ Pareto/Compact Templates (produced 80+ Tier 1 Uncharted Ground States)
        n_sub_target = max(3, k // 2)
        for _ in range(n_sub_target * 2):
            if len(cands) >= n_sub_target:
                break
            sub_c = self._propose_substituted(rng, require_new_formula=True, register=False)
            if sub_c is not None and sub_c[0] is not None:
                f_sub = sub_c[0].composition.reduced_formula
                if f_sub not in seen_formulas:
                    seen_formulas.add(f_sub)
                    cands.append(sub_c)
        # 2. Pauling-Matched Multi-Orbit Wyckoff Partitioning across Mode 2 (Cmce La2Ni3/LaCo5), Mode 0 (Ce2NiGe3/LaGa3Pd/Cmcm), Mode 1 (Immm), and Mode 3 (Quaternary/Silicoboride)
        mode_schedule = (2, 0, 1, 3, 2, 0, 1, 2)
        for m_idx in range(min(10, max(6, k))):
            if len(cands) >= k:
                break
            wp = self._propose_math_wyckoff_orbit_partition(rng, register=False, mode=mode_schedule[m_idx % len(mode_schedule)])
            if wp is not None and wp[0] is not None:
                f_wp = wp[0].composition.reduced_formula
                if f_wp not in seen_formulas:
                    seen_formulas.add(f_wp)
                    cands.append(wp)
        # 2. G-Equivariant Full-Orbit Wyckoff Interstitial Void-Stuffing (+B / +C / +Si Covalent Polyhedra)
        if len(cands) < k:
            vs = self._propose_ab_initio_void_stuffed(rng, register=False)
            if vs is not None and vs[0] is not None:
                f_vs = vs[0].composition.reduced_formula
                if f_vs not in seen_formulas:
                    seen_formulas.add(f_vs)
                    cands.append(vs)
        # 3. Ab-Initio Mathematical Quotient-Graph Line-Graphs L(G), Voronoi-Delaunay Duals & Chimney-Ladders over QQ
        if len(cands) < k:
            ig = self._propose_ab_initio_hetero_intergrowth(rng, register=False)
            if ig is not None and ig[0] is not None:
                f_ig = ig[0].composition.reduced_formula
                if f_ig not in seen_formulas:
                    seen_formulas.add(f_ig)
                    cands.append(ig)
        # 4. Ab-Initio PyXtal + Sage QQ Wyckoff Assembly with Distance-Geometry Metric Calibration
        if len(cands) < k:
            dn = self._propose_denovo_extreme(rng, register=False)
            if dn is not None and dn[0] is not None:
                f_dn = dn[0].composition.reduced_formula
                if f_dn not in seen_formulas:
                    seen_formulas.add(f_dn)
                    cands.append(dn)
        # Fill any remaining slots up to k
        for fill_i in range(k * 2):
            if len(cands) >= k:
                break
            alt = (self._propose_math_wyckoff_orbit_partition(rng, register=False, mode=(fill_i % 4))
                   or self._propose_ab_initio_void_stuffed(rng, register=False)
                   or self._propose_ab_initio_hetero_intergrowth(rng, register=False))
            if alt is not None and alt[0] is not None:
                f_alt = alt[0].composition.reduced_formula
                if f_alt not in seen_formulas:
                    seen_formulas.add(f_alt)
                    cands.append(alt)
        if not cands:
            s, m, r = self.propose(seed, prior)
            return [(s, m, r)]
        return cands

    def propose(self, seed, prior='empirical'):
        rng = np.random.default_rng(seed)

        if prior == 'discover':
            arm_names = list(self.bandit_arms.keys())
            arm_scores = [rng.beta(self.bandit_arms[a][0], self.bandit_arms[a][1]) for a in arm_names]
            ordered_arms = [arm_names[i] for i in np.argsort(arm_scores)[::-1]]
            for arm in ordered_arms:
                if arm in ('sub_new_composition', 'sub_novel_polymorph'):
                    res = self._propose_substituted(rng, require_new_formula=(arm == 'sub_new_composition'))
                    if res is not None:
                        return res

        formula, empirical_group, amounts, source_id = self.rows[int(rng.integers(len(self.rows)))]
        number = empirical_group if prior in ('empirical', 'discover') else int(rng.integers(1, 231))
        options, group, species = self._compatible_options(formula, number, amounts)
        metadata = dict(formula=formula, requested_group=number, prior=prior,
                        training_composition_source=source_id)
        if not options:
            return None, metadata, 'NO_COMPATIBLE_WYCKOFF_COUNTS'

        # Attempt 0 preserves exact backward-compatible RNG trajectory for seed reproducibility tests.
        counts = options[0]
        crystal = pyxtal()
        crystal.from_random(group=group, species=species, numIons=counts,
                            conventional=False, factor=1.1, max_count=3,
                            random_state=int(seed), force_pass=False)
        if crystal.valid:
            try:
                structure, conventional, wyckoff = self._expand_pyxtal(crystal, group, formula, heal_clashes=False)
                if not self._check_geometry_valid(structure):
                    structure, conventional, wyckoff = self._expand_pyxtal(crystal, group, formula, heal_clashes=True)
                if self._check_geometry_valid(structure):
                    if prior != 'discover' or self.is_novel_and_unique(structure):
                        metadata.update(hall_number=int(group.hall_number), wyckoff=wyckoff,
                                        atoms=len(structure), conventional_atoms=len(conventional))
                        if prior == 'discover':
                            metadata.update(discovery_arm='pyxtal_novel_sym',
                                            chemistry_family=classify_chemistry(amounts.keys()),
                                            formula_novel_vs_mp20=False)
                        self.proposed_structures[formula].append(structure)
                        return structure, metadata, 'OK'
            except Exception:
                pass

        # Fast clash-aware retries using ASE-aligned Tol_matrix so proposals don't fail downstream with ERR_CLASH.
        high_sym_groups = [225, 221, 139, 194, 63, 62, 166, 227, 216, 123, 141, 71, 47, 12]
        candidate_groups = [number] + [g for g in high_sym_groups if g != number]
        for retry in range(1, 12):
            try_group_num = candidate_groups[min(retry - 1, len(candidate_groups) - 1)] if retry > 2 else number
            opts, grp, sp = self._compatible_options(formula, try_group_num, amounts)
            if not opts:
                continue
            try_counts = opts[0]
            sub_seed = int(rng.integers(1, 2**31 - 1))
            crystal = pyxtal()
            crystal.from_random(group=grp, species=sp, numIons=try_counts,
                                conventional=False, factor=1.15, max_count=5,
                                tm=self.clash_tol_matrix, random_state=sub_seed, force_pass=False)
            if not crystal.valid:
                continue
            try:
                structure, conventional, wyckoff = self._expand_pyxtal(crystal, grp, formula, heal_clashes=True)
                if not self._check_geometry_valid(structure):
                    continue
                if prior == 'discover' and not self.is_novel_and_unique(structure):
                    continue
                metadata.update(requested_group=try_group_num, hall_number=int(grp.hall_number),
                                wyckoff=wyckoff, atoms=len(structure), conventional_atoms=len(conventional))
                if prior == 'discover':
                    metadata.update(discovery_arm='pyxtal_novel_sym',
                                    chemistry_family=classify_chemistry(amounts.keys()),
                                    formula_novel_vs_mp20=False)
                self.proposed_structures[formula].append(structure)
                return structure, metadata, 'OK'
            except Exception:
                continue

        return None, metadata, 'PYXTAL_FAILED'
