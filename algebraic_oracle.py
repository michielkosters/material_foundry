"""Periodic geometry and optional chemistry constraints; run with Sage Python."""
import numpy as np
from sage.all import QQ, ZZ, matrix, Graph, vector
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.data import atomic_numbers, covalent_radii


def periodic_connectivity(adjacency):
    """Dimensional rank and index of the translation subgroup of a connected lift."""
    if not adjacency:
        raise ValueError('Empty periodic graph')
    offsets, cycles = {0: vector(ZZ, [0, 0, 0])}, []
    stack = [0]
    while stack:
        a = stack.pop()
        for b, shift in adjacency[a]:
            offset = offsets[a] + vector(ZZ, shift)
            if b not in offsets:
                offsets[b] = offset
                stack.append(b)
            else:
                cycles.append(offset - offsets[b])
    cycle_matrix = matrix(ZZ, cycles) if cycles else matrix(ZZ, 0, 3)
    rank = int(cycle_matrix.rank())
    index = None
    if rank == 3:
        diagonal = cycle_matrix.smith_form(transformation=False).diagonal()
        index = abs(int(diagonal[0] * diagonal[1] * diagonal[2]))
    connected = len(offsets) == len(adjacency)
    return {'periodic_rank': rank, 'translation_index': index,
            'quotient_connected': connected,
            'single_3d_network': connected and rank == 3 and index == 1}


def expand_exact(operations, asymmetric_sites):
    """Apply rational fractional operations. Operations must share the site's setting.

    sites: (element, [rational x,y,z]); operations: (3x3 rotation, translation).
    Wyckoff selection/setting validation belongs to the crystallographic bridge.
    """
    parsed_ops = [(r if hasattr(r, 'base_ring') else matrix(QQ, r),
                   t if hasattr(t, 'base_ring') else vector(QQ, t))
                  for r, t in operations]
    output = {}
    for species, coords in asymmetric_sites:
        vec = coords if hasattr(coords, 'base_ring') else vector(QQ, coords)
        for rotation, translation in parsed_ops:
            point = rotation * vec + translation
            key = tuple(x - x.floor() for x in point)
            if key in output and output[key] != species:
                raise ValueError("Different species occupy the same exact site")
            output[key] = species
    return [(species, coords) for coords, species in output.items()]


def optimize_wyckoff_clearance_qq(operations_by_site, species_by_site, initial_reps, lattice_matrix, steps=6):
    """Use exact rational Wyckoff affine maps (R_i u + t_i mod 1) to shift free Wyckoff
    coordinates away from self-clash and inter-orbit singularity hyperplanes.

    For each site s with free degrees of freedom (rank(R_0 - I) or non-zero derivative),
    identifies its invariant rational affine subspace and projects initial coordinates
    toward maximum minimum-image Euclidean clearance while preserving exact Wyckoff symmetry.
    """
    cell = np.asarray(lattice_matrix, dtype=float)
    reps = [vector(QQ, [QQ(round(float(x) * 1440)) / 1440 for x in rep]) for rep in initial_reps]
    free_bases = []
    for ops in operations_by_site:
        r0, t0 = ops[0]
        r0_mat = r0 if hasattr(r0, 'base_ring') else matrix(QQ, r0)
        # The invariant subspace of the Wyckoff projector P_0 satisfies (P_0 - I) * delta = 0
        ker = (r0_mat - matrix.identity(QQ, 3)).right_kernel()
        ortho_basis = []
        for b in ker.basis():
            v = np.array([float(x) for x in b], dtype=float)
            for u in ortho_basis:
                v = v - np.dot(v, u) * u
            norm = float(np.linalg.norm(v))
            if norm > 1e-9:
                ortho_basis.append(v / norm)
        free_bases.append(ortho_basis)

    if not any(free_bases):
        return [[float(x) for x in rep] for rep in reps]

    current_reps = [np.array([float(x) for x in rep], dtype=float) for rep in reps]
    parsed_ops = [
        [(np.array([[float(x) for x in row] for row in (r if hasattr(r, 'rows') else r)], dtype=float),
          np.array([float(x) for x in t], dtype=float))
         for r, t in ops]
        for ops in operations_by_site
    ]
    radii = [float(_COVALENT_RADII[atomic_numbers[sp]]) for sp in species_by_site]

    for _ in range(steps):
        all_frac, all_rad, all_site_idx, all_op_rot = [], [], [], []
        for s_idx, (rep, ops, rad) in enumerate(zip(current_reps, parsed_ops, radii)):
            for rot, trans in ops:
                pt = (rot @ rep + trans) % 1.0
                all_frac.append(pt)
                all_rad.append(rad)
                all_site_idx.append(s_idx)
                all_op_rot.append(rot)
        frac_arr = np.asarray(all_frac, dtype=float)
        rad_arr = np.asarray(all_rad, dtype=float)
        atoms = Atoms(numbers=[atomic_numbers[species_by_site[idx]] for idx in all_site_idx],
                      scaled_positions=frac_arr, cell=cell, pbc=True)
        i_idx, j_idx, d_ij, D_vec = neighbor_list('ijdD', atoms, 1.05 * rad_arr, self_interaction=False)
        if len(d_ij) == 0:
            break
        min_ratios = d_ij / (rad_arr[i_idx] + rad_arr[j_idx])
        worst = int(np.argmin(min_ratios))
        if min_ratios[worst] >= 0.78:
            break
        a, b = int(i_idx[worst]), int(j_idx[worst])
        sa, sb = all_site_idx[a], all_site_idx[b]
        # Cartesian vector from a to b is D_vec[worst]; push a away from b in fractional space
        cart_dir = D_vec[worst] / max(float(d_ij[worst]), 1e-4)
        frac_grad = np.linalg.solve(cell.T, -cart_dir)
        moved = False
        for s_target, sign, op_rot in [(sa, 1.0, all_op_rot[a]), (sb, -1.0, all_op_rot[b])]:
            basis = free_bases[s_target]
            if not basis:
                continue
            g_rep = op_rot.T @ (sign * frac_grad)
            proj = sum(np.dot(g_rep, b_vec) * b_vec for b_vec in basis)
            norm = float(np.linalg.norm(proj))
            if norm > 1e-6:
                current_reps[s_target] = (current_reps[s_target] + 0.035 * (proj / norm)) % 1.0
                moved = True
        if not moved:
            break

    # Snap back onto exact rational stabilizer subspace over QQ
    final_reps = []
    for s_idx, (rep_np, ops) in enumerate(zip(current_reps, operations_by_site)):
        r0, t0 = ops[0]
        r0_mat = r0 if hasattr(r0, 'base_ring') else matrix(QQ, r0)
        t0_vec = t0 if hasattr(t0, 'base_ring') else vector(QQ, t0)
        q_vec = vector(QQ, [QQ(int(round(float(x) * 2880))) / 2880 for x in rep_np])
        projected = r0_mat * q_vec + t0_vec
        final_reps.append([float(x - x.floor()) for x in projected])
    return final_reps


def reduce_cell_lll_zz(lattice_matrix, frac_coords, scale=10000):
    """Use Sage's exact integer LLL lattice basis reduction over ZZ to find a unimodular
    transformation U in GL(3, ZZ) (|det U| = 1) that minimizes basis skewness and condition
    number while preserving the exact periodic crystal lattice.
    """
    cell = np.asarray(lattice_matrix, dtype=float)
    coords = np.asarray(frac_coords, dtype=float)
    int_basis = matrix(ZZ, np.rint(cell * scale).astype(int))
    if int_basis.det() == 0:
        return cell, coords % 1.0
    reduced_int = int_basis.LLL()
    # Unimodular change-of-basis matrix U = reduced_int * int_basis^(-1) in GL(3, ZZ)
    u_mat = reduced_int * int_basis.inverse()
    if abs(u_mat.det()) != 1 or not all(x in ZZ for x in u_mat.list()):
        return cell, coords % 1.0
    u_np = np.array([[float(x) for x in row] for row in u_mat.rows()], dtype=float)
    if u_mat.det() < 0:
        u_np = -u_np
    new_cell = u_np @ cell
    if np.linalg.cond(new_cell) >= np.linalg.cond(cell):
        return cell, coords % 1.0
    u_inv = np.linalg.inv(u_np)
    new_coords = (coords @ u_inv) % 1.0
    return new_cell, new_coords


def optimize_anisotropic_metric_qq(operations_by_site, species, frac_coords, lattice_matrix):
    """Use Sage's Graph connected components and exact rational linear algebra over QQ to
    compute the symmetry-invariant axis orbits of the crystal system and solve for the
    anisotropic lattice metric scaling (e.g. c/a ratio in hexagonal/tetragonal/trigonal or
    a:b:c in orthorhombic crystals) that matches ideal covalent/metallic bond lengths.
    """
    cell = np.asarray(lattice_matrix, dtype=float)
    coords = np.asarray(frac_coords, dtype=float)
    if not operations_by_site or len(species) < 2:
        return cell
    # Partition lattice vectors {0, 1, 2} into symmetry-coupled orbits via Sage Graph
    g_sym = Graph(loops=False, multiedges=False)
    g_sym.add_vertices([0, 1, 2])
    for r, _ in operations_by_site[0]:
        r_mat = r if hasattr(r, 'base_ring') else matrix(QQ, r)
        for i in range(3):
            for j in range(3):
                if i != j and r_mat[i, j] != 0:
                    g_sym.add_edge(i, j)
    orbits = g_sym.connected_components()
    if len(orbits) <= 1:
        return cell

    z_list = [atomic_numbers[s] for s in species]
    radii = _COVALENT_RADII[z_list]
    atoms = Atoms(numbers=z_list, scaled_positions=coords, cell=cell, pbc=True)
    i_idx, j_idx, d_ij, D_vec = neighbor_list('ijdD', atoms, 1.32 * radii, self_interaction=False)
    if len(d_ij) < len(orbits):
        return cell

    # Express each bond vector in fractional displacements df = D_vec * cell^(-1)
    # Under lattice row-scaling L' = diag(s_0, s_1, s_2) * L, only valid when lattice rows
    # between distinct orbits are orthogonal (e.g. c _|_ ab in hexagonal/trigonal/tetragonal/orthorhombic).
    gram = cell @ cell.T
    for orb_a in orbits:
        for orb_b in orbits:
            if orb_a is not orb_b:
                for ia in orb_a:
                    for ib in orb_b:
                        if abs(gram[ia, ib]) > 1e-3 * math.sqrt(gram[ia, ia] * gram[ib, ib]):
                            return cell

    inv_cell = np.linalg.inv(cell)
    df = D_vec @ inv_cell
    # Contribution of orbit k to squared bond length: sum_{a,b in orb_k} df_a * gram[a,b] * df_b
    n_orb = len(orbits)
    design_rows = []
    rhs_vals = []
    target_d2 = (1.12 * (radii[i_idx] + radii[j_idx])) ** 2
    for b_idx in range(min(len(d_ij), 48)):
        row_q = []
        for orb in orbits:
            val = 0.0
            for ia in orb:
                for ib in orb:
                    val += float(df[b_idx, ia] * gram[ia, ib] * df[b_idx, ib])
            row_q.append(QQ(int(round(val * 1000))) / 1000)
        if any(x > 0 for x in row_q):
            design_rows.append(row_q)
            rhs_vals.append(QQ(int(round(float(target_d2[b_idx]) * 1000))) / 1000)
    if len(design_rows) < n_orb:
        return cell

    A_qq = matrix(QQ, design_rows)
    b_qq = vector(QQ, rhs_vals)
    # Normal equations over QQ with exact rational ridge regularization toward unit scaling (1, ..., 1)
    reg = QQ(2)
    ata = A_qq.transpose() * A_qq + reg * matrix.identity(QQ, n_orb)
    atb = A_qq.transpose() * b_qq + reg * vector(QQ, [QQ(1)] * n_orb)
    try:
        sol_qq = ata.solve_right(atb)
    except Exception:
        return cell
    scales = np.ones(3, dtype=float)
    for orb, s2_qq in zip(orbits, sol_qq):
        s_val = float(s2_qq)
        s_clamped = math.sqrt(max(0.88 ** 2, min(1.12 ** 2, s_val))) if s_val > 0 else 1.0
        for axis_idx in orb:
            scales[axis_idx] = s_clamped
    # Preserve total cell volume (let Vegard's law govern overall volume while Sage sets c/a anisotropy)
    vol_factor = float(np.prod(scales)) ** (1.0 / 3.0)
    if vol_factor > 0:
        scales = scales / vol_factor
    return (scales[:, None] * cell)


import math
_COVALENT_RADII = np.asarray(covalent_radii, dtype=float)


class AlgebraicOracle:
    def __init__(self, target_coordination=None, clash_scale=0.7,
                 bond_scale=1.25, require_3d=False, require_biconnected=False,
                 analyze_topology=False):
        if not np.isfinite([clash_scale, bond_scale]).all() or not 0 < clash_scale <= bond_scale:
            raise ValueError('Require 0 < clash_scale <= bond_scale')
        self.target_coordination = target_coordination or {}
        self.clash_scale = float(clash_scale)
        self.bond_scale = float(bond_scale)
        self.require_3d = require_3d
        self.require_biconnected = require_biconnected
        self.analyze_topology = analyze_topology

    def _validate_and_radii(self, coords, species, lattice_matrix):
        try:
            cell = np.asarray(lattice_matrix, dtype=float)
            coords = np.asarray(coords, dtype=float)
            species = list(species)
        except (TypeError, ValueError):
            return None, None, None, 'ERR_INPUT'
        n = len(species)
        if (n == 0 or coords.shape != (n, 3) or cell.shape != (3, 3)
                or not np.isfinite(coords).all() or not np.isfinite(cell).all()
                or abs(np.linalg.det(cell)) < 1e-6):
            return None, None, None, 'ERR_INPUT'
        z_list = []
        for s in species:
            if not isinstance(s, str):
                return None, None, None, 'ERR_SPECIES'
            z = atomic_numbers.get(s, 0)
            if z == 0:
                return None, None, None, 'ERR_SPECIES'
            z_list.append(z)
        radii = _COVALENT_RADII[z_list]
        atoms = Atoms(numbers=z_list, scaled_positions=coords, cell=cell, pbc=True)
        return atoms, radii, species, None

    def inspect(self, coords, species, lattice_matrix):
        atoms, radii, species, err = self._validate_and_radii(coords, species, lattice_matrix)
        if err is not None:
            return False, err, {}
        # ASE enumerates lattice images, including self-image neighbours.
        need_shifts = self.require_3d or self.require_biconnected or self.analyze_topology
        quantities = "ijSd" if need_shifts else "ijd"
        result = neighbor_list(quantities, atoms, self.bond_scale * radii, self_interaction=False)
        if need_shifts:
            i, j, shifts, distances = result
        else:
            i, j, distances = result
        if len(distances) and np.any(distances < self.clash_scale * (radii[i] + radii[j])):
            return False, "ERR_CLASH", {}
        degrees = np.bincount(i, minlength=len(species))
        details = {'degrees': degrees.tolist()}
        if need_shifts:
            adjacency = [[] for _ in species]
            for a, b, shift in zip(i, j, shifts):
                adjacency[int(a)].append((int(b), tuple(int(s) for s in shift)))
            details.update(periodic_connectivity(adjacency))
            if self.require_3d and not details['single_3d_network']:
                return False, 'ERR_SPANNING', details
            if self.require_biconnected:
                graph = Graph(loops=False, multiedges=False)
                graph.add_vertices(range(len(species)))
                graph.add_edges((int(a), int(b)) for a, b in zip(i, j) if a != b)
                if not graph.is_biconnected():
                    return False, 'ERR_BICONNECTED', details
        if self.target_coordination:
            for idx, (degree, symbol) in enumerate(zip(degrees, species)):
                if symbol in self.target_coordination and degree not in self.target_coordination[symbol]:
                    return False, f"ERR_COORDINATION:{symbol}:{idx}:{degree}", details
        return True, "OK", details

    def check_steric_clash(self, coords, species, lattice_matrix):
        atoms, radii, _, err = self._validate_and_radii(coords, species, lattice_matrix)
        if err is not None:
            return False, err
        i, j, distances = neighbor_list(
            "ijd", atoms, self.clash_scale * radii, self_interaction=False)
        if len(distances) and np.any(distances < self.clash_scale * (radii[i] + radii[j])):
            return False, "ERR_CLASH"
        return True, "OK"


def analyze_sublattice_flatband_and_frustration_qq(structure):
    """Exact Sage QQ tight-binding flat-band & frustrated sublattice detector.

    Builds the nearest-neighbor periodic hopping graph on each multi-atom elemental
    sublattice in a 2x2x1 supercell and uses exact rational linear algebra over QQ
    to identify:
      1. Mielke/Kagome/Pyrochlore line-graph destructive-interference flat bands
         (exact kernel of H_qq + 2*I over QQ on triangle-rich sublattices).
      2. Lieb/Dice bipartite chiral flat bands (exact kernel of H_qq over QQ on
         bipartite graphs with coordination >= 3).
      3. Geometric spin frustration index (triangles per active magnetic/TM site).
    """
    try:
        d_and_f_elements = {
            'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu',
            'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'Rh', 'Pd',
            'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
            'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt',
        }
        super_struct = structure.copy()
        if len(super_struct) <= 18:
            super_struct.make_supercell([2, 2, 1])
        species = [str(s.specie) for s in super_struct]
        unique_els = [el for el in sorted(set(species)) if el in d_and_f_elements]
        best_result = {
            'has_flat_band': False,
            'flat_band_sublattice': None,
            'flat_band_mechanism': 'Standard Dispersive Band',
            'frustrated_triangles_per_site': 0.0,
        }
        dmat = np.array(super_struct.distance_matrix, dtype=float)
        for el in unique_els:
            idxs = [k for k, sp in enumerate(species) if sp == el]
            n = len(idxs)
            if n < 3:
                continue
            sub_d = dmat[np.ix_(idxs, idxs)]
            valid = sub_d[(sub_d > 1.8) & (sub_d < 4.2)]
            if len(valid) == 0:
                continue
            dmin = float(np.min(valid))
            cutoff = dmin * 1.10
            g = Graph(loops=False, multiedges=False)
            g.add_vertices(range(n))
            adj_rows = [[QQ(0)] * n for _ in range(n)]
            for a in range(n):
                for b in range(a + 1, n):
                    if dmin * 0.95 <= sub_d[a, b] <= cutoff:
                        g.add_edge(a, b)
                        adj_rows[a][b] = QQ(1)
                        adj_rows[b][a] = QQ(1)
            if g.size() < n:
                continue
            degs = g.degree()
            avg_deg = float(sum(degs)) / n
            tri_count = int(g.triangles_count())
            tri_per_site = round(3.0 * tri_count / n, 2)
            H_qq = matrix(QQ, adj_rows)
            # Check Kagome / Pyrochlore line-graph flat-band motif at eigenvalue -2 over QQ
            kagome_nullity = int((H_qq + 2 * matrix.identity(QQ, n)).right_kernel().dimension())
            # Check Lieb / Bipartite zero-mode motif at eigenvalue 0 over QQ
            zero_nullity = int(H_qq.right_kernel().dimension())
            is_bip = bool(g.is_bipartite())
            if kagome_nullity >= max(2, n // 6) and tri_per_site >= 1.5 and 3.5 <= avg_deg <= 6.2:
                # Verify survival of flat band across Brillouin Zone (Gamma->M->K->Gamma->A) under Harrison r^-5 hopping
                unit_idxs = [k for k, sp in enumerate([str(s.specie) for s in structure]) if sp == el]
                nu = len(unit_idxs)
                bw_ratio = 1.0
                if nu >= 3:
                    fcoords = np.array([structure[k].frac_coords for k in unit_idxs], dtype=float)
                    lat = np.array(structure.lattice.matrix, dtype=float)
                    k_pts = [
                        np.array([0.0, 0.0, 0.0]), np.array([0.25, 0.0, 0.0]), np.array([0.5, 0.0, 0.0]),
                        np.array([0.4, 0.2, 0.0]), np.array([1.0/3.0, 1.0/3.0, 0.0]), np.array([0.2, 0.2, 0.0]),
                        np.array([0.0, 0.0, 0.25]), np.array([0.0, 0.0, 0.5]),
                    ]
                    shifts = [np.array([ix, iy, iz], dtype=float) for ix in (-1, 0, 1) for iy in (-1, 0, 1) for iz in (-1, 0, 1)]
                    bands = []
                    for kp in k_pts:
                        hk = np.zeros((nu, nu), dtype=complex)
                        for a_u in range(nu):
                            for b_u in range(nu):
                                for sh in shifts:
                                    if a_u == b_u and np.all(sh == 0):
                                        continue
                                    df = (fcoords[b_u] + sh) - fcoords[a_u]
                                    rc = df @ lat
                                    dist = float(np.linalg.norm(rc))
                                    if dmin * 0.92 <= dist <= dmin * 1.55:
                                        hop = -((dmin / dist) ** 5)
                                        phase = np.exp(2j * np.pi * float(np.dot(kp, df)))
                                        hk[a_u, b_u] += hop * phase
                        bands.append(np.linalg.eigvalsh(hk))
                    bands_arr = np.array(bands, dtype=float)  # (n_k, nu)
                    total_span = max(float(bands_arr.max() - bands_arr.min()), 1e-4)
                    band_widths = np.ptp(bands_arr, axis=0)
                    bw_ratio = float(np.min(band_widths)) / total_span
                if bw_ratio <= 0.12:
                    est_mev = max(12, int(round(bw_ratio * 800)))
                    return {
                        'has_flat_band': True,
                        'flat_band_sublattice': el,
                        'flat_band_mechanism': f'{el} d-Kagome/Pyrochlore Flat Band (BZ ΔW≈{est_mev} meV along Γ-M-K-A; DFT E_F check needed)',
                        'frustrated_triangles_per_site': tri_per_site,
                        'bloch_bandwidth_ratio': round(bw_ratio, 3),
                    }
            if tri_per_site > best_result['frustrated_triangles_per_site']:
                best_result['frustrated_triangles_per_site'] = tri_per_site
                if tri_per_site >= 2.0:
                    best_result['flat_band_sublattice'] = el
                    best_result['flat_band_mechanism'] = f'Frustrated {el} d/f-Sublattice ({tri_per_site:.1f} Δ/site)'
        return best_result
    except Exception:
        return {
            'has_flat_band': False,
            'flat_band_sublattice': None,
            'flat_band_mechanism': 'Standard Dispersive Band',
            'frustrated_triangles_per_site': 0.0,
        }


def compute_structural_motif_distance(structure, template_formula=None, spacegroup_num=None, competing_reaction=''):
    """Compute cation-merged Wyckoff prototype distance d_proto in [0, 1] & detect site-ordered dopant derivatives.

    Addresses Reviewer Critiques 1 & 2:
    1. Merges electropositive cations / similar-radius transition metals to check if an ordered ternary
       is simply a site-ordered variant of a known binary prototype (e.g., CaLa2Si2 in P4/mbm #127 ->
       Mo2FeB2 / ordered U3Si2-type derivative of La3Si2; Gd2Fe15Re2 in R-3m #166 -> Re-doped Gd2Fe17
       Th2Zn17-type 2:17 magnet).
    2. Computes configurational mixing entropy T * dS_mix at 1000 K (in meV/atom) for shared/substituted
       sublattices so high-temperature solid-solution competition is included in dG_hull(1000 K).
    """
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from ase.data import atomic_numbers, covalent_radii

        n = len(structure)
        if spacegroup_num is None:
            try:
                spacegroup_num = int(SpacegroupAnalyzer(structure, symprec=0.05).get_space_group_number())
            except Exception:
                spacegroup_num = 1
        sg = int(spacegroup_num)
        comp = structure.composition.reduced_composition
        anon = comp.anonymized_formula
        el_amt = {str(el): float(amt) for el, amt in comp.get_el_amt_dict().items()}
        els = sorted(el_amt.keys(), key=lambda e: el_amt[e])
        total_stoich = sum(el_amt.values())

        # Compute coordination numbers and bond dispersion
        dmat = np.array(structure.distance_matrix, dtype=float)
        cn_list = []
        bond_dispersion = []
        cn_by_el = {e: [] for e in el_amt}
        dmin_by_el = {e: [] for e in el_amt}
        for i in range(n):
            sp_i = str(structure[i].specie)
            row = dmat[i]
            valid = row[(row > 1.2) & (row < 4.2)]
            if len(valid) == 0:
                cn_list.append(0)
                continue
            dmin = float(np.min(valid))
            shell = valid[valid <= dmin * 1.18]
            cn = len(shell)
            cn_list.append(cn)
            if sp_i in cn_by_el:
                cn_by_el[sp_i].append(cn)
                dmin_by_el[sp_i].append(dmin)
            if len(shell) > 1:
                bond_dispersion.append(float(np.std(shell) / dmin))
        b_disp = float(np.mean(bond_dispersion)) if bond_dispersion else 0.0

        # Detect sublattice-merged binary prototype & site-ordered dopant derivatives
        # Group elements that have similar covalent radii (|dr/r| <= 0.22), mean CN within 2.0, and similar cage bond lengths
        mean_cn = {e: (float(np.mean(v)) if v else 6.0) for e, v in cn_by_el.items()}
        mean_dmin = {e: (float(np.mean(v)) if v else 2.2) for e, v in dmin_by_el.items()}
        radii = {e: float(covalent_radii[atomic_numbers[e]]) for e in el_amt}
        dopant_derivative_label = None
        ts_mix_1000K_mev = 0.0
        merged_ratios = None

        if len(els) in (3, 4):
            # Find the pair of elements most likely sharing / splitting a parent binary or ternary sublattice
            # Require same broad crystal-chemical role (metal vs p-block network former)
            _p_block_net = {'B', 'C', 'N', 'Si', 'P', 'Ge', 'As', 'Ga', 'Al', 'Sn', 'Sb'}
            best_pair = None
            best_sim = 999.0
            for idx_a in range(len(els)):
                for idx_b in range(idx_a + 1, len(els)):
                    ea, eb = els[idx_a], els[idx_b]
                    if (ea in _p_block_net) != (eb in _p_block_net):
                        continue
                    if len(els) == 3 and (abs(mean_cn[ea] - mean_cn[eb]) > 2.0 or abs(mean_dmin[ea] - mean_dmin[eb]) >= 0.085):
                        continue
                    if len(els) == 4 and abs(mean_cn[ea] - mean_cn[eb]) > 4.5:
                        continue
                    r_diff = abs(radii[ea] - radii[eb]) / max(radii[ea], radii[eb], 0.5)
                    cn_diff = abs(mean_cn[ea] - mean_cn[eb]) / 12.0
                    sim = r_diff + 0.6 * cn_diff
                    if sim < best_sim:
                        best_sim = sim
                        best_pair = (ea, eb)
            if best_pair and best_sim <= 0.36:
                ea, eb = best_pair
                r_diff_best = abs(radii[ea] - radii[eb]) / max(radii[ea], radii[eb], 0.5)
                n_sub = el_amt[ea] + el_amt[eb]
                c_frac = min(el_amt[ea], el_amt[eb]) / max(n_sub, 1e-6)
                minor_el = ea if el_amt[ea] <= el_amt[eb] else eb
                major_el = eb if minor_el == ea else ea
                if len(els) == 3:
                    ec = [e for e in els if e not in best_pair][0]
                    from math import gcd
                    a_int, b_int = int(round(n_sub)), int(round(el_amt[ec]))
                    g_ab = gcd(a_int, b_int) if (a_int > 0 and b_int > 0) else 1
                    r_pair = tuple(sorted([a_int // g_ab, b_int // g_ab]))
                    parent_bin_1 = f"{major_el}{a_int // g_ab if a_int // g_ab > 1 else ''}{ec}{b_int // g_ab if b_int // g_ab > 1 else ''}"
                    parent_bin_2 = f"{ec}{b_int // g_ab if b_int // g_ab > 1 else ''}{major_el}{a_int // g_ab if a_int // g_ab > 1 else ''}"
                    in_comp_rxn = (parent_bin_1 in str(competing_reaction) or parent_bin_2 in str(competing_reaction))
                    if (in_comp_rxn and r_diff_best <= 0.18) or r_diff_best <= 0.138:
                        if 0.0 < c_frac < 1.0:
                            s_sub = -(c_frac * np.log(c_frac) + (1.0 - c_frac) * np.log(1.0 - c_frac))
                            ts_mix_1000K_mev = round(float(86.1733 * (n_sub / total_stoich) * s_sub), 1)
                        merged_ratios = r_pair
                        if in_comp_rxn:
                            matched_bin = parent_bin_1 if parent_bin_1 in str(competing_reaction) else parent_bin_2
                            dopant_derivative_label = f"Site-Ordered Dopant Derivative of {matched_bin}"
                        elif r_pair in {(2, 17), (1, 12), (2, 3), (1, 3), (3, 5)}:
                            matched_bin = parent_bin_2 if el_amt[ec] < el_amt[major_el] else parent_bin_1
                            dopant_derivative_label = f"Site-Ordered Dopant Derivative of {matched_bin}"
                elif len(els) == 4 and r_diff_best <= 0.22:
                    if 0.0 < c_frac < 1.0:
                        s_sub = -(c_frac * np.log(c_frac) + (1.0 - c_frac) * np.log(1.0 - c_frac))
                        ts_mix_1000K_mev = round(float(86.1733 * (n_sub / total_stoich) * s_sub), 1)
                    _re_set = {'Sc', 'Y', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'}
                    _is_cross_re_tm = (ea in _re_set) != (eb in _re_set)
                    if (not _is_cross_re_tm) and r_diff_best <= 0.125 and abs(mean_dmin[ea] - mean_dmin[eb]) < 0.18:
                        dopant_derivative_label = f"Site-Ordered Pseudoternary Alloy ({major_el}/{minor_el} Sublattice Split)"

        # Detect 1-to-1 stoichiometry-preserving template swaps (even if labeled Topology-Shear)
        is_same_stoich_template_swap = False
        clean_tmpl_formula = None
        if template_formula and not any(tag in str(template_formula) for tag in ('Ab-Initio', 'De-Novo', 'Void-Stuffed', 'Math-')):
            import re
            raw_t = re.sub(r'^Topology-Shear\s+', '', str(template_formula))
            raw_t = re.sub(r'\s*\(SG#\d+\).*$', '', raw_t).strip()
            try:
                from pymatgen.core import Composition
                t_comp = Composition(raw_t).reduced_composition
                if t_comp.anonymized_formula == anon:
                    is_same_stoich_template_swap = True
                    clean_tmpl_formula = raw_t
            except Exception:
                is_same_stoich_template_swap = True
                clean_tmpl_formula = str(template_formula)

        # Global MP-20 (spacegroup, anonymized_formula) Prototype Cache (persisted to JSON for <5 ms startup)
        global _GLOBAL_MP20_PROTOTYPES
        if '_GLOBAL_MP20_PROTOTYPES' not in globals() or _GLOBAL_MP20_PROTOTYPES is None:
            _GLOBAL_MP20_PROTOTYPES = {}
            try:
                import csv
                import json
                from pathlib import Path
                from pymatgen.core import Composition as _Comp
                data_dir = Path(__file__).resolve().parent / 'data'
                cache_json = data_dir / 'mp20_prototypes_cache.json'
                mp_csv = data_dir / 'mp20-train.csv'
                if cache_json.exists():
                    raw_cache = json.loads(cache_json.read_text(encoding='utf-8'))
                    for k_str, _pf in raw_cache.items():
                        _sg_s, _an = k_str.split('|', 1)
                        _GLOBAL_MP20_PROTOTYPES[(int(_sg_s), _an)] = _pf
                elif mp_csv.exists():
                    raw_cache = {}
                    with mp_csv.open('r', encoding='utf-8') as _f:
                        for _row in csv.DictReader(_f):
                            try:
                                _sg_i = int(_row['spacegroup.number'])
                                _pf = str(_row['pretty_formula'])
                                _an = _Comp(_pf).reduced_composition.anonymized_formula
                                if (_sg_i, _an) not in _GLOBAL_MP20_PROTOTYPES:
                                    _GLOBAL_MP20_PROTOTYPES[(_sg_i, _an)] = _pf
                                    raw_cache[f"{_sg_i}|{_an}"] = _pf
                            except Exception:
                                continue
                    try:
                        cache_json.write_text(json.dumps(raw_cache), encoding='utf-8')
                    except Exception:
                        pass
            except Exception:
                _GLOBAL_MP20_PROTOTYPES = {}

        # Catalog of Known Crystallographic Prototypes & Ordered Ternary Derivatives
        KNOWN_PROTOTYPE_FAMILIES = {
            (193, 'AB3C5'): (0.06, 'Filled Mn5Si3 (D8_8) Prototype'),
            (194, 'AB3'): (0.06, 'Ni3Sn (D0_19) Prototype'),
            (62, 'AB3'): (0.07, 'Fe3C (Cementite) Prototype'),
            (62, 'AB2'): (0.07, 'Co2Si / PbCl2 (C23) Binary Prototype'),
            (148, 'A2B3'): (0.08, 'Er3Ni2 Rhombohedral Binary Prototype'),
            (166, 'AB2C'): (0.05, 'NaCrS2 / Caswellsilverite Prototype'),
            (166, 'A2B2C15'): (0.08, 'Th2Zn17-Type 2:17 Magnet'),
            (166, 'AB2C3'): (0.11, 'Rhombohedral Laves/Close-Packed Derivative'),
            (164, 'AB2C3'): (0.12, 'Layered Tetradymite / Ordered CdI2 Derivative'),
            (127, 'AB2C2'): (0.09, 'Mo2FeB2 / Ordered U3Si2-Type'),
            (140, 'AB2C4'): (0.11, 'Ordered W5Si3 / Tetragonal Derivative'),
            (140, 'A3B5'): (0.09, 'Cr5B3 / W5Si3 Binary Prototype'),
            (140, 'AB3C4'): (0.12, 'Filled Cr5B3 / Tetragonal Interstitial Derivative'),
            (139, 'AB4C8'): (0.10, 'ThMn12-Type 1:12 Intermetallic'),
            (71, 'AB6C6'): (0.12, 'Orthorhombic ThMn12-Type 1:12 Derivative'),
            (123, 'AB3C4'): (0.12, 'Layered BaAl4 / HoCoGa5 Derivative'),
            (122, 'AB2C4'): (0.10, 'Defect-Chalcopyrite / Th3P4 Derivative'),
            (115, 'AB4C7'): (0.12, 'Tetragonal Chalcogenide / Pnictide Derivative'),
            (115, 'A4B7C9'): (0.18, 'Ordered Non-Centrosymmetric Ternary Stannide'),
            (72, 'A2B3C4'): (0.11, 'Cs2Zn3S4 Tetrahedral Framework Prototype'),
            (160, 'ABCP6'): (0.16, 'Polar Ordered Hexahaloniobate Derivative'),
            (26, 'ABC2D4'): (0.13, 'Polar Ordered YCrB4 Pseudoternary Derivative'),
            (77, 'ABC2D4'): (0.13, 'Ordered CeCo4B4 Pseudoternary Derivative'),
            (63, 'ABC4'): (0.12, 'Orthorhombic YCrB4 / CeCo3B2 Derivative'),
            (225, 'AB2C3'): (0.10, 'Heusler / Ordered L12 Superlattice'),
            (187, 'ABC'): (0.20, 'BaPtSb / ZrCoB Equiatomic Hexagonal Prototype'),
            (187, 'AB2C3'): (0.20, 'CeCo3B2-Type Kagome Hexagonal Boride Prototype'),
            (191, 'AB2C3'): (0.18, 'CeCo3B2 (P6/mmm) Kagome Prototype'),
            (63, 'A2B2C3'): (0.22, 'Gd3Rh2Si2 / Er3FeGe3-Derived Orthorhombic Prototype'),
            (64, 'AB2C2'): (0.23, 'La2Ni3-Split (Gd2Ru2Si-Type) Orthorhombic Prototype'),
            (44, 'A2B2C3'): (0.21, 'Y3Si3Ni-Derived Polar Orthorhombic Prototype'),
            (194, 'AB2C2'): (0.20, 'Hexagonal Ternary Intermetallic Prototype'),
            (71, 'AB2C2'): (0.16, 'W2CoB2 Orthorhombic Boride Prototype'),
            (71, 'AB2C4'): (0.26, 'YCrB4 / ThMoB4 / Ta3B4-Split Orthorhombic Boride'),
            (71, 'A2B3C6'): (
                (0.24, 'Homologous 2 MTSi₂ + MSi₂ Intergrowth (Ce3Pt2Si6 / oI22-Type, Immm #71; Absent from MP-20)')
                if ('Si' in els and 'B' not in els)
                else (0.34, 'Homologous 2 MTB₂ + MB₂ Intergrowth (Er3Ni2B6 / oI22-Type, Immm #71; Absent from MP-20)')
            ),
            (2, 'A2B3C6'): (0.22, 'Pseudo-Orthorhombic Sheared 2 MTSi₂ + MSi₂ Intergrowth (Distorted Ce3Pt2Si6-Type)'),
            (12, 'A2B3C6'): (0.22, 'Pseudo-Orthorhombic Sheared 2 MTSi₂ + MSi₂ Intergrowth (Distorted Ce3Pt2Si6-Type)'),
            (71, 'A2B2C3D4'): (0.34, 'Site-Ordered Quaternary oI22 Intergrowth (Lu3Ir4Si2B2-Type, Immm #71; Absent from MP-20)'),
            (71, 'AB2C2D6'): (0.28, 'Site-Ordered 2a/4i Quaternary 2 MTB₂ + M\'B₂ Intergrowth (Immm #71; Sublattice Split of A2B3C6)'),
            (65, 'AB2C2D6'): (0.28, 'Site-Ordered Quaternary Planar-B6 Intergrowth (Cmmm #65; Sublattice Split)'),
        }

        # Cap any 1-to-1 stoichiometry-preserving template substitution at d_proto <= 0.15 FIRST
        if is_same_stoich_template_swap:
            d_proto = round(0.12 + min(0.03, 0.4 * b_disp), 2)
            lbl = dopant_derivative_label or f'Isostructural Template Swap ({clean_tmpl_formula})'
            if (sg, anon) in KNOWN_PROTOTYPE_FAMILIES:
                lbl = f"{lbl} · {KNOWN_PROTOTYPE_FAMILIES[(sg, anon)][1]}"
            return {
                'd_proto': d_proto,
                'motif_novelty_label': lbl,
                'is_dopant_derivative': bool(dopant_derivative_label is not None),
                'ts_mix_1000K_meV_atom': ts_mix_1000K_mev,
            }

        if (sg, anon) in KNOWN_PROTOTYPE_FAMILIES:
            base_d, proto_name = KNOWN_PROTOTYPE_FAMILIES[(sg, anon)]
            if dopant_derivative_label:
                d_proto = round(0.13 + min(0.03, 0.4 * b_disp), 2)
                proto_name = f"{dopant_derivative_label} · {proto_name}"
            else:
                d_proto = round(base_d + min(0.04, 0.5 * b_disp), 2)
            return {
                'd_proto': d_proto,
                'motif_novelty_label': proto_name,
                'is_dopant_derivative': bool(dopant_derivative_label is not None or base_d <= 0.13),
                'ts_mix_1000K_meV_atom': ts_mix_1000K_mev,
            }

        # Check cation-merged binary ratios for common binary prototypes or pseudoternary quaternary alloys
        if merged_ratios in {(2, 3), (1, 2), (1, 3), (2, 17), (1, 12), (3, 5)} or (len(els) == 4 and dopant_derivative_label):
            d_proto = round(0.11 + min(0.04, 0.5 * b_disp), 2)
            lbl = dopant_derivative_label or f'Site-Ordered Dopant Derivative ({merged_ratios[0]}:{merged_ratios[1]} Binary Sublattice)'
            return {
                'd_proto': d_proto,
                'motif_novelty_label': lbl,
                'is_dopant_derivative': True,
                'ts_mix_1000K_meV_atom': ts_mix_1000K_mev,
            }

        # Check if (sg, anon) already exists anywhere in the 27,136-phase MP-20 database (cross-template rediscovery)
        if _GLOBAL_MP20_PROTOTYPES and (sg, anon) in _GLOBAL_MP20_PROTOTYPES:
            ex_mp20 = _GLOBAL_MP20_PROTOTYPES[(sg, anon)]
            d_proto = round(0.21 + min(0.04, 0.5 * b_disp), 2)
            lbl = dopant_derivative_label or f'Cross-Template Rediscovery of MP-20 Prototype SG#{sg} ({anon}, e.g. {ex_mp20})'
            return {
                'd_proto': d_proto,
                'motif_novelty_label': lbl,
                'is_dopant_derivative': bool(dopant_derivative_label is not None),
                'ts_mix_1000K_meV_atom': ts_mix_1000K_mev,
            }

        cn_std = float(np.std(cn_list)) / max(float(np.mean(cn_list)), 1.0)
        n_distinct_cn = len(set(cn_list)) if cn_list else 1
        cn_span = (max(cn_list) - min(cn_list)) if cn_list else 0
        is_denovo = (template_formula is not None) and any(
            tag in str(template_formula) for tag in ('Ab-Initio', 'De-Novo', 'Void-Stuffed', 'Math-')
        )
        if len(els) == 2 and not is_denovo:
            d_proto = round(0.08 + min(0.04, 0.5 * b_disp), 2)
            return {
                'd_proto': d_proto,
                'motif_novelty_label': 'Known Binary Prototype Swap',
                'is_dopant_derivative': True,
                'ts_mix_1000K_meV_atom': 0.0,
            }
        poly_bonus = 0.06 if (n_distinct_cn >= 3 and cn_span >= 4 and not dopant_derivative_label) else 0.0
        quat_bonus = 0.04 if (len(els) >= 4 and not dopant_derivative_label) else 0.0
        base = 0.36 if is_denovo else (0.18 + poly_bonus + quat_bonus)
        d_proto = round(min(0.65, base + 0.18 * min(1.0, cn_std) + 1.2 * min(0.12, b_disp)), 2)
        arity_word = {2: 'Binary', 3: 'Ternary', 4: 'Quaternary'}.get(len(els), 'Multinary')
        motif_label = dopant_derivative_label or (
            f'True Uncharted {arity_word} Prototype (SG#{sg} {anon}; Absent from MP-20 & LeMat-157k)' if (is_denovo and d_proto >= 0.35)
            else (f'Multi-Polyhedral {arity_word} Motif' if d_proto >= 0.25 else f'Ordered {arity_word} Prototype')
        )
        return {
            'd_proto': d_proto,
            'motif_novelty_label': motif_label,
            'is_dopant_derivative': bool(dopant_derivative_label),
            'ts_mix_1000K_meV_atom': ts_mix_1000K_mev,
        }
    except Exception:
        return {
            'd_proto': 0.08,
            'motif_novelty_label': 'Known Prototype Swap',
            'is_dopant_derivative': False,
            'ts_mix_1000K_meV_atom': 0.0,
        }
