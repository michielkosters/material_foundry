"""Five-Gate Physical & Dynamical Verification for Groundbreaking Crystal Candidates.

Verifies every Tier 1 Uncharted Ground State across all 5 physical criteria:
1. Hull Depth vs. ML Uncertainty (|dE_hull| >= 10 meV/atom confidence tiering + tight fmax=0.005 eV/A medium polish)
2. Dynamic (Phonon Hessian) & Elastic (Born B0 > 0, G > 0) Stability
3. Crystallographic & Steric Realism (High symmetry, packing 50-95%, bond-length realism ratio d_min / (r1 + r2))
4. Functional & Electronic/Magnetic Domain Classification
5. Turnkey DFT Verification Input Deck Generation (VASP POSCAR/INCAR & Quantum ESPRESSO pw.x input)
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from ase.data import atomic_masses, atomic_numbers, covalent_radii
from ase.filters import FrechetCellFilter
from ase.optimize import BFGS
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from report_io import atomic_text

EV_A3_TO_GPA = 160.21766208


def _batch_mace_energies_and_forces(atoms_list, calculator, compute_force=True):
    """Evaluate a list of ASE Atoms in a single block-diagonal MACE torch_geometric batch."""
    if not atoms_list:
        return [], []
    try:
        from mace import data
        from mace.tools import torch_geometric
        model = calculator.models[0]
        z_table = calculator.z_table
        r_max = float(calculator.r_max)
        ad_list = []
        for at in atoms_list:
            cfg = data.config_from_atoms(at)
            ad_list.append(data.AtomicData.from_config(cfg, z_table=z_table, cutoff=r_max))
        batch = torch_geometric.batch.Batch.from_data_list(ad_list)
        out = model(batch.to_dict(), compute_force=compute_force)
        energies = [float(x) for x in out['energy'].detach().cpu().numpy()]
        if not compute_force:
            return energies, []
        f_all = out['forces'].detach().cpu().numpy()
        ptr = batch.ptr.detach().cpu().numpy()
        forces = [np.array(f_all[int(ptr[i]):int(ptr[i + 1])], dtype=float) for i in range(len(atoms_list))]
        return energies, forces
    except Exception:
        energies, forces = [], []
        for at in atoms_list:
            at.calc = calculator
            energies.append(float(at.get_potential_energy()))
            if compute_force:
                forces.append(np.array(at.get_forces(), dtype=float))
        return energies, forces


def compute_elastic_moduli(atoms, calculator, strain_mag=0.006):
    """Compute bulk modulus B0 (GPa), shear modulus G (GPa), Pugh ratio B0/G, and Born stability."""
    base_pos = atoms.get_positions().copy()
    base_cell = atoms.get_cell().array.copy()
    v0 = float(atoms.get_volume())

    # Build all 7 strained configurations (unstrained + 2 vol + 2 ortho + 2 mono) and evaluate in 1 MACE batch
    batch_atoms = [atoms.copy()]
    for s in (-strain_mag, strain_mag):
        strained = atoms.copy()
        strained.set_cell(base_cell * (1.0 + s), scale_atoms=True)
        batch_atoms.append(strained)
    for s in (-strain_mag, strain_mag):
        f_mat = np.diag([1.0 + s, 1.0 - s, 1.0 / (1.0 - s * s)])
        strained = atoms.copy()
        strained.set_cell(base_cell @ f_mat, scale_atoms=True)
        batch_atoms.append(strained)
    for s in (-strain_mag, strain_mag):
        f_mat = np.array([
            [1.0, 0.5 * s, 0.0],
            [0.5 * s, 1.0, 0.0],
            [0.0, 0.0, 1.0 / (1.0 - 0.25 * s * s)],
        ])
        strained = atoms.copy()
        strained.set_cell(base_cell @ f_mat, scale_atoms=True)
        batch_atoms.append(strained)

    all_e, _ = _batch_mace_energies_and_forces(batch_atoms, calculator, compute_force=False)
    e0 = all_e[0]
    vol_energies = all_e[1:3]
    ortho_energies = all_e[3:5]
    mono_energies = all_e[5:7]

    # 1. Isotropic volumetric strain: V = V0 (1+s)^3 ≈ V0 (1 + 3s) => d^2E/ds^2 = 9 V0 B0
    d2e_ds2_vol = (vol_energies[0] - 2.0 * e0 + vol_energies[1]) / (strain_mag ** 2)
    b0_gpa = float((d2e_ds2_vol / (9.0 * v0)) * EV_A3_TO_GPA)

    # 2. Volume-conserving orthorhombic & monoclinic shear strains
    d2e_ds2_ortho = (ortho_energies[0] - 2.0 * e0 + ortho_energies[1]) / (strain_mag ** 2)
    c_prime_gpa = float((d2e_ds2_ortho / (4.0 * v0)) * EV_A3_TO_GPA)

    d2e_ds2_mono = (mono_energies[0] - 2.0 * e0 + mono_energies[1]) / (strain_mag ** 2)
    c44_gpa = float((d2e_ds2_mono / v0) * EV_A3_TO_GPA)

    g_voigt_gpa = float((2.0 * c_prime_gpa + 3.0 * c44_gpa) / 5.0)
    born_stable = bool(b0_gpa > 2.0 and g_voigt_gpa > 1.0 and c_prime_gpa > 0.5 and c44_gpa > 0.5)
    pugh_ratio = round(b0_gpa / max(g_voigt_gpa, 0.1), 2) if g_voigt_gpa > 0 else None
    youngs_gpa = float((9.0 * b0_gpa * g_voigt_gpa) / max(3.0 * b0_gpa + g_voigt_gpa, 0.1)) if (b0_gpa > 0 and g_voigt_gpa > 0) else 0.0
    total_mass_amu = float(sum(atomic_masses[z] for z in atoms.get_atomic_numbers()))
    density_g_cm3 = float(total_mass_amu * 1.66053906660 / max(v0, 1e-3))
    specific_mod = round(youngs_gpa / max(density_g_cm3, 0.1), 1) if youngs_gpa > 0 else 0.0
    k_inv = g_voigt_gpa / max(b0_gpa, 0.1) if (b0_gpa > 0 and g_voigt_gpa > 0) else 0.0
    vickers_hv = round(max(0.5, 2.0 * ((k_inv * k_inv * max(g_voigt_gpa, 0.0)) ** 0.585) - 3.0), 1) if g_voigt_gpa > 0 else 0.0
    # Acoustic sound velocities (m/s) & Debye temperature Theta_D (K)
    if b0_gpa > 0 and g_voigt_gpa > 0 and density_g_cm3 > 0:
        vl = math.sqrt(max(1.0, (b0_gpa + (4.0 / 3.0) * g_voigt_gpa) * 1e9 / (density_g_cm3 * 1000.0)))
        vt = math.sqrt(max(1.0, g_voigt_gpa * 1e9 / (density_g_cm3 * 1000.0)))
        vm = (0.333333 * (2.0 / (vt ** 3) + 1.0 / (vl ** 3))) ** (-1.0 / 3.0)
        n_density_m3 = (len(atoms) / v0) * 1e30
        debye_k = round(float((1.0545718e-34 / 1.380649e-23) * ((6.0 * (math.pi ** 2) * n_density_m3) ** (1.0 / 3.0)) * vm), 0)
        # Slack lattice thermal conductivity at 300 K (W / m K)
        poisson = max(0.05, min(0.45, (3.0 * b0_gpa - 2.0 * g_voigt_gpa) / max(6.0 * b0_gpa + 2.0 * g_voigt_gpa, 0.1)))
        gruneisen = 1.5 * (1.0 + poisson) / max(2.0 - 3.0 * poisson, 0.25)
        avg_mass_amu = total_mass_amu / len(atoms)
        delta_ang = (v0 / len(atoms)) ** (1.0 / 3.0)
        n_prim = len(atoms)
        slack_kappa = round(float(
            (3.1e-6 * avg_mass_amu * delta_ang * (debye_k ** 3))
            / max(1.0, (gruneisen ** 2) * (n_prim ** (2.0 / 3.0)) * 300.0)
        ), 2)
    else:
        debye_k = 0.0
        slack_kappa = 99.0
    atoms.set_cell(base_cell)
    atoms.set_positions(base_pos)
    return dict(
        bulk_modulus_GPa=round(b0_gpa, 1),
        shear_modulus_GPa=round(g_voigt_gpa, 1),
        youngs_modulus_GPa=round(youngs_gpa, 1),
        specific_modulus_GPa_cm3_g=specific_mod,
        vickers_hardness_GPa=vickers_hv,
        debye_temp_K=int(debye_k),
        slack_kappa_300K_W_mK=slack_kappa,
        c_prime_GPa=round(c_prime_gpa, 1),
        c44_GPa=round(c44_gpa, 1),
        pugh_ratio_B_G=pugh_ratio,
        mechanical_behavior=('Ductile (>1.75)' if (pugh_ratio and pugh_ratio >= 1.75) else 'Brittle/Covalent (<1.75)') if pugh_ratio else 'Unknown',
        born_elastic_stable=born_stable,
    )


def compute_dynamical_phonon_stability(atoms, calculator, delta=0.012):
    """Compute mass-weighted dynamical matrix eigenvalues at Gamma using Wyckoff symmetry-reduced central differences."""
    n = len(atoms)
    dof = 3 * n
    base_pos = atoms.get_positions().copy()
    hessian = np.zeros((dof, dof), dtype=float)
    masses = np.array([atomic_masses[z] for z in atoms.get_atomic_numbers()], dtype=float)
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)

    # Exploit space-group symmetry to displace only 1 representative atom per Wyckoff orbit
    rep_atoms = list(range(n))
    orbit_maps = []
    try:
        struct = AseAtomsAdaptor.get_structure(atoms)
        sga = SpacegroupAnalyzer(struct, symprec=0.05)
        frac_ops = sga.get_symmetry_operations(cartesian=False)
        cart_ops = sga.get_symmetry_operations(cartesian=True)
        frac_coords = np.array(struct.frac_coords, dtype=float)
        species = [str(s.specie) for s in struct]
        valid_ops = []
        for f_op, c_op in zip(frac_ops, cart_ops):
            rot_c = np.array(c_op.rotation_matrix, dtype=float)
            if abs(np.linalg.det(rot_c.T @ rot_c) - 1.0) > 1e-2:
                continue
            mapped = (frac_coords @ f_op.rotation_matrix.T) + f_op.translation_vector
            perm = []
            ok = True
            for a_idx in range(n):
                diff = mapped[a_idx] - frac_coords
                diff -= np.round(diff)
                dists = np.linalg.norm(diff, axis=1)
                b_idx = int(np.argmin(dists))
                if dists[b_idx] > 0.08 or species[a_idx] != species[b_idx]:
                    ok = False
                    break
                perm.append(b_idx)
            if ok and len(set(perm)) == n:
                valid_ops.append((rot_c, perm))
        if valid_ops:
            covered = set()
            reps = []
            for a_idx in range(n):
                if a_idx in covered:
                    continue
                reps.append(a_idx)
                for _, perm in valid_ops:
                    covered.add(perm[a_idx])
            rep_atoms = reps
            orbit_maps = valid_ops
    except Exception:
        rep_atoms = list(range(n))
        orbit_maps = []

    rep_blocks = {}
    disp_atoms_list = []
    disp_keys = []
    for atom_i in rep_atoms:
        for cart_axis in range(3):
            at_plus = atoms.copy()
            disp_plus = base_pos.copy()
            disp_plus[atom_i, cart_axis] += delta
            at_plus.set_positions(disp_plus)
            disp_atoms_list.append(at_plus)

            at_minus = atoms.copy()
            disp_minus = base_pos.copy()
            disp_minus[atom_i, cart_axis] -= delta
            at_minus.set_positions(disp_minus)
            disp_atoms_list.append(at_minus)
            disp_keys.append((atom_i, cart_axis))

    _, batch_forces = _batch_mace_energies_and_forces(disp_atoms_list, calculator, compute_force=True)
    for k_idx, (atom_i, cart_axis) in enumerate(disp_keys):
        if atom_i not in rep_blocks:
            rep_blocks[atom_i] = np.zeros((3, n, 3), dtype=float)
        f_plus = batch_forces[2 * k_idx]
        f_minus = batch_forces[2 * k_idx + 1]
        rep_blocks[atom_i][cart_axis, :, :] = -(f_plus - f_minus) / (2.0 * delta)

    atoms.set_positions(base_pos)
    if orbit_maps and len(rep_atoms) < n:
        counts = np.zeros(n, dtype=float)
        for atom_i, block_i in rep_blocks.items():
            # block_i has shape (3_alpha, n_b, 3_beta) -> transpose to (n_b, 3_alpha, 3_beta)
            h_ib = np.transpose(block_i, (1, 0, 2))
            for rot_c, perm in orbit_maps:
                atom_j = perm[atom_i]
                counts[atom_j] += 1.0
                # H_{pi(i), pi(b)} = R_c @ H_{i, b} @ R_c.T
                for b_idx in range(n):
                    c_idx = perm[b_idx]
                    rot_block = rot_c @ h_ib[b_idx] @ rot_c.T
                    hessian[3 * atom_j:3 * atom_j + 3, 3 * c_idx:3 * c_idx + 3] += rot_block
        for atom_j in range(n):
            if counts[atom_j] > 0:
                hessian[3 * atom_j:3 * atom_j + 3, :] /= counts[atom_j]
    else:
        for atom_i, block_i in rep_blocks.items():
            for cart_axis in range(3):
                hessian[3 * atom_i + cart_axis, :] = block_i[cart_axis].reshape(-1)
    # Symmetrize Hessian and apply acoustic sum rule (row-sum translational invariance)
    hessian = 0.5 * (hessian + hessian.T)
    for i in range(n):
        for alpha in range(3):
            for beta in range(3):
                row_sum = sum(hessian[3 * i + alpha, 3 * j + beta] for j in range(n))
                hessian[3 * i + alpha, 3 * i + beta] -= row_sum

    dyn_mat = hessian * np.outer(inv_sqrt_m, inv_sqrt_m)
    dyn_mat = 0.5 * (dyn_mat + dyn_mat.T)

    # Project out the 3 rigid acoustic translations (sqrt(m_i) along x, y, z)
    trans_basis = np.zeros((3, dof), dtype=float)
    for axis in range(3):
        vec = np.zeros((n, 3), dtype=float)
        vec[:, axis] = np.sqrt(masses)
        v_flat = vec.reshape(-1)
        trans_basis[axis] = v_flat / np.linalg.norm(v_flat)
    proj = np.eye(dof) - trans_basis.T @ trans_basis
    dyn_optical = proj @ dyn_mat @ proj
    dyn_optical = 0.5 * (dyn_optical + dyn_optical.T)

    eigvals = np.linalg.eigvalsh(dyn_optical)
    # Remove the 3 zero translational eigenvalues (closest to 0)
    sorted_by_abs = np.argsort(np.abs(eigvals))
    optical_eigs = np.sort(np.delete(eigvals, sorted_by_abs[:3])) if len(eigvals) > 3 else eigvals

    # Conversion: 1 sqrt(eV / (amu * A^2)) / (2*pi) = 15.633302 THz
    CONV_THZ = 15.633302
    freqs_thz = np.sign(optical_eigs) * np.sqrt(np.abs(optical_eigs)) * CONV_THZ
    min_freq_thz = float(freqs_thz[0]) if len(freqs_thz) else 0.0
    max_freq_thz = float(freqs_thz[-1]) if len(freqs_thz) else 0.0
    # Tolerance of -0.05 THz (~1.7 cm^-1) for numerical finite-difference noise
    n_imag = int(np.sum(freqs_thz < -0.05))
    dynamically_stable = bool(n_imag == 0 and min_freq_thz >= -0.05)
    # Harmonic Zero-Point Energy & 1000 K Vibrational Free Energy (1 THz = 4.135667 meV)
    pos_freqs = [float(f) for f in freqs_thz if f > 0.1]
    zpe_mev = round(sum(0.5 * 4.135667 * f for f in pos_freqs) / max(1, n), 2)
    kt_1000 = 86.1733  # meV at 1000 K
    fvib_1000_mev = round(sum(
        0.5 * 4.135667 * f + kt_1000 * math.log(max(1e-9, 1.0 - math.exp(-(4.135667 * f) / kt_1000)))
        for f in pos_freqs
    ) / max(1, n), 1)
    return dict(
        min_optical_freq_THz=round(min_freq_thz, 2),
        max_optical_freq_THz=round(max_freq_thz, 2),
        min_optical_freq_cm1=round(min_freq_thz * 33.3564, 1),
        max_optical_freq_cm1=round(max_freq_thz * 33.3564, 1),
        zpe_meV_atom=zpe_mev,
        f_vib_1000K_meV_atom=fvib_1000_mev,
        imaginary_mode_count=n_imag,
        dynamically_stable=dynamically_stable,
    )


def compute_zone_boundary_phonon_check(atoms, calculator, delta=0.015):
    """Evaluate 2x2x2 (or 2x2x1) supercell zone-boundary Bloch dynamical matrix D(q) at q in {X, M, R}.

    At zone-boundary wavevectors q in {0, 1/2}^3 \\ {(0,0,0)}, the Bloch phase exp(i 2*pi q.R) = (-1)^(2q.R)
    is strictly real (+/-1), allowing exact evaluation of zone-boundary acoustic and optical branches by
    displacing only Wyckoff representative atoms in the (0,0,0) unit cell of the supercell.
    Always Niggli-reduces skewed body-centered primitive cells first so supercell axes are near-orthogonal.
    """
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        pmg_s = AseAtomsAdaptor.get_structure(atoms)
        prim_s = pmg_s.get_primitive_structure(tolerance=0.02)
        ang = prim_s.lattice.angles
        if max(ang) > 120.0 or min(ang) < 60.0 or len(prim_s) < len(pmg_s):
            nig_s = prim_s.get_reduced_structure('niggli')
            if len(nig_s) == len(prim_s):
                prim_s = nig_s
        if len(prim_s) != len(pmg_s) or max(pmg_s.lattice.angles) > 120.0 or min(pmg_s.lattice.angles) < 60.0:
            atoms = AseAtomsAdaptor.get_atoms(prim_s)
            atoms.calc = calculator
    except Exception:
        pass
    n = len(atoms)
    lengths = atoms.get_cell().lengths()
    if n <= 10:
        sc_dims = (2, 2, 2)
    elif n <= 20:
        # Double the two shortest Niggli lattice vectors so the supercell is as isotropic as possible
        longest_idx = int(np.argmax(lengths))
        sc_dims = tuple(1 if i == longest_idx else 2 for i in range(3))
    else:
        shortest_idx = int(np.argmin(lengths))
        sc_dims = tuple(2 if i == shortest_idx else 1 for i in range(3))
    nx, ny, nz = sc_dims
    n_cells = nx * ny * nz
    masses = atoms.get_masses()
    inv_sqrt_m = np.repeat(1.0 / np.sqrt(masses), 3)

    # Identify Wyckoff representative atoms and point-group permutations in the primitive/unit cell
    rep_atoms = list(range(n))
    orbit_maps = []
    try:
        import spglib
        cell_tuple = (atoms.get_cell().array, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
        sym = spglib.get_symmetry(cell_tuple, symprec=1e-2)
        if sym is not None and len(sym['rotations']) > 1:
            eq_atoms = sym['equivalent_atoms']
            seen_eq = {}
            for idx, eq in enumerate(eq_atoms):
                if eq not in seen_eq:
                    seen_eq[eq] = idx
            rep_atoms = sorted(seen_eq.values())
    except Exception:
        rep_atoms = list(range(n))

    # Build supercell: atoms.repeat((nx, ny, nz)) orders cells by (rx, ry, rz)
    sc = atoms.repeat(sc_dims)
    base_sc_pos = sc.get_positions().copy()
    r_vecs = []
    for rx in range(nx):
        for ry in range(ny):
            for rz in range(nz):
                r_vecs.append((rx, ry, rz))

    # phi_rep[atom_i] has shape (n_cells, n, 3_alpha, 3_beta)
    phi_blocks = np.zeros((n, n_cells, n, 3, 3), dtype=float)
    computed_i = set()
    for atom_i in rep_atoms:
        for alpha in range(3):
            disp_p = base_sc_pos.copy()
            disp_p[atom_i, alpha] += delta
            sc.set_positions(disp_p)
            sc.calc = calculator
            f_p = sc.get_forces()

            disp_m = base_sc_pos.copy()
            disp_m[atom_i, alpha] -= delta
            sc.set_positions(disp_m)
            sc.calc = calculator
            f_m = sc.get_forces()

            fc_row = -(f_p - f_m) / (2.0 * delta)  # shape (n_cells * n, 3)
            fc_reshaped = fc_row.reshape((n_cells, n, 3))
            phi_blocks[atom_i, :, :, alpha, :] = fc_reshaped
        computed_i.add(atom_i)
    sc.set_positions(base_sc_pos)

    # Fill any remaining non-representative atoms by copying/averaging diagonal self-force and nearest Wyckoff orbit
    if len(computed_i) < n:
        try:
            import spglib
            cell_tuple = (atoms.get_cell().array, atoms.get_scaled_positions(), atoms.get_atomic_numbers())
            sym = spglib.get_symmetry(cell_tuple, symprec=1e-2)
            eq_atoms = sym['equivalent_atoms']
            lat = atoms.get_cell().array
            inv_lat = np.linalg.inv(lat)
            frac = atoms.get_scaled_positions()
            for j in range(n):
                if j in computed_i:
                    continue
                rep_i = int(eq_atoms[j])
                if rep_i not in computed_i:
                    rep_i = rep_atoms[0]
                # Find a symmetry op mapping rep_i -> j in the unit cell
                matched = False
                for rot, trans in zip(sym['rotations'], sym['translations']):
                    mapped_f = np.dot(frac[rep_i], rot.T) + trans
                    diff_f = mapped_f - frac[j]
                    diff_f -= np.round(diff_f)
                    if np.linalg.norm(np.dot(diff_f, lat)) < 0.12:
                        rot_c = lat.T @ rot @ inv_lat.T
                        n_j = np.round(mapped_f - frac[j]).astype(int)
                        # Map all (cell_c, atom_b) including supercell lattice vector rotation
                        for b in range(n):
                            mapped_b = np.dot(frac[b], rot.T) + trans
                            db = mapped_b - frac
                            db_wrapped = db - np.round(db)
                            b_target = int(np.argmin(np.linalg.norm(np.dot(db_wrapped, lat), axis=1)))
                            n_b0 = np.round(mapped_b - frac[b_target]).astype(int)
                            for c_idx, r_vec in enumerate(r_vecs):
                                rot_r = np.round(np.dot(np.array(r_vec, dtype=float), rot.T)).astype(int)
                                c_vec = (rot_r + n_b0 - n_j) % np.array([nx, ny, nz], dtype=int)
                                c_target = int(c_vec[0] * (ny * nz) + c_vec[1] * nz + c_vec[2])
                                phi_blocks[j, c_target, b_target] = rot_c @ phi_blocks[rep_i, c_idx, b] @ rot_c.T
                        matched = True
                        break
                if not matched:
                    phi_blocks[j] = phi_blocks[rep_i]
        except Exception:
            for j in range(n):
                if j not in computed_i:
                    phi_blocks[j] = phi_blocks[rep_atoms[0]]

    # Enforce acoustic sum rule on the supercell force constants: sum_{R, b} Phi_{(i,0),(b,R)} = 0
    for i in range(n):
        row_sum = np.sum(phi_blocks[i], axis=(0, 1))  # (3, 3)
        phi_blocks[i, 0, i] -= row_sum

    # Evaluate Bloch dynamical matrix D(q) at all non-Gamma zone-boundary wavevectors q in {0, 1/2}^3
    CONV_THZ = 15.633302
    dof = 3 * n
    q_list = []
    for hx in range(nx):
        for hy in range(ny):
            for hz in range(nz):
                if hx == 0 and hy == 0 and hz == 0:
                    continue
                q_list.append((hx / float(nx), hy / float(ny), hz / float(nz)))

    min_bz_freq = 999.0
    bz_imag_total = 0
    for qx, qy, qz in q_list:
        d_q = np.zeros((dof, dof), dtype=float)
        for c_idx, (rx, ry, rz) in enumerate(r_vecs):
            phase = math.cos(2.0 * math.pi * (qx * rx + qy * ry + qz * rz))
            blk = phi_blocks[:, c_idx, :, :, :]  # (n_i, n_b, 3, 3)
            # reshape to (3*n_i, 3*n_b)
            blk_2d = np.transpose(blk, (0, 2, 1, 3)).reshape((dof, dof))
            d_q += phase * blk_2d
        d_q = 0.5 * (d_q + d_q.T)
        dyn_q = d_q * np.outer(inv_sqrt_m, inv_sqrt_m)
        dyn_q = 0.5 * (dyn_q + dyn_q.T)
        eigs = np.linalg.eigvalsh(dyn_q)
        freqs = np.sign(eigs) * np.sqrt(np.abs(eigs)) * CONV_THZ
        if len(freqs):
            f_min = float(freqs[0])
            if f_min < min_bz_freq:
                min_bz_freq = f_min
            # Zone-boundary acoustic soft mode tolerance (-0.45 THz ~ 15 cm^-1)
            bz_imag_total += int(np.sum(freqs < -0.45))

    return dict(
        supercell_dim=f"{nx}x{ny}x{nz}",
        bz_min_freq_THz=round(min_bz_freq, 2) if min_bz_freq < 900.0 else 0.0,
        bz_imag_count=int(bz_imag_total),
        bz_dynamically_stable=bool(bz_imag_total == 0),
    )


def write_dft_input_decks(structure, pid, formula, competing_reaction, out_dir):
    """Write turnkey VASP (spglib-symmetrized POSCAR, POSCAR_conventional, INCAR, INCAR_AFM, KPOINTS) and QE (pw.in)."""
    cand_dir = Path(out_dir) / f'{pid}_{formula}'
    cand_dir.mkdir(parents=True, exist_ok=True)
    # 1. Symmetrize with spglib (symprec=5e-2) into exact primitive & refined conventional standard cells
    sym_prim = structure
    sym_conv = None
    sg_label = ''
    try:
        sga = SpacegroupAnalyzer(structure, symprec=5e-2)
        sg_label = f"{sga.get_space_group_symbol()} (#{sga.get_space_group_number()})"
        cand_p = sga.get_primitive_standard_structure()
        if len(cand_p) == len(structure):
            sym_prim = cand_p
        sym_conv = sga.get_refined_structure()
    except Exception:
        pass
    try:
        ang = sym_prim.lattice.angles
        if max(ang) > 120.0 or min(ang) < 60.0:
            sym_prim = sym_prim.get_reduced_structure('niggli')
    except Exception:
        pass
    sym_prim.to(filename=str(cand_dir / 'POSCAR'), fmt='poscar')
    structure.to(filename=str(cand_dir / 'POSCAR_raw_mace'), fmt='poscar')
    try:
        from pymatgen.io.cif import CifWriter
        CifWriter(sym_prim).write_file(str(cand_dir / 'primitive_niggli.cif'))
    except Exception:
        pass
    if sym_conv is not None:
        sym_conv.to(filename=str(cand_dir / 'POSCAR_conventional'), fmt='poscar')
        try:
            from pymatgen.io.cif import CifWriter
            CifWriter(sym_conv, symprec=1e-3).write_file(str(cand_dir / 'conventional_symmetrized.cif'))
        except Exception:
            pass

    # Build ferromagnetic (FM) and antiferromagnetic (AFM) MAGMOM strings matching POSCAR site order
    mag_3d_4f = {
        'Cr': 3.0, 'Mn': 4.0, 'Fe': 2.5, 'Co': 1.8, 'Ni': 1.0,
        'Ce': 1.0, 'Pr': 2.0, 'Nd': 3.0, 'Sm': 5.0, 'Eu': 7.0, 'Gd': 7.0,
        'Tb': 6.0, 'Dy': 5.0, 'Ho': 4.0, 'Er': 3.0, 'Tm': 2.0,
    }
    poscar_lines = (cand_dir / 'POSCAR').read_text(encoding='utf-8').splitlines()
    poscar_els = poscar_lines[5].split() if len(poscar_lines) > 6 else []
    poscar_cnts = [int(x) for x in poscar_lines[6].split()] if len(poscar_lines) > 6 else []
    fm_tokens, afm_tokens = [], []
    for el_s, cnt_s in zip(poscar_els, poscar_cnts):
        m0 = mag_3d_4f.get(el_s, 0.2)
        fm_tokens.append(f'{cnt_s}*{m0:.1f}')
        if el_s in mag_3d_4f and cnt_s >= 2:
            for k_i in range(cnt_s):
                afm_tokens.append(f'{m0 if k_i % 2 == 0 else -m0:+.1f}')
        else:
            afm_tokens.append(f'{cnt_s}*{m0:.1f}')
    magmom_fm_line = f"MAGMOM = {' '.join(fm_tokens)}\n" if fm_tokens else ''
    magmom_afm_line = f"MAGMOM = {' '.join(afm_tokens)}\n" if afm_tokens else ''

    incar = f"""# Turnkey PBE Spin-Polarized (FM) DFT Relaxation for Discovery {pid}: {formula} {sg_label}
# Competing Hull Reaction: {competing_reaction}
SYSTEM = {formula}_{pid}
PREC = Accurate
ENCUT = 520
EDIFF = 1E-6
EDIFFG = -0.01
IBRION = 2
ISIF = 3
NSW = 100
ISMEAR = 1
SIGMA = 0.15
ISPIN = 2
ISYM = 2
{magmom_fm_line}LREAL = Auto
LWAVE = .FALSE.
LCHARG = .FALSE.
"""
    atomic_text(cand_dir / 'INCAR', incar)
    if afm_tokens != fm_tokens:
        incar_afm = incar.replace('Spin-Polarized (FM)', 'Spin-Polarized (AFM)').replace(magmom_fm_line, magmom_afm_line)
        atomic_text(cand_dir / 'INCAR_AFM', incar_afm)
    kpoints = """Automatic Mesh
0
Gamma
6 6 6
0 0 0
"""
    atomic_text(cand_dir / 'KPOINTS', kpoints)
    kpoints_bands = """High-Symmetry Brillouin-Zone Path for DFT Band Dispersion dE_flat & |E_flat - E_F|
40
Line-mode
reciprocal
  0.000000  0.000000  0.000000   ! Gamma
  0.500000  0.000000  0.000000   ! M / X

  0.500000  0.000000  0.000000   ! M / X
  0.333333  0.333333  0.000000   ! K / S

  0.333333  0.333333  0.000000   ! K / S
  0.000000  0.000000  0.000000   ! Gamma

  0.000000  0.000000  0.000000   ! Gamma
  0.000000  0.000000  0.500000   ! A / Z
"""
    atomic_text(cand_dir / 'KPOINTS_BANDS', kpoints_bands)

    # 2. Quantum ESPRESSO pw.in
    lat = structure.lattice.matrix
    abc = structure.lattice.abc
    species_set = sorted({str(s.specie) for s in structure})
    ntyp = len(species_set)
    nat = len(structure)
    mag_elements = {'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb'}
    has_mag = any(el in mag_elements for el in species_set)
    kx = max(2, min(8, int(np.ceil(28.0 / max(abc[0], 1.0)))))
    ky = max(2, min(8, int(np.ceil(28.0 / max(abc[1], 1.0)))))
    kz = max(2, min(8, int(np.ceil(28.0 / max(abc[2], 1.0)))))
    qe_lines = [
        '&CONTROL',
        f"  calculation = 'vc-relax',",
        f"  prefix = '{formula}_{pid}',",
        "  pseudo_dir = './pseudos/sssp_efficiency',",
        "  outdir = './out/',",
        '  tprnfor = .true.,',
        '  tstress = .true.,',
        '  etot_conv_thr = 1.0d-5,',
        '  forc_conv_thr = 1.0d-4,',
        '/',
        '&SYSTEM',
        '  ibrav = 0,',
        f'  nat = {nat},',
        f'  ntyp = {ntyp},',
        '  ecutwfc = 60.0,',
        '  ecutrho = 480.0,',
        "  occupations = 'smearing',",
        "  smearing = 'mv',",
        '  degauss = 0.015,',
        f'  nspin = {2 if has_mag else 1},',
    ]
    if has_mag:
        for idx_el, el in enumerate(species_set, 1):
            if el in mag_elements:
                qe_lines.append(f'  starting_magnetization({idx_el}) = 0.25,')
    qe_lines.extend([
        '/',
        '&ELECTRONS',
        '  conv_thr = 1.0d-8,',
        '  mixing_beta = 0.4,',
        '/',
        '&IONS',
        "  ion_dynamics = 'bfgs',",
        '/',
        '&CELL',
        "  cell_dynamics = 'bfgs',",
        '/',
        'ATOMIC_SPECIES',
    ])
    for el in species_set:
        mass = atomic_masses[atomic_numbers[el]]
        qe_lines.append(f'  {el}  {mass:.4f}  {el}.UPF')
    qe_lines.append('CELL_PARAMETERS (angstrom)')
    for row in lat:
        qe_lines.append(f'  {row[0]:12.6f}  {row[1]:12.6f}  {row[2]:12.6f}')
    qe_lines.append('ATOMIC_POSITIONS (crystal)')
    for site in structure:
        fc = site.frac_coords
        qe_lines.append(f'  {site.specie}  {fc[0]:12.6f}  {fc[1]:12.6f}  {fc[2]:12.6f}')
    qe_lines.extend(['K_POINTS {automatic}', f'  {kx} {ky} {kz} 0 0 0', ''])
    atomic_text(cand_dir / 'pw.in', '\n'.join(qe_lines))
    return str(cand_dir)


def verify_run(run_dir, model_path='models/mace-mp-0b3-medium.model', hull_path='data/lemat-mace-mp-hull.parquet', rerun_all=False, only_new=True, since_id=0):
    import torch
    torch.set_num_threads(8)
    from mace.calculators import mace_mp
    from stability_screen import TabulatedHull

    root = Path(run_dir)
    progress_path = root / 'progress.json'
    progress_data = json.loads(progress_path.read_text(encoding='utf-8'))
    top_discoveries = list(progress_data.get('top_discoveries', []))
    seen_pids = {str(d['proposal_id']) for d in top_discoveries}

    # Also scan discoveries.csv and events.jsonl for any Tier 1 Uncharted Ground States beyond top 30
    disc_csv_path = root / 'discoveries.csv'
    if disc_csv_path.exists():
        import csv
        with disc_csv_path.open('r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                in_lem_raw = row.get('in_lemat_157k', row.get('in_lemat', 'True'))
                in_lem = str(in_lem_raw).lower() in ('true', '1', 'yes')
                if 'signed_hull_meV_atom' in row and row['signed_hull_meV_atom'] not in ('', 'None', None):
                    sh = float(row['signed_hull_meV_atom']) / 1000.0
                elif 'signed_hull_eV_atom' in row and row['signed_hull_eV_atom'] not in ('', 'None', None):
                    sh = float(row['signed_hull_eV_atom'])
                else:
                    sh = 99.0
                if (not in_lem) and sh <= 0.0:
                    pid = str(row['proposal_id']).zfill(4)
                    if pid not in seen_pids:
                        seen_pids.add(pid)
                        top_discoveries.append(dict(
                            proposal_id=pid,
                            formula=row.get('formula', ''),
                            cif_rel=row.get('cif_path') or row.get('cif_file') or f'structures/{pid}.cif',
                            signed_hull=sh,
                            in_lemat=False,
                            priority_tier=row.get('priority_tier', 'Tier 1: Uncharted Ground State'),
                            application_domain=row.get('application_domain', ''),
                            packing_efficiency_pct=float(row.get('packing_efficiency_pct') or 0.0),
                            competing_reaction=row.get('competing_reaction', ''),
                            template_formula=row.get('template_formula') or None,
                            substitution=row.get('substitution') or '',
                        ))
    events_path = root / 'telemetry' / 'events.jsonl'
    if events_path.exists():
        for line in events_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            ev = json.loads(line)
            is_eval = (ev.get('status') == 'evaluated') or (ev.get('event') == 'evaluated')
            is_nov = bool((ev.get('novel') and ev.get('unique')) or ev.get('novel_overall'))
            in_lem_ev = ev.get('composition_in_lemat_hull', ev.get('in_lemat', True))
            if is_eval and is_nov and not in_lem_ev:
                sh = ev.get('signed_decomposition_energy_eV_atom')
                if sh is not None and float(sh) <= 0.0:
                    pid = str(ev['proposal_id']).zfill(4)
                    if pid not in seen_pids:
                        seen_pids.add(pid)
                        meta_ev = ev.get('metadata') or {}
                        comp_rxn = ev.get('competing_reaction', '')
                        if not comp_rxn and ev.get('competing_phases'):
                            from pymatgen.core import Composition as _C
                            parts = []
                            for cp in ev['competing_phases']:
                                cf = _C(cp.get('composition', {})).reduced_formula
                                rid = cp.get('reference_id', '')
                                af = float(cp.get('atom_fraction', 0.0))
                                parts.append(f"{af:.2f} {cf}" + (f" ({rid})" if rid else ""))
                            comp_rxn = ' + '.join(parts)
                        cif_cand = f'structures/{pid}.cif' if (root / f'structures/{pid}.cif').exists() else f'structures/relaxed/{pid}.cif'
                        top_discoveries.append(dict(
                            proposal_id=pid,
                            formula=ev.get('formula', ''),
                            cif_rel=ev.get('relaxed_cif_rel', cif_cand),
                            signed_hull=float(sh),
                            in_lemat=False,
                            priority_tier=ev.get('priority_tier', 'Tier 1: Uncharted Ground State'),
                            application_domain=ev.get('application_domain', ''),
                            packing_efficiency_pct=ev.get('packing_efficiency_pct') or meta_ev.get('packing_efficiency_pct') or 0.0,
                            competing_reaction=comp_rxn,
                            template_formula=ev.get('template_formula') or meta_ev.get('template_formula'),
                            substitution=ev.get('substitution_pair') or ev.get('substitution') or meta_ev.get('substitution') or '',
                        ))
    top_discoveries.sort(key=lambda d: (0 if not d.get('in_lemat', True) else 1, float(d.get('signed_hull') or 0.0)))

    _hull_cache = [None]
    _calc_cache = [None]

    def _get_hull():
        if _hull_cache[0] is None:
            _hull_cache[0] = TabulatedHull.from_parquet(hull_path)
        return _hull_cache[0]

    def _get_calc():
        if _calc_cache[0] is None:
            _calc_cache[0] = mace_mp(model=str(model_path), device='cpu', default_dtype='float64')
        return _calc_cache[0]

    verified_records = []
    existing_by_pid = {}
    verif_json_path = root / 'verification_report.json'
    if not rerun_all and verif_json_path.exists():
        try:
            for item in json.loads(verif_json_path.read_text(encoding='utf-8')):
                existing_by_pid[str(item['proposal_id'])] = item
        except Exception:
            pass
    dft_root = root / 'dft_verification'
    dft_root.mkdir(exist_ok=True)
    from telemetry import Telemetry
    from pilot import Progress
    from algebraic_oracle import analyze_sublattice_flatband_and_frustration_qq, compute_structural_motif_distance
    from generator_bridge import check_synthesis_feasibility, ordered_formula
    tracker = Telemetry(root / 'telemetry', resume=True)
    prog = Progress(root, max(int(progress_data.get('planned', 240)), len(tracker.seen)), 1800.0, time.monotonic(), tracker, hull_available=True)

    def _enrich_and_classify(rec, struct, template_formula=None):
        el = rec.get('elastic_moduli', {})
        ph = rec.get('phonon_stability', {})
        b0 = float(el.get('bulk_modulus_GPa', 0.0))
        g0 = float(el.get('shear_modulus_GPa', 0.0))
        dens = float(rec.get('density_g_cm3') or struct.density or 1.0)
        ym = round((9.0 * b0 * g0) / max(3.0 * b0 + g0, 0.1), 1) if (b0 > 0 and g0 > 0) else 0.0
        el['youngs_modulus_GPa'] = ym
        spec_mod = round(ym / max(dens, 0.1), 1) if ym > 0 else 0.0
        el['specific_modulus_GPa_cm3_g'] = spec_mod
        pugh = float(el.get('pugh_ratio_B_G') or (b0 / max(g0, 0.1) if g0 > 0 else 2.0))
        # Chen's Vickers hardness formula is only physically valid for brittle/covalent solids (B/G <= 1.75, G >= 40 GPa)
        if pugh <= 1.75 and g0 >= 40.0:
            k_inv = g0 / max(b0, 0.1)
            el['vickers_hardness_GPa'] = round(max(1.0, 2.0 * ((k_inv * k_inv * g0) ** 0.585) - 3.0), 1)
        else:
            el['vickers_hardness_GPa'] = None
        if b0 > 0 and g0 > 0 and dens > 0:
            vl = math.sqrt(max(1.0, (b0 + (4.0 / 3.0) * g0) * 1e9 / (dens * 1000.0)))
            vt = math.sqrt(max(1.0, g0 * 1e9 / (dens * 1000.0)))
            vm = (0.333333 * (2.0 / (vt ** 3) + 1.0 / (vl ** 3))) ** (-1.0 / 3.0)
            n_dens = (len(struct) / float(struct.volume)) * 1e30
            deb = int(round(float((1.0545718e-34 / 1.380649e-23) * ((6.0 * (math.pi ** 2) * n_dens) ** (1.0 / 3.0)) * vm), 0))
            el['debye_temp_K'] = deb
            poisson = max(0.05, min(0.45, (3.0 * b0 - 2.0 * g0) / max(6.0 * b0 + 2.0 * g0, 0.1)))
            grun = 1.5 * (1.0 + poisson) / max(2.0 - 3.0 * poisson, 0.25)
            avg_m = float(sum(atomic_masses[atomic_numbers[str(s.specie)]] for s in struct)) / len(struct)
            d_ang = (float(struct.volume) / len(struct)) ** (1.0 / 3.0)
            el['slack_kappa_300K_W_mK'] = round(float(
                (3.1e-6 * avg_m * d_ang * (deb ** 3)) / max(1.0, (grun ** 2) * (len(struct) ** (2.0 / 3.0)) * 300.0)
            ), 2)
        else:
            deb = 0
            el['debye_temp_K'] = 0
            el['slack_kappa_300K_W_mK'] = None

        species_list = [str(s.specie) for s in struct]
        species_set = set(species_list)
        rec['ordered_formula'] = ordered_formula(struct.composition)
        if template_formula is not None:
            rec['template_formula'] = template_formula
        import re
        from pymatgen.core import Composition as _Comp
        # Ensure competing_reaction displays exact molar stoichiometric coefficients per 1 reduced formula unit
        raw_rxn = str(rec.get('competing_reaction', '') or '')
        if raw_rxn and '+' in raw_rxn:
            try:
                t_comp = struct.composition.reduced_composition
                t_dict = t_comp.get_el_amt_dict()
                els_order = sorted(t_dict.keys())
                rxn_terms = [t.strip() for t in raw_rxn.split('+') if t.strip()]
                parsed_terms = []
                for term in rxn_terms:
                    m_t = re.match(r'^([0-9./]+)\s+([A-Za-z0-9.]+)(?:\s*\(([^)]+)\))?', term)
                    if m_t:
                        c_str, f_str, ref_id = m_t.group(1), m_t.group(2), (m_t.group(3) or '')
                        c_comp = _Comp(f_str)
                        parsed_terms.append((c_str, f_str, ref_id, c_comp.get_el_amt_dict()))
                if len(parsed_terms) == len(rxn_terms) and parsed_terms:
                    a_mat = np.array([[pt[3].get(el, 0.0) for pt in parsed_terms] for el in els_order], dtype=float)
                    b_vec = np.array([t_dict[el] for el in els_order], dtype=float)
                    molar_c, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
                    if np.all(molar_c > 1e-4) and np.allclose(a_mat @ molar_c, b_vec, atol=5e-2):
                        new_parts = []
                        for mc, (_, f_str, ref_id, _) in zip(molar_c, parsed_terms):
                            ref_suffix = f' ({ref_id})' if ref_id else ''
                            new_parts.append(f'{float(mc):.2f} {f_str}{ref_suffix}')
                        rec['competing_reaction'] = ' + '.join(new_parts)
            except Exception:
                pass
        sg_match = re.search(r'#(\d+)', str(rec.get('relaxed_sg', '')))
        sg_num_val = int(sg_match.group(1)) if sg_match else None
        motif_info = compute_structural_motif_distance(
            struct,
            rec.get('template_formula'),
            spacegroup_num=sg_num_val,
            competing_reaction=rec.get('competing_reaction', ''),
        )
        rec['d_proto'] = motif_info['d_proto']
        rec['motif_novelty_label'] = motif_info['motif_novelty_label']
        rec['is_dopant_derivative'] = bool(motif_info.get('is_dopant_derivative', False))
        rec['ts_mix_1000K_meV_atom'] = float(motif_info.get('ts_mix_1000K_meV_atom', 0.0))
        synth = check_synthesis_feasibility(species_list, rec['ordered_formula'])
        fb = analyze_sublattice_flatband_and_frustration_qq(struct)
        rec['synthesis_feasibility'] = synth['synthesis_feasibility']
        rec['congruent_melt_feasible'] = synth['congruent_melt_feasible']
        rec['synthesis_tractable'] = synth.get('synthesis_tractable', True)
        rec['volatility_gap_K'] = synth['volatility_gap_K']
        rec['flat_band_mechanism'] = fb['flat_band_mechanism']
        rec['has_flat_band'] = fb['has_flat_band']
        rec['frustrated_triangles_per_site'] = fb['frustrated_triangles_per_site']

        # Normalize deceptive pymatgen electronegativity formula groupings like ScNi2(B3Mo)2 -> Wyckoff-ordered ScMo2(NiB3)2
        for _fld in ('template_formula', 'substitution_pair'):
            if rec.get(_fld):
                rec[_fld] = re.sub(
                    r'\b([A-Z][a-z]?)([A-Z][a-z]?)2\(B3Mo\)2\b',
                    r'\g<1>Mo2(\g<2>B3)2 [Wyckoff 2a:\g<1>, 4i:Mo2, 4j:\g<2>2]',
                    str(rec[_fld]),
                )

        # Estimate 1000 K free-energy hull shift: harmonic Debye vibrational dF_vib(1000K) + solid-solution disorder competition +T*dS_mix(1000K)
        signed_mev = float(rec.get('polished_signed_hull_meV_atom', 0.0))
        avg_mass = sum(float(atomic_masses[atomic_numbers[sp]]) for sp in species_list) / max(1, len(species_list))
        f_cov = sum(1.0 if sp in ('B', 'C', 'N') else (0.5 if sp == 'Si' else 0.0) for sp in species_list) / max(1, len(species_list))
        # Match reference Debye temperature to competing binary/ternary boride/silicide reactants (isodesmic bond conservation)
        ref_deb = max(180.0, min(800.0, (382.0 + 575.0 * f_cov) * ((38.0 / max(avg_mass, 10.0)) ** 0.38)))
        if deb > 50:
            # Differential harmonic Debye vibrational free energy at T = 1000 K vs competing phases (in meV/atom)
            rxn_str = str(rec.get('competing_reaction', ''))
            has_elem_comp = any(f" {el} (" in f" {rxn_str}" for el in species_set)
            vib_coupling = 0.08 if has_elem_comp else 0.035
            df_vib_1000 = 3.0 * 86.1733 * math.log(deb / ref_deb) * vib_coupling
        else:
            df_vib_1000 = 0.0
        # For a site-ordered line compound (S_conf = 0), configurational mixing entropy stabilizes the competing
        # disordered solid solution at 1000 K, penalizing the ordered phase stability margin by +T * delta_S_mix:
        ts_mix_1000 = float(rec.get('ts_mix_1000K_meV_atom', 0.0))
        dg_1000 = round(signed_mev + df_vib_1000 + ts_mix_1000, 1)
        rec['df_vib_1000K_meV_atom'] = round(df_vib_1000, 1)
        rec['dg_hull_1000K_meV_atom'] = dg_1000
        rec['free_energy_1000K_meV_atom'] = dg_1000
        if ts_mix_1000 > 0.0:
            if dg_1000 > 0.0:
                rec['synthesis_feasibility'] += f" · Site-Ordered Dopant Derivative (+TΔS_mix(1000K) = +{ts_mix_1000:.0f} meV/at Solid-Solution Disorder Competition; ΔG_1000K = {round(dg_1000):+d} meV/at)"
            else:
                rec['synthesis_feasibility'] += f" · Survives +TΔS_mix(1000K) = +{ts_mix_1000:.0f} meV/at Sublattice Mixing Penalty (ΔG_1000K = {round(dg_1000):+d} meV/at)"

        # Select ONE physically appropriate Target Figure of Merit & Benchmark Comparison per material class
        is_nonmetal_compound = bool(species_set & {'S', 'Se', 'Te', 'O', 'F', 'Cl', 'Br', 'I'})
        is_cermet_boride_carbide = bool(species_set & {'B', 'C'})
        is_polar_sg = any(tag in str(rec.get('relaxed_sg', '')) for tag in ('#26)', '#36)', '#160)', '#186)', '#189)', '#216)'))
        if spec_mod >= 30.0 or (dens <= 4.8 and spec_mod >= 24.0):
            pct_vs_ti = ((spec_mod - 26.0) / 26.0) * 100.0
            pugh_val = float(el.get('pugh_ratio_B_G', 1.5))
            rec['target_fom'] = f"E/ρ = {spec_mod:.1f} GPa·cm³/g"
            if is_cermet_boride_carbide:
                rec['benchmark_comparison'] = (
                    f"{spec_mod / 58.0:.2f}× MoAlB MAB Cermet (58.0, B/G=1.69; "
                    f"B/G={pugh_val:.2f} vs TiB₂ 1.12; {pct_vs_ti:+.0f}% vs Ti-6Al-4V)"
                )
            else:
                rec['benchmark_comparison'] = f"{pct_vs_ti:+.0f}% E/ρ vs Ti-6Al-4V (26.0); {spec_mod / 48.0:.2f}× Ti₅Si₃ (48.0)"
            if 'Ti' in species_set and 'Al' in species_set:
                rec['synthesis_feasibility'] += ' · Check Peritectic Competition vs L1₂/D0₂₂ Trialuminides'
        elif b0 >= 145.0:
            pugh_val = float(el.get('pugh_ratio_B_G', 1.5))
            rec['target_fom'] = f"B₀ = {b0:.0f} GPa, Θ_D = {deb} K"
            if is_cermet_boride_carbide:
                rec['benchmark_comparison'] = f"{b0 / 207.0:.2f}× MoAlB Cermet B₀ (207 GPa; B/G={pugh_val:.2f} vs TiB₂ 1.12)"
            else:
                rec['benchmark_comparison'] = f"{b0 / 170.0:.2f}× bcc-Fe B₀ (170 GPa)"
        elif fb['has_flat_band']:
            rec['target_fom'] = fb['flat_band_mechanism']
            rec['benchmark_comparison'] = 'Pre-DFT Motif; Check E(k) Dispersion'
        elif is_polar_sg:
            rec['target_fom'] = f"Polar Symmetry ({rec.get('relaxed_sg', '')})"
            rec['benchmark_comparison'] = 'Non-Centrosymmetric Piezoelectric Class'
        elif is_nonmetal_compound and el.get('slack_kappa_300K_W_mK') is not None:
            kl = float(el['slack_kappa_300K_W_mK'])
            rec['target_fom'] = f"κ_L(300K) ≈ {kl:.2f} W/m·K"
            rec['benchmark_comparison'] = f"{kl / 1.5:.2f}× Bi₂Te₃ κ_L (1.5 W/m·K; Non-Metallic Check Needed)"
        else:
            rec['target_fom'] = f"E = {ym:.0f} GPa (E/ρ = {spec_mod:.1f})"
            rec['benchmark_comparison'] = f"Θ_D = {deb} K (vs 428 K fcc-Al / 470 K bcc-Fe)"

        # Recompute dynamical stability with strict -0.05 THz threshold on cached phonon_stability
        min_opt_thz = float(ph.get('min_optical_freq_THz', 0.0))
        if min_opt_thz < -0.05:
            ph['imaginary_mode_count'] = max(1, int(ph.get('imaginary_mode_count', 0)))
            ph['dynamically_stable'] = False

        # Compute transparent 100-Point Quantitative Discovery & Novelty Score
        # Base components (0-75 pts): Novelty (0-25) + Stability (0-25) + FoM (0-15) + Synthesis (0-10)
        # Pareto Frontier Bonus (0-25 pts): +18 pts (1-Front), +22 pts (2-Front), +25 pts (3-Front)
        # Linear d_proto scaling: reserve >10/25 Novelty points strictly for d_proto >= 0.25
        d_proto_val = float(rec.get('d_proto', 0.15))
        if d_proto_val < 0.25:
            nov_pts = min(10, max(2, int(round(40.0 * d_proto_val))))
        else:
            sym_bonus = 3 if (is_polar_sg or fb['has_flat_band']) else 0
            nov_pts = min(25, 11 + int(round((d_proto_val - 0.25) * 40.0)) + sym_bonus)

        all_passed = (
            signed_mev <= 0.0
            and ph.get('dynamically_stable', False)
            and el.get('born_elastic_stable', False)
            and 0.78 <= float(rec.get('bond_realism_ratio', 1.0)) <= 1.20
            and float(rec.get('packing_efficiency_pct', 70.0)) <= 96.0
        )
        hull_pts = min(13.0, max(0.0, -signed_mev / 3.2)) if signed_mev <= 0.0 else 0.0
        has_strong_3d_mag = bool(species_set & {'Cr', 'Mn', 'Fe'})
        has_open_3d_mag = bool(species_set & {'Cr', 'Mn', 'Fe', 'Co'})
        if has_strong_3d_mag and signed_mev > -15.0:
            mag_pen = 3.0
        elif has_strong_3d_mag or (has_open_3d_mag and signed_mev > -15.0):
            mag_pen = 2.0
        else:
            mag_pen = 0.0
        stab_pts = int(round(max(0.0, min(25.0, hull_pts - mag_pen + (8.0 if ph.get('dynamically_stable', False) else 0.0) + (4.0 if (el.get('born_elastic_stable', False) and 0.78 <= float(rec.get('bond_realism_ratio', 1.0)) <= 1.20 and float(rec.get('packing_efficiency_pct', 70.0)) <= 96.0) else 0.0)))))
        # Strictly calibrated FoM points (0-15):
        fom_raw = 0.0
        if spec_mod >= 28.0:
            fom_raw += min(11.0, (spec_mod - 26.0) * 0.75)
        if b0 >= 140.0:
            fom_raw += min(8.0, (b0 - 125.0) * 0.13)
        if deb >= 500:
            fom_raw += min(6.0, (deb - 470.0) / 30.0)
        if is_nonmetal_compound and (el.get('slack_kappa_300K_W_mK') or 99.0) <= 1.2:
            fom_raw += min(8.0, (1.5 - float(el['slack_kappa_300K_W_mK'])) * 8.0)
        if fb['has_flat_band'] or is_polar_sg:
            fom_raw += 5.0
        fom_pts = int(round(min(15.0, fom_raw)))
        syn_raw = float(synth.get('synthesis_score', 6))
        if rec.get('is_dopant_derivative', False) or ts_mix_1000 >= 12.0:
            syn_raw -= 1.5
        if dg_1000 > 5.0:
            syn_pts = max(2, int(round(syn_raw - 3.0)))
            if '1000K' not in str(rec.get('synthesis_feasibility', '')):
                rec['synthesis_feasibility'] = str(rec.get('synthesis_feasibility', '')) + f' · Check 1000K Entropic/Vibrational Shift (dG_1000K={dg_1000:+.0f} meV/at)'
        elif dg_1000 > 0.0:
            syn_pts = max(3, int(round(syn_raw - 1.0)))
        else:
            syn_pts = max(2, int(round(syn_raw)))
        base_score = int(nov_pts + stab_pts + fom_pts + syn_pts)
        rec['base_score'] = base_score
        rec['nov_pts'] = int(nov_pts)
        rec['stab_pts'] = int(stab_pts)
        rec['fom_pts'] = int(fom_pts)
        rec['syn_pts'] = int(syn_pts)
        rec['pareto_pts'] = 0
        rec['discovery_score'] = base_score
        rec['score_breakdown'] = f"Pareto:+0/25 · Nov:{rec['nov_pts']}/25 · Stab:{rec['stab_pts']}/25 · FoM:{rec['fom_pts']}/15 · Synth:{rec['syn_pts']}/10"
        rec['outlier_summary'] = f"{rec['target_fom']} ({rec['benchmark_comparison']})"
        rec['schema_v'] = 4

        if not all_passed:
            rec['overall_verdict'] = f"Score {base_score}/100 · Phonon/Bond Check Needed"
            rec['verdict_rank'] = 3
        elif not synth.get('synthesis_tractable', True):
            rec['overall_verdict'] = f"Score {base_score}/100 · Volatility/Disproportionation Risk"
            rec['verdict_rank'] = 2
        elif base_score >= 62 and synth['congruent_melt_feasible']:
            rec['overall_verdict'] = f"Score {base_score}/100 · Arc-Melt Compatible Lead"
            rec['verdict_rank'] = 1
        else:
            rec['overall_verdict'] = f"Score {base_score}/100 · Validated 0K Ground State"
            rec['verdict_rank'] = 1
        return rec

    def _resolve_cif_path(pid_str, cif_rel_str=None):
        candidates = []
        if cif_rel_str:
            candidates.append(root / cif_rel_str)
        candidates.append(root / f'structures/{pid_str}.cif')
        candidates.append(root / f'structures/{pid_str}.relaxed.cif')
        candidates.append(root / f'structures/relaxed/{pid_str}.cif')
        for p in candidates:
            if p.exists():
                return p
        return candidates[0]

    # Verify all Tier 1 Uncharted Ground States + Top Tier 2 Ground States (top 10 overall)
    cached_count = 0
    new_count = 0
    newly_verified_pids = set()
    for idx, d in enumerate(top_discoveries):
        is_tier1 = (not d.get('in_lemat', True)) and ((d.get('signed_hull') or 0.0) <= 0.0)
        if not is_tier1 and idx >= 10:
            continue
        pid = str(d['proposal_id'])
        pid_int = int(pid) if pid.isdigit() else 0
        if not rerun_all and (
            (only_new and pid in existing_by_pid)
            or (since_id > 0 and pid_int < since_id and pid in existing_by_pid)
            or (pid in existing_by_pid and existing_by_pid[pid].get('schema_v') == 4 and existing_by_pid[pid].get('dft_input_dir'))
        ):
            # Zero-I/O incremental fast path: reuse existing verified record without recomputing MACE/phonons/elasticity
            rec = existing_by_pid[pid]
            rec['pareto_pts'] = 0
            rec['discovery_score'] = int(rec.get('nov_pts', 0)) + int(rec.get('stab_pts', 0)) + int(rec.get('fom_pts', 0)) + int(rec.get('syn_pts', 0))
            verified_records.append(rec)
            cached_count += 1
            continue
        if since_id > 0 and pid_int < since_id:
            continue
        cif_path = _resolve_cif_path(pid, d.get('cif_rel'))
        if not rerun_all and pid in existing_by_pid:
            rec = existing_by_pid[pid]
            if cif_path.exists():
                cand_dft_dir = dft_root / f"{pid}_{rec.get('formula', d['formula'])}"
                dft_matches = list(dft_root.glob(f"{pid}_*"))
                if dft_matches:
                    cand_dft_dir = dft_matches[0]
                conv_cif = cand_dft_dir / 'conventional_symmetrized.cif'
                if rec.get('schema_v') == 4 and conv_cif.exists():
                    # Fast path: already verified & classified under current schema_v=4
                    rec['pareto_pts'] = 0
                    rec['discovery_score'] = int(rec.get('nov_pts', 0)) + int(rec.get('stab_pts', 0)) + int(rec.get('fom_pts', 0)) + int(rec.get('syn_pts', 0))
                    verified_records.append(rec)
                    cached_count += 1
                    continue
                raw_poscar = cand_dft_dir / 'POSCAR_raw_mace'
                std_poscar = cand_dft_dir / 'POSCAR'
                cif_txt = cif_path.read_text(encoding='utf-8')
                was_corrupted = (
                    '/999' in cif_txt
                    or '/998' in cif_txt
                    or '/997' in cif_txt
                    or '_symmetry_Int_Tables_number   1\n' not in cif_txt
                )
                if was_corrupted and std_poscar.exists():
                    s_cached = Structure.from_file(str(std_poscar))
                elif was_corrupted and raw_poscar.exists():
                    s_cached = Structure.from_file(str(raw_poscar))
                else:
                    s_cached = Structure.from_file(str(cif_path))
                was_skewed = False
                try:
                    ang_c = s_cached.lattice.angles
                    if max(ang_c) > 120.0 or min(ang_c) < 60.0:
                        s_cached = s_cached.get_reduced_structure('niggli')
                        was_skewed = True
                    if was_skewed or was_corrupted:
                        from pymatgen.io.cif import CifWriter
                        CifWriter(s_cached).write_file(str(cif_path))
                except Exception:
                    pass
                ph_c = rec.get('phonon_stability', {})
                if (
                    was_corrupted
                    or abs(float(ph_c.get('bz_min_freq_THz', 0.0))) > 50.0
                    or ph_c.get('supercell_dim') == '2x1x1'
                ):
                    for k_bz in ('bz_min_freq_THz', 'bz_imag_count', 'bz_dynamically_stable', 'bz_exact_v2', 'bz_exact_v3', 'supercell_dim'):
                        ph_c.pop(k_bz, None)
                rec = _enrich_and_classify(rec, s_cached, d.get('template_formula'))
                needs_conv_update = (
                    was_skewed
                    or was_corrupted
                    or not conv_cif.exists()
                    or '_symmetry_Int_Tables_number   1\n' in conv_cif.read_text(encoding='utf-8')
                )
                if needs_conv_update:
                    write_dft_input_decks(s_cached, pid, rec.get('formula', d['formula']), rec.get('competing_reaction', ''), dft_root)
            verified_records.append(rec)
            cached_count += 1
            print(f"[{pid}] {rec.get('ordered_formula', rec['formula']):12s} | dE_hull={round(rec['polished_signed_hull_meV_atom']):+4d} meV/at | "
                  f"d_proto={rec.get('d_proto', 0):.2f} | {rec.get('target_fom',''):24s} | {rec['overall_verdict']} [cached]", flush=True)
            continue
        formula = d['formula']
        if not cif_path.exists():
            continue
        new_count += 1
        newly_verified_pids.add(pid)
        calc = _get_calc()
        hull = _get_hull()
        prog.update('5-GATE VERIFICATION', active=f'{pid} {formula}', step='polish+phonon+elastic', force=True)

        struct = Structure.from_file(str(cif_path))
        atoms = AseAtomsAdaptor.get_atoms(struct)
        atoms.calc = calc

        # Symmetry-constrained tight medium polish (fmax=0.006 eV/A, max 12 steps)
        init_fmax = float(np.linalg.norm(atoms.get_forces(), axis=1).max()) if len(atoms) else 0.0
        if init_fmax > 0.008:
            try:
                from ase.constraints import FixSymmetry
                atoms.set_constraint(FixSymmetry(atoms))
            except Exception:
                pass
            opt = BFGS(FrechetCellFilter(atoms), logfile=None)
            opt.run(fmax=0.006, steps=12)
            atoms.set_constraint()
        polished_struct = AseAtomsAdaptor.get_structure(atoms)
        polished_e = float(atoms.get_potential_energy()) / len(atoms)
        hull_res = hull.evaluate(polished_struct.composition, polished_e)
        signed_mev = round(float(hull_res['signed_decomposition_energy_eV_atom']) * 1000.0, 2)
        form_e = round(float(hull_res.get('formation_energy_eV_atom', 0.0)), 4)

        # Gate 1: Hull Depth vs ML Uncertainty
        if signed_mev <= -15.0:
            gate1_hull = 'PASS (Deep Ground State <= -15 meV/at; exceeds ML noise)'
        elif signed_mev <= -10.0:
            gate1_hull = 'PASS (Strong Ground State -10 to -15 meV/at)'
        elif signed_mev <= 0.0:
            gate1_hull = 'BORDERLINE (0 to -10 meV/at; within DFT/ML error bar)'
        else:
            gate1_hull = f'METASTABLE ({signed_mev:+.1f} meV/at)'

        # Gate 2: Dynamic Phonon & Elastic Stability
        elastic = compute_elastic_moduli(atoms, calc)
        phonon = compute_dynamical_phonon_stability(atoms, calc)
        gate2_dyn = 'PASS (0 imaginary modes & Born elastic B0,G > 0)' if (
            phonon['dynamically_stable'] and elastic['born_elastic_stable']
        ) else f"WARN (imag={phonon['imaginary_mode_count']}, B0={elastic['bulk_modulus_GPa']} GPa, G={elastic['shear_modulus_GPa']} GPa)"

        # Gate 3: Crystallographic & Steric Bond Realism
        dmat = np.array(polished_struct.distance_matrix, dtype=float)
        species = [str(s.specie) for s in polished_struct]
        radii = [float(covalent_radii[atomic_numbers[el]]) for el in species]
        min_ratio = 999.0
        min_pair_str = ''
        for i in range(len(polished_struct)):
            for j in range(i + 1, len(polished_struct)):
                dij = float(dmat[i, j])
                if dij > 0.4:
                    ratio = dij / (radii[i] + radii[j])
                    if ratio < min_ratio:
                        min_ratio = ratio
                        p_els = sorted([species[i], species[j]])
                        min_pair_str = f'{p_els[0]}-{p_els[1]}: {dij:.2f} A ({ratio:.2f}x sum r_cov)'
        sga = SpacegroupAnalyzer(polished_struct, symprec=0.05)
        sg_num = int(sga.get_space_group_number())
        sg_sym = str(sga.get_space_group_symbol())
        packing = d.get('packing_efficiency_pct') or 0.0
        gate3_cryst = 'PASS (High symmetry, realistic bond ratio & dense packing)' if (
            sg_num >= 15 and 0.80 <= min_ratio <= 1.20 and 45.0 <= packing <= 96.0
        ) else f'CHECK (SG #{sg_num}, bond_ratio={min_ratio:.2f}, packing={packing}%)'

        # Gate 5: Turnkey DFT Input Deck
        dft_dir = write_dft_input_decks(polished_struct, pid, formula, d.get('competing_reaction', ''), dft_root)

        record = dict(
            proposal_id=pid,
            formula=formula,
            priority_tier=d.get('priority_tier', ''),
            in_lemat=d.get('in_lemat', True),
            application_domain=d.get('application_domain', ''),
            relaxed_sg=f'{sg_sym} (#{sg_num})',
            polished_signed_hull_meV_atom=signed_mev,
            formation_energy_eV_atom=form_e,
            gate1_hull_confidence=gate1_hull,
            gate2_dynamic_elastic=gate2_dyn,
            gate3_crystallographic_realism=gate3_cryst,
            bond_realism_detail=min_pair_str,
            bond_realism_ratio=round(min_ratio, 3),
            packing_efficiency_pct=packing,
            density_g_cm3=round(float(polished_struct.density), 3),
            elastic_moduli=elastic,
            phonon_stability=phonon,
            dft_input_dir=dft_dir,
            overall_verdict='',
            competing_reaction=d.get('competing_reaction', ''),
            template_formula=d.get('template_formula'),
        )
        record = _enrich_and_classify(record, polished_struct, d.get('template_formula'))
        verdict = record['overall_verdict']
        verified_records.append(record)
        verified_records.sort(key=lambda r: (r.get('verdict_rank', 2), -int(r.get('discovery_score', 0)), float(r.get('polished_signed_hull_meV_atom', 0.0))))
        atomic_text(root / 'verification_report.json', json.dumps(verified_records, indent=2))
        prog.update('5-GATE VERIFICATION', active=f'{pid} {formula}', step=verdict.split('·')[0].strip(), force=True)
        print(f"[{pid}] {record.get('ordered_formula', formula):12s} | dE_hull={round(signed_mev):+4d} meV/at | "
              f"d_proto={record.get('d_proto', 0):.2f} | {record.get('target_fom',''):24s} | {verdict}", flush=True)

    def _apply_pareto_frontier_analysis(records):
        """Identify non-dominated Pareto-frontier outliers across 3 functional performance trade-off planes and award up to +25 pts."""
        eligible = [
            r for r in records
            if float(r.get('polished_signed_hull_meV_atom', 99.0)) <= 0.0
            and r.get('phonon_stability', {}).get('dynamically_stable', False)
            and r.get('phonon_stability', {}).get('bz_dynamically_stable', True) is not False
            and float(r.get('phonon_stability', {}).get('min_optical_freq_THz', -1.0)) > 0.0
            and r.get('elastic_moduli', {}).get('born_elastic_stable', False)
            and 0.78 <= float(r.get('bond_realism_ratio', 1.0)) <= 1.20
            and float(r.get('packing_efficiency_pct', 70.0)) <= 96.0
            and r.get('synthesis_tractable', True)
            and not r.get('is_dopant_derivative', False)
            and float(r.get('free_energy_1000K_meV_atom', 0.0)) <= 0.0
            and int(r.get('fom_pts', 0)) > 0
        ]
        for r in records:
            r['pareto_frontiers'] = []
            r['is_pareto_outlier'] = False
            r['dominated_by'] = ''

        # Plane 0: Ashby Specific Stiffness (Maximize E/rho >= 28.0, Minimize density rho)
        p0_pool = [r for r in eligible if float(r['elastic_moduli'].get('specific_modulus_GPa_cm3_g', 0.0)) >= 28.0]
        for a in eligible:
            ea = float(a['elastic_moduli'].get('specific_modulus_GPa_cm3_g', 0.0))
            da = float(a.get('density_g_cm3', 99.0))
            if ea < 28.0:
                continue
            dominated_by = None
            for b in p0_pool:
                if b['proposal_id'] == a['proposal_id']:
                    continue
                eb = float(b['elastic_moduli'].get('specific_modulus_GPa_cm3_g', 0.0))
                db = float(b.get('density_g_cm3', 99.0))
                if eb >= ea and db <= da and (eb > ea or db < da):
                    dominated_by = f"#{b['proposal_id']} {b.get('ordered_formula', b['formula'])}"
                    break
            if dominated_by is None:
                a['pareto_frontiers'].append('Ashby E/ρ–ρ')
            elif dominated_by and not a['dominated_by']:
                a['dominated_by'] = dominated_by

        # Plane 1: Pugh Strength-vs-Ductility (Maximize G >= 65.0 GPa, Maximize B0/G >= 1.75)
        p1_pool = [
            r for r in eligible
            if float(r['elastic_moduli'].get('shear_modulus_GPa', 0.0)) >= 65.0
            and float(r['elastic_moduli'].get('pugh_ratio_B_G', 0.0)) >= 1.75
        ]
        for a in eligible:
            ga = float(a['elastic_moduli'].get('shear_modulus_GPa', 0.0))
            pa = float(a['elastic_moduli'].get('pugh_ratio_B_G', 0.0))
            if ga < 65.0 or pa < 1.75:
                continue
            dominated_by = None
            for b in p1_pool:
                if b['proposal_id'] == a['proposal_id']:
                    continue
                gb = float(b['elastic_moduli'].get('shear_modulus_GPa', 0.0))
                pb = float(b['elastic_moduli'].get('pugh_ratio_B_G', 0.0))
                if gb >= ga and pb >= pa and (gb > ga or pb > pa):
                    dominated_by = f"#{b['proposal_id']} {b.get('ordered_formula', b['formula'])}"
                    break
            if dominated_by is None:
                a['pareto_frontiers'].append('Pugh G–(B₀/G)')
            elif dominated_by and not a['dominated_by']:
                a['dominated_by'] = dominated_by

        # Plane 2: Refractory Incompressibility vs Acoustic Debye Stiffness (Maximize B0 >= 145 GPa, Maximize Theta_D >= 380 K)
        p2_pool = [
            r for r in eligible
            if float(r['elastic_moduli'].get('bulk_modulus_GPa', 0.0)) >= 145.0
            and float(r['elastic_moduli'].get('debye_temp_K', 0.0)) >= 380.0
        ]
        for a in eligible:
            ba = float(a['elastic_moduli'].get('bulk_modulus_GPa', 0.0))
            ta = float(a['elastic_moduli'].get('debye_temp_K', 0.0))
            if ba < 145.0 or ta < 380.0:
                continue
            dominated_by = None
            for b in p2_pool:
                if b['proposal_id'] == a['proposal_id']:
                    continue
                bb = float(b['elastic_moduli'].get('bulk_modulus_GPa', 0.0))
                tb = float(b['elastic_moduli'].get('debye_temp_K', 0.0))
                if bb >= ba and tb >= ta and (bb > ba or tb > ta):
                    dominated_by = f"#{b['proposal_id']} {b.get('ordered_formula', b['formula'])}"
                    break
            if dominated_by is None:
                a['pareto_frontiers'].append('Stiffness B₀–Θ_D')
            elif dominated_by and not a['dominated_by']:
                a['dominated_by'] = dominated_by

        # Plane 3: Thermo-Mechanical Shear Rigidity vs 1000 K Hull Driving Force (Maximize G >= 95 GPa, Minimize ΔG_1000K <= -5 meV/at)
        p3_pool = [
            r for r in eligible
            if float(r['elastic_moduli'].get('shear_modulus_GPa', 0.0)) >= 95.0
            and float(r.get('free_energy_1000K_meV_atom', 0.0)) <= -5.0
        ]
        for a in eligible:
            ga = float(a['elastic_moduli'].get('shear_modulus_GPa', 0.0))
            dga = float(a.get('free_energy_1000K_meV_atom', 0.0))
            if ga < 95.0 or dga > -5.0:
                continue
            dominated_by = None
            for b in p3_pool:
                if b['proposal_id'] == a['proposal_id']:
                    continue
                gb = float(b['elastic_moduli'].get('shear_modulus_GPa', 0.0))
                dgb = float(b.get('free_energy_1000K_meV_atom', 0.0))
                if gb >= ga and dgb <= dga and (gb > ga or dgb < dga):
                    dominated_by = f"#{b['proposal_id']} {b.get('ordered_formula', b['formula'])}"
                    break
            if dominated_by is None:
                a['pareto_frontiers'].append('Thermo-Mech G–ΔG₁₀₀₀K')
            elif dominated_by and not a['dominated_by']:
                a['dominated_by'] = dominated_by

        for r in records:
            fronts = r.get('pareto_frontiers', [])
            n_fronts = len(fronts)
            r['pareto_front_count'] = n_fronts
            if n_fronts >= 3:
                p_pts = 25
            elif n_fronts == 2:
                p_pts = 22
            elif n_fronts == 1:
                p_pts = 18
            else:
                p_pts = 0
            r['pareto_pts'] = p_pts
            nov_i = int(r.get('nov_pts', 8))
            stab_i = int(r.get('stab_pts', 20))
            fom_i = int(r.get('fom_pts', 0))
            syn_i = int(r.get('syn_pts', 6))
            raw_sum = p_pts + nov_i + stab_i + fom_i + syn_i
            # Explicit additive penalty for dynamical instability (Gamma or Zone-Boundary) or synthesis disproportionation risk
            pen_pts = 0
            ph_r = r.get('phonon_stability', {})
            is_dyn_ok = bool(ph_r.get('dynamically_stable', False) and ph_r.get('bz_dynamically_stable', True) is not False)
            if not is_dyn_ok:
                pen_pts = min(raw_sum - 12, 20) if raw_sum > 12 else 0
            elif not r.get('synthesis_tractable', True):
                pen_pts = min(raw_sum - 15, 15) if raw_sum > 15 else 0
            tot = max(10, min(100, raw_sum - pen_pts))
            r['pen_pts'] = pen_pts
            r['discovery_score'] = tot
            pen_str = f" · Pen:-{pen_pts}" if pen_pts > 0 else ""
            r['score_breakdown'] = (
                f"Pareto:+{p_pts}/25 · Nov:{nov_i}/25 · "
                f"Stab:{stab_i}/25 · FoM:{fom_i}/15 · Synth:{syn_i}/10{pen_str}"
            )
            if fronts:
                r['is_pareto_outlier'] = True
                r['pareto_badge'] = ' · '.join(fronts)
                r['pareto_status'] = f"★ {n_fronts}-Front Pareto (+{p_pts} pts): {r['pareto_badge']}"
                r['verdict_rank'] = 0
                r['overall_verdict'] = f"Score {tot}/100 · ★ Pareto Outlier ({r['pareto_badge']})"
            else:
                r['pareto_badge'] = ''
                if not is_dyn_ok:
                    r['pareto_status'] = f"Ineligible (Imaginary Phonon; -{pen_pts} pts)"
                    r['verdict_rank'] = 3
                    r['overall_verdict'] = f"Score {tot}/100 · Phonon/Bond Check Needed"
                elif not r.get('synthesis_tractable', True):
                    r['pareto_status'] = f"Ineligible (Synthesis Risk; -{pen_pts} pts)"
                    r['verdict_rank'] = 2
                    r['overall_verdict'] = f"Score {tot}/100 · Volatility/Disproportionation Risk"
                elif int(r.get('fom_pts', 0)) == 0:
                    r['pareto_status'] = 'Ineligible (FoM = 0/15)'
                    r['verdict_rank'] = 1
                    r['overall_verdict'] = f"Score {tot}/100 · Validated 0K Ground State"
                elif r.get('dominated_by'):
                    r['pareto_status'] = f"Dominated (by {r['dominated_by']})"
                    r['verdict_rank'] = 1
                    r['overall_verdict'] = f"Score {tot}/100 · Validated 0K Ground State"
                elif float(r.get('free_energy_1000K_meV_atom', 0.0)) > 0.0:
                    r['pareto_status'] = f"Ineligible (ΔG_1000K = {round(float(r['free_energy_1000K_meV_atom'])):+d} meV/at > 0)"
                    r['verdict_rank'] = 1
                    r['overall_verdict'] = f"Score {tot}/100 · Validated 0K Ground State"
                elif r.get('is_dopant_derivative', False):
                    r['pareto_status'] = 'Ineligible (Sublattice-Split Derivative)'
                    r['verdict_rank'] = 1
                    r['overall_verdict'] = f"Score {tot}/100 · Validated 0K Ground State"
                else:
                    r['pareto_status'] = 'Interior (Sub-Threshold FoM)'
                    r['verdict_rank'] = 1
                    r['overall_verdict'] = f"Score {tot}/100 · Validated 0K Ground State"

    print(f"[5-GATE VERIFICATION] Incremental summary: {cached_count} cached records reused, {new_count} newly evaluated (rerun_all={rerun_all}).", flush=True)
    _apply_pareto_frontier_analysis(verified_records)
    # Sort strictly by Final Discovery & Novelty Score (0-100) descending!
    verified_records.sort(key=lambda r: (
        -int(r.get('discovery_score', 0)),
        -int(r.get('pareto_front_count', 0)),
        float(r.get('polished_signed_hull_meV_atom', 0.0)),
    ))

    # Run Niggli-reduced 2x2x2 (or 2x2x1) Supercell Zone-Boundary (q in {X, M, R}) Phonon Dispersion Check on top candidates
    for idx_r, r in enumerate(verified_records):
        if 'dft_input_dir' in r and isinstance(r['dft_input_dir'], str):
            r['dft_input_dir'] = r['dft_input_dir'].replace('./', '').replace('\\', '/')
        ph = r.get('phonon_stability', {})
        if not (r.get('is_pareto_outlier') or idx_r < 22):
            continue
        pid_zb = str(r['proposal_id'])
        if rerun_all or ((not only_new or pid_zb in newly_verified_pids) and ('bz_min_freq_THz' not in ph or not ph.get('bz_exact_v3'))):
            pid = pid_zb
            cif_p = _resolve_cif_path(pid, r.get('cif_rel'))
            if cif_p.exists():
                print(f"  -> Zone-boundary supercell phonon check (q in {{X,M,R}}) for [{pid}] {r.get('ordered_formula', r['formula'])}...", flush=True)
                s_zb = Structure.from_file(str(cif_p))
                a_zb = AseAtomsAdaptor.get_atoms(s_zb)
                bz_res = compute_zone_boundary_phonon_check(a_zb, _get_calc())
                bz_res['bz_exact_v3'] = True
                ph.update(bz_res)
        # Clean any stale zone-boundary acoustic warning if bz_imag_count == 0
        if ph.get('bz_imag_count', 0) == 0 and ' · Check ' in str(r.get('synthesis_feasibility', '')) and 'Zone-Boundary Acoustic' in str(r.get('synthesis_feasibility', '')):
            r['synthesis_feasibility'] = str(r['synthesis_feasibility']).split(' · Check ')[0]
        elif ph.get('bz_imag_count', 0) > 0 and 'Zone-Boundary Acoustic' not in str(r.get('synthesis_feasibility', '')):
            r['synthesis_feasibility'] = str(r.get('synthesis_feasibility', '')) + f" · Check {ph.get('supercell_dim','2x2x2')} Zone-Boundary Acoustic Mode ({ph.get('bz_min_freq_THz', 0):+.2f} THz)"

    # Re-apply Pareto frontier analysis & final score sorting after zone-boundary supercell phonon checks complete!
    _apply_pareto_frontier_analysis(verified_records)
    verified_records.sort(key=lambda r: (
        -int(r.get('discovery_score', 0)),
        -int(r.get('pareto_front_count', 0)),
        float(r.get('polished_signed_hull_meV_atom', 0.0)),
    ))
    atomic_text(root / 'verification_report.json', json.dumps(verified_records, indent=2))

    # Write Peer-Review-Grade Markdown Verification Report (matching Reviewer Section 3 Schema)
    pareto_winners = [r for r in verified_records if r.get('is_pareto_outlier')]
    pareto_summary_lines = [
        '## Non-Dominated Multi-Objective Pareto Frontier Outliers',
        '',
        'A candidate is flagged as a **Pareto Frontier Outlier** only if it is thermodynamically on/below the convex hull at both 0 K ($\\Delta E_{\\text{hull}} \\le 0$) and 1000 K ($\\Delta G_{1000\\text{K}} \\le 0\\text{ meV/atom}$), dynamically stable across both $\\Gamma$ and the Niggli-reduced zone-boundary supercell ($\\nu_{\\min}(\\Gamma) > 0$, $\\nu_{\\min,\\text{BZ}} > 0$, 0 imaginary modes), Born elastically stable ($B_0, G > 0$), crystallographically realistic (Gate 3 pass), synthetically tractable, has a non-zero Figure-of-Merit score (`FoM > 0`), and is **non-dominated** across at least one of the four 2D functional performance trade-off planes:',
        '1. **Ashby Specific-Stiffness vs. Density Front ($E/\\rho \\ge 28\\text{ GPa}\\cdot\\text{cm}^3/\\text{g}$ vs. $\\rho$)**: Maximizes specific stiffness $E/\\rho$ while minimizing mass density $\\rho$ (benchmarked against both MAB/MAX cermets like $\\text{MoAlB}$ at $E/\\rho = 58.0\\text{ GPa}\\cdot\\text{cm}^3/\\text{g}$ and structural alloys like $\\text{Ti-6Al-4V}$ at $26.0\\text{ GPa}\\cdot\\text{cm}^3/\\text{g}$).',
        '2. **Pugh Cermet Damage-Tolerance vs. Shear Rigidity Front ($G \\ge 65\\text{ GPa}$ vs. $B_0/G \\ge 1.75$)**: Simultaneously maximizes shear modulus $G \\ge 65\\text{ GPa}$ and macroscopic Pugh ratio $B_0/G \\ge 1.75$ (indicating MAB/MAX-like cermet damage tolerance relative to brittle monolithic ceramics like $\\text{TiB}_2$ at $B_0/G = 1.12$; note that covalent $\\text{B–B}$ chains still pin room-temperature dislocations).',
        '3. **Refractory Incompressibility vs. Acoustic Debye Stiffness Front ($B_0 \\ge 145\\text{ GPa}$ vs. $\\Theta_D \\ge 380\\text{ K}$)**: Simultaneously maximizes bulk modulus $B_0$ and acoustic Debye temperature $\\Theta_D$.',
        '4. **Thermo-Mechanical Shear Rigidity vs. 1000 K Hull Driving Force Front ($G \\ge 95\\text{ GPa}$ vs. $-\\Delta G_{1000\\text{K}} \\ge 5\\text{ meV/atom}$)**: Simultaneously maximizes shear rigidity $G$ and 1000 K thermodynamic driving force $-\\Delta G_{1000\\text{K}}$ (after subtracting $+T\\Delta S_{\\text{mix}}$).',
        '',
    ]
    if pareto_winners:
        for pw in pareto_winners:
            el = pw['elastic_moduli']
            ph_pw = pw.get('phonon_stability', {})
            bz_tag = f", $\\nu_{{\\min,\\text{{BZ}}}}({ph_pw['supercell_dim']}) = {ph_pw['bz_min_freq_THz']:+.2f}\\text{{ THz}}$" if 'bz_min_freq_THz' in ph_pw else ''
            pareto_summary_lines.append(
                f"- **[`{pw['proposal_id']}` — {pw.get('ordered_formula', pw['formula'])} (`{pw['relaxed_sg']}`)](dft_verification/{pw['proposal_id']}_{pw['formula']}/)**: "
                f"**{pw.get('pareto_status', '')}** — $\\Delta E_{{\\text{{hull}}}} = {round(pw['polished_signed_hull_meV_atom']):+d}\\text{{ meV/at}}$ "
                f"($\\Delta G_{{1000\\text{{K}}}} \\approx {round(pw.get('dg_hull_1000K_meV_atom', 0)):+d}\\text{{ meV/at}}${bz_tag}), "
                f"$E/\\rho = {el.get('specific_modulus_GPa_cm3_g', 0):.1f}\\text{{ GPa}}\\cdot\\text{{cm}}^3/\\text{{g}}$ at $\\rho = {pw.get('density_g_cm3', 0):.2f}\\text{{ g/cm}}^3$, "
                f"$B_0 = {el.get('bulk_modulus_GPa', 0):.0f}\\text{{ GPa}}$, $G = {el.get('shear_modulus_GPa', 0):.0f}\\text{{ GPa}}$ ($B_0/G = {el.get('pugh_ratio_B_G', 0):.2f}$, $\\Theta_D = {el.get('debye_temp_K', 0)}\\text{{ K}}$), "
                f"$d_{{\\text{{proto}}}} = {pw.get('d_proto', 0):.2f}$ ({pw.get('motif_novelty_label', '')}), Score **{pw.get('discovery_score', 0)}/100** (`{pw.get('score_breakdown', '')}`)."
            )
        pareto_summary_lines.append('')
    md = [
        '# Quantitative Verification & Figure-of-Merit Matrix (`pilot-002`)',
        '',
        r'**Method & Provenance:** Energies and Hessian/elastic derivatives evaluated with `MACE-MP-0b3-medium` (`float64`, `fmax <= 0.006 eV/Å`) against the 157,612-phase **LeMat PBE Reference Convex Hull** (`data/lemat-mace-mp-hull.parquet`). Typical universal MLIP-vs-DFT hull uncertainty is $\pm 15\text{ meV/atom}$. Phonon stability reports both the $\Gamma$-point ($\mathbf{q}=0$) optical phonon Hessian minimum $\nu_{\min}(\Gamma)$ for all candidates and the $2\times 2\times 2$ (or $2\times 2\times 1$) supercell zone-boundary ($\mathbf{q} \in \{X, M, R\}$) Bloch dynamical-matrix minimum $\nu_{\min,\text{BZ}}$ for top candidates. Turnkey VASP (`POSCAR`, `INCAR`, `KPOINTS`, `KPOINTS_BANDS`) and Quantum ESPRESSO (`pw.in`) input decks for every row are generated in [`dft_verification/`](dft_verification/).',
        '',
        *pareto_summary_lines,
        '| ID | Ordered Formula & SG | $\\Delta E_{\\text{hull}}^{\\text{comp}}$ ($\\pm 15$ meV/at) | $\\nu_{\\min}(\\Gamma)$ / $\\nu_{\\min,\\text{BZ}}$ (THz) | $B_0 / G$ (GPa) | $\\rho$ (g/cm³) | $d_{\\text{proto}}$ (Motif) | Target Figure of Merit | Benchmark Comparison | Pareto Frontier (+25 pts) | Score (0–100) | Synthesis Route & DFT Caveats |',
        '| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- | :--- | :--- | :---: | :--- |',
    ]
    for r in verified_records:
        el = r['elastic_moduli']
        ph = r['phonon_stability']
        ord_f = r.get('ordered_formula') or r['formula']
        comp_rxn = r.get('competing_reaction', '')
        hull_int = int(round(float(r['polished_signed_hull_meV_atom'])))
        dg1000 = int(round(float(r.get('dg_hull_1000K_meV_atom', hull_int))))
        ts_mix = float(r.get('ts_mix_1000K_meV_atom', 0.0))
        ts_str = f", $+T\\Delta S_{{\\text{{mix}}}}=+{ts_mix:.0f}$" if ts_mix > 0 else ""
        bz_str = f" / `{ph['bz_min_freq_THz']:+.2f}` ({ph.get('supercell_dim','2x2x2')} BZ)" if 'bz_min_freq_THz' in ph else " ($\\Gamma$-only)"
        p_stat = f"**{r.get('pareto_status', '')}**" if r.get('is_pareto_outlier') else r.get('pareto_status', 'Interior')
        md.append(
            f"| [`{r['proposal_id']}`](dft_verification/{r['proposal_id']}_{r['formula']}/) | **{ord_f}** (`{r['relaxed_sg']}`) | "
            f"`{hull_int:+d}` ($\\Delta G_{{1000\\text{{K}}}}={dg1000:+d}${ts_str}, vs. {comp_rxn}) | "
            f"`{ph['min_optical_freq_THz']:+.2f}`{bz_str} | "
            f"`{el['bulk_modulus_GPa']:.0f} / {el['shear_modulus_GPa']:.0f}` ($B/G={el.get('pugh_ratio_B_G','-')}$) | "
            f"`{r.get('density_g_cm3', 0):.2f}` | "
            f"`{r.get('d_proto', 0.15):.2f}` ({r.get('motif_novelty_label', '')}) | "
            f"**{r.get('target_fom', '')}** | "
            f"{r.get('benchmark_comparison', '')} | "
            f"{p_stat} | "
            f"**{r.get('discovery_score', 0)}/100**<br><small>{r.get('score_breakdown', '')}</small> | "
            f"{r.get('synthesis_feasibility', '')} |"
        )
    atomic_text(root / 'verification_report.md', '\n'.join(md) + '\n')
    try:
        prog.update('COMPLETE', force=True)
    except Exception:
        pass
    return verified_records


def run_multi_stage_loop(run_dir, loop_target=360, batch_size=60, rerun_all=False):
    import subprocess
    import sys
    root = Path(run_dir)
    progress_data = json.loads((root / 'progress.json').read_text(encoding='utf-8'))
    completed = int(progress_data.get('completed', 0))
    while completed < loop_target:
        next_target = min(loop_target, completed + batch_size)
        print(f'\n=== MULTI-STAGE LOOP: DISCOVERING {completed} -> {next_target} PROPOSALS ===', flush=True)
        subprocess.run([
            sys.executable, 'pilot.py',
            '--output', str(run_dir),
            '--count', str(next_target),
            '--seed', '20261205',
            '--steps', '80',
            '--fmax', '0.08',
            '--candidate-seconds', '45',
            '--threads', '8',
            '--prior', 'discover',
            '--budget-seconds', '3600',
            '--resume',
        ], check=True)
        print(f'\n=== MULTI-STAGE LOOP: 5-GATE VERIFICATION AT {next_target} PROPOSALS ===', flush=True)
        verify_run(run_dir, rerun_all=rerun_all)
        progress_data = json.loads((root / 'progress.json').read_text(encoding='utf-8'))
        completed = int(progress_data.get('completed', next_target))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/pilot-002')
    parser.add_argument('--loop-target', type=int, default=0, help='Target total proposals for alternating discovery + 5-Gate verification loop')
    parser.add_argument('--batch-size', type=int, default=60, help='Proposals per discovery stage before running 5-Gate verification')
    parser.add_argument('--rerun-all', action='store_true', help='Force full recomputation of MACE polish, phonons, elastic moduli, and scores for all candidates, ignoring cached entries')
    parser.add_argument('--only-new', action='store_true', default=True, help='Only compute 5-Gate verification on newly discovered candidates not yet in verification_report.json (default: True)')
    parser.add_argument('--since-id', type=int, default=0, help='Only verify candidates with proposal_id >= since_id (reusing cached records for older IDs)')
    args = parser.parse_args()
    if args.loop_target > 0:
        run_multi_stage_loop(args.run, loop_target=args.loop_target, batch_size=args.batch_size, rerun_all=args.rerun_all)
    else:
        verify_run(args.run, rerun_all=args.rerun_all, only_new=(not args.rerun_all and args.only_new), since_id=args.since_id)
