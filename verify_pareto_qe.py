#!/usr/bin/env python3
"""First-Principles Quantum ESPRESSO (pw.x, PBE + SSSP Efficiency) Verification
of the 14 Pareto Frontier Outliers in runs/pilot-002.

For each Pareto Frontier Outlier (and its elemental reference crystals), runs
Quantum ESPRESSO 7.6 `pw.x` to verify:
  1. Kohn-Sham PBE SCF Electronic Convergence (Total Energy in Ry and eV/atom)
  2. First-Principles PBE Formation Enthalpy (dH_form_QE in eV/atom vs dH_form_MACE)
  3. Hellmann-Feynman Residual Forces on MACE-Polished Coordinates (max |F_QE|, RMS |F_QE| in eV/A)
  4. Ab-Initio Stress Tensor & Hydrostatic Pressure P_QE (kbar & GPa)
  5. Fermi Energy E_F (eV)

Supports `--only-new` incremental caching in `runs/pilot-002/qe_pareto_verification.json`.
"""
import argparse
import json
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from ase.filters import FrechetCellFilter
from ase.data import atomic_masses, atomic_numbers
from ase.io import write as ase_write
from ase.optimize import BFGS
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

RY_TO_EV = 13.605693122994
BOHR_TO_ANG = 0.529177210903
RY_BOHR_TO_EV_ANG = RY_TO_EV / BOHR_TO_ANG  # 25.711033 eV/A

SSSP_DIR = Path("./pseudos/sssp_efficiency")
PW_BIN = "pw.x"


def build_pseudo_map(pseudo_dir: Path) -> dict:
    """Map each chemical symbol to its SSSP Efficiency UPF filename."""
    mapping = {}
    for p in sorted(pseudo_dir.iterdir()):
        if p.suffix.lower() != ".upf":
            continue
        m = re.match(r"^([A-Za-z]{1,2})[\._-]", p.name)
        if m:
            el = m.group(1).capitalize()
            mapping[el] = p.name
    return mapping


def get_elemental_ground_state_structure(el: str) -> Structure:
    """Standard experimental 0K elemental reference crystal structures."""
    refs = {
        "Al": ("fcc", 4.046),
        "Cu": ("fcc", 3.615),
        "Ni": ("fcc", 3.524),
        "Ir": ("fcc", 3.839),
        "La": ("fcc", 5.305),
        "Cr": ("bcc", 2.884),
        "Fe": ("bcc", 2.866),
        "Mo": ("bcc", 3.147),
        "Nb": ("bcc", 3.300),
        "Ta": ("bcc", 3.303),
        "V": ("bcc", 3.030),
        "W": ("bcc", 3.165),
        "Mn": ("bcc", 2.910),
        "Ti": ("hcp", (2.951, 4.684)),
        "Zr": ("hcp", (3.232, 5.147)),
        "Sc": ("hcp", (3.309, 5.268)),
        "Lu": ("hcp", (3.505, 5.549)),
        "Os": ("hcp", (2.734, 4.317)),
        "Si": ("diamond", 5.431),
        "C": ("diamond", 3.567),
        "B": ("b12_icosa", 5.057),
    }
    kind, param = refs[el]
    if kind == "fcc":
        a = float(param)
        lat = Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
        return Structure(lat, [el], [[0, 0, 0]])
    if kind == "bcc":
        a = float(param)
        lat = Lattice([[-a / 2, a / 2, a / 2], [a / 2, -a / 2, a / 2], [a / 2, a / 2, -a / 2]])
        return Structure(lat, [el], [[0, 0, 0]])
    if kind == "hcp":
        a, c = param
        lat = Lattice.hexagonal(a, c)
        return Structure(lat, [el, el], [[1 / 3, 2 / 3, 1 / 4], [2 / 3, 1 / 3, 3 / 4]])
    if kind == "diamond":
        a = float(param)
        lat = Lattice([[0, a / 2, a / 2], [a / 2, 0, a / 2], [a / 2, a / 2, 0]])
        return Structure(lat, [el, el], [[0, 0, 0], [1 / 4, 1 / 4, 1 / 4]])
    if kind == "b12_icosa":
        # Alpha-rhombohedral B12 (R-3m, 12 atoms)
        a = 5.057
        lat = Lattice.rhombohedral(a, 58.06)
        b_coords = [
            [0.0104, 0.0104, 0.6540],
            [0.6540, 0.0104, 0.0104],
            [0.0104, 0.6540, 0.0104],
            [0.9896, 0.9896, 0.3460],
            [0.3460, 0.9896, 0.9896],
            [0.9896, 0.3460, 0.9896],
            [0.2211, 0.2211, 0.6305],
            [0.6305, 0.2211, 0.2211],
            [0.2211, 0.6305, 0.2211],
            [0.7789, 0.7789, 0.3695],
            [0.3695, 0.7789, 0.7789],
            [0.7789, 0.3695, 0.7789],
        ]
        return Structure(lat, ["B"] * 12, b_coords)
    raise ValueError(f"Unknown elemental reference for {el}")


def choose_kgrid(structure: Structure, k_spacing: float = 0.24) -> tuple:
    """Choose a Gamma-centered Monkhorst-Pack grid based on reciprocal lattice lengths."""
    recip_lengths = [2.0 * math.pi / l for l in structure.lattice.abc]
    kgrid = [max(2, min(8, int(math.ceil(rl / k_spacing)))) for rl in recip_lengths]
    return tuple(kgrid)


def write_qe_input(
    structure: Structure,
    pseudo_map: dict,
    pseudo_dir: Path,
    ecutwfc: float = 45.0,
    ecutrho: float = 360.0,
    k_spacing: float = 0.24,
) -> str:
    """Generate a Quantum ESPRESSO pw.x SCF input file with forces and stress."""
    species_unique = sorted({str(s.specie) for s in structure})
    nat = len(structure)
    ntyp = len(species_unique)
    kx, ky, kz = choose_kgrid(structure, k_spacing=k_spacing)

    lines = [
        "&CONTROL",
        "  calculation = 'scf',",
        "  restart_mode = 'from_scratch',",
        f"  pseudo_dir = '{pseudo_dir.as_posix()}',",
        "  outdir = './out',",
        "  prefix = 'pwscf',",
        "  tprnfor = .true.,",
        "  tstress = .true.,",
        "  verbosity = 'low',",
        "  disk_io = 'none',",
        "/",
        "&SYSTEM",
        "  ibrav = 0,",
        f"  nat = {nat},",
        f"  ntyp = {ntyp},",
        f"  ecutwfc = {ecutwfc:.1f},",
        f"  ecutrho = {ecutrho:.1f},",
        "  occupations = 'smearing',",
        "  smearing = 'mv',",
        "  degauss = 0.020,",
        "/",
        "&ELECTRONS",
        "  conv_thr = 1.0d-5,",
        "  mixing_beta = 0.35,",
        "  electron_maxstep = 150,",
        "/",
        "ATOMIC_SPECIES",
    ]
    for el in species_unique:
        mass = float(atomic_masses[atomic_numbers[el]])
        upf = pseudo_map[el]
        lines.append(f"  {el:3s}  {mass:10.4f}  {upf}")

    lines.append("CELL_PARAMETERS {angstrom}")
    for vec in structure.lattice.matrix:
        lines.append(f"  {vec[0]:16.10f}  {vec[1]:16.10f}  {vec[2]:16.10f}")

    lines.append("ATOMIC_POSITIONS {crystal}")
    for site in structure:
        fc = site.frac_coords
        lines.append(f"  {str(site.specie):3s}  {fc[0]:16.10f}  {fc[1]:16.10f}  {fc[2]:16.10f}")

    lines.append("K_POINTS {automatic}")
    lines.append(f"  {kx} {ky} {kz} 0 0 0")
    return "\n".join(lines) + "\n"


def parse_qe_output(stdout_text: str, nat: int) -> dict:
    """Parse total energy, Fermi energy, Hellmann-Feynman forces, and stress/pressure from pw.x output."""
    converged = "convergence has been achieved" in stdout_text
    e_matches = re.findall(r"!\s+total energy\s+=\s+([-\d\.]+)\s+Ry", stdout_text)
    total_ry = float(e_matches[-1]) if e_matches else None
    total_ev = total_ry * RY_TO_EV if total_ry is not None else None
    ev_per_atom = total_ev / max(nat, 1) if total_ev is not None else None

    ef_matches = re.findall(r"the Fermi energy is\s+([-\d\.]+)\s+ev", stdout_text)
    fermi_ev = float(ef_matches[-1]) if ef_matches else None

    # Parse forces (Ry/au -> eV/A)
    force_norms = []
    for m in re.finditer(
        r"atom\s+\d+\s+type\s+\d+\s+force\s*=\s*([-\d\.]+)\s+([-\d\.]+)\s+([-\d\.]+)",
        stdout_text,
        re.IGNORECASE,
    ):
        fx, fy, fz = float(m.group(1)), float(m.group(2)), float(m.group(3))
        f_ev_a = math.sqrt(fx * fx + fy * fy + fz * fz) * RY_BOHR_TO_EV_ANG
        force_norms.append(f_ev_a)

    if not force_norms:
        tf_matches = re.findall(r"Total force\s*=\s*([-\d\.]+)", stdout_text, re.IGNORECASE)
        if tf_matches:
            tot_f_ev_a = float(tf_matches[-1]) * RY_BOHR_TO_EV_ANG
            force_norms = [tot_f_ev_a / math.sqrt(max(nat, 1))]

    max_force_ev_a = round(float(max(force_norms)), 4) if force_norms else None
    rms_force_ev_a = round(float(np.sqrt(np.mean(np.square(force_norms)))), 4) if force_norms else None

    # Parse pressure (kbar -> GPa)
    p_matches = re.findall(r"total\s+stress\s+\(Ry/bohr\*\*3\)\s+\(kbar\)\s+P=\s*([-\d\.]+)", stdout_text)
    p_kbar = float(p_matches[-1]) if p_matches else None
    p_gpa = round(p_kbar / 10.0, 2) if p_kbar is not None else None

    n_scf_matches = re.findall(r"convergence has been achieved in\s+(\d+)\s+iterations", stdout_text)
    scf_steps = int(n_scf_matches[-1]) if n_scf_matches else None

    return {
        "converged": converged,
        "scf_iterations": scf_steps,
        "total_energy_Ry": round(total_ry, 6) if total_ry is not None else None,
        "energy_eV_atom": round(ev_per_atom, 6) if ev_per_atom is not None else None,
        "fermi_energy_eV": round(fermi_ev, 3) if fermi_ev is not None else None,
        "max_force_eV_A": max_force_ev_a,
        "rms_force_eV_A": rms_force_ev_a,
        "pressure_kbar": round(p_kbar, 2) if p_kbar is not None else None,
        "pressure_GPa": p_gpa,
    }


def run_qe_scf(
    structure: Structure,
    pseudo_map: dict,
    pseudo_dir: Path,
    ecutwfc: float = 45.0,
    ecutrho: float = 360.0,
    k_spacing: float = 0.24,
    np_mpi: int = 4,
    omp_threads: int = 2,
    timeout_s: int = 480,
    save_dir: Path | None = None,
) -> dict:
    """Execute Quantum ESPRESSO pw.x in a temporary directory and return parsed DFT results."""
    inp_text = write_qe_input(
        structure, pseudo_map, pseudo_dir, ecutwfc=ecutwfc, ecutrho=ecutrho, k_spacing=k_spacing
    )
    if save_dir is not None and save_dir.exists():
        (save_dir / "pw_scf.in").write_text(inp_text, encoding="utf-8")
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="qe_verify_") as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "out").mkdir(exist_ok=True)
        in_file = tmp_path / "pw.in"
        in_file.write_text(inp_text, encoding="utf-8")

        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(omp_threads)
        env["OMPI_MCA_btl_vader_single_copy_mechanism"] = "none"
        cmd = [
            "mpirun",
            "--oversubscribe",
            "-np",
            str(np_mpi),
            PW_BIN,
            "-in",
            "pw.in",
        ]
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            elapsed = round(time.perf_counter() - t0, 2)
            if save_dir is not None and save_dir.exists():
                (save_dir / "pw_scf.out").write_text(proc.stdout, encoding="utf-8")
            parsed = parse_qe_output(proc.stdout, len(structure))
            parsed["wall_seconds"] = elapsed
            parsed["kgrid"] = list(choose_kgrid(structure, k_spacing=k_spacing))
            if not parsed["converged"]:
                parsed["stderr_tail"] = (proc.stdout[-600:] + "\n" + proc.stderr[-400:]).strip()
            return parsed
        except subprocess.TimeoutExpired:
            return {
                "converged": False,
                "wall_seconds": round(time.perf_counter() - t0, 2),
                "error": f"TIMEOUT ({timeout_s}s)",
            }


def main():
    parser = argparse.ArgumentParser(description="Verify 14 Pareto Frontier Outliers with Quantum ESPRESSO pw.x")
    parser.add_argument("--run", default="runs/pilot-002", help="Run directory")
    parser.add_argument("--only-new", action="store_true", help="Reuse cached QE verification results")
    parser.add_argument("--ecutwfc", type=float, default=45.0, help="Wavefunction cutoff (Ry)")
    parser.add_argument("--ecutrho", type=float, default=360.0, help="Charge density cutoff (Ry)")
    parser.add_argument("--k-spacing", type=float, default=0.25, help="Reciprocal k-point spacing (A^-1)")
    args = parser.parse_args()

    run_dir = Path(args.run)
    ver_report_path = run_dir / "verification_report.json"
    events_path = run_dir / "telemetry" / "events.jsonl"
    qe_out_path = run_dir / "qe_pareto_verification.json"
    qe_refs_path = run_dir / "qe_elemental_refs.json"

    pseudo_map = build_pseudo_map(SSSP_DIR)
    print(f"[QE-VERIFY] Loaded {len(pseudo_map)} SSSP Efficiency UPF pseudopotentials from {SSSP_DIR}")

    ver_report = json.loads(ver_report_path.read_text(encoding="utf-8"))
    pareto_entries = [r for r in ver_report if r.get("is_pareto_outlier")]
    pareto_ids = {str(r["proposal_id"]): r for r in pareto_entries}
    print(f"[QE-VERIFY] Found {len(pareto_entries)} Pareto Frontier Outliers to verify with Quantum ESPRESSO 7.6 (pw.x)")

    # Load MACE-polished primitive POSCAR structures directly from dft_input_dir
    structures_by_id = {}
    dft_dirs_by_id = {}
    for r in pareto_entries:
        pid = str(r["proposal_id"])
        dft_dir = Path(r.get("dft_input_dir", ""))
        poscar_path = dft_dir / "POSCAR"
        if not poscar_path.exists():
            dft_dir = run_dir / "dft_verification" / f"{pid}_{r['formula']}"
            poscar_path = dft_dir / "POSCAR"
        dft_dirs_by_id[pid] = dft_dir
        structures_by_id[pid] = Structure.from_file(str(poscar_path)).get_primitive_structure(tolerance=0.05)

    # Step 1: Ensure Elemental Reference Energies in QE (for formation enthalpy dH_form_QE)
    needed_elements = set()
    for pid, s in structures_by_id.items():
        for el in s.composition.elements:
            needed_elements.add(str(el))

    el_refs = {}
    if qe_refs_path.exists():
        try:
            el_refs = json.loads(qe_refs_path.read_text(encoding="utf-8"))
        except Exception:
            el_refs = {}

    for el in sorted(needed_elements):
        if el in el_refs and el_refs[el].get("converged") and el_refs[el].get("energy_eV_atom") is not None:
            continue
        el_s = get_elemental_ground_state_structure(el)
        print(f"  [QE-REF] Computing Quantum ESPRESSO PBE elemental reference for {el:2s} ({len(el_s)} atoms)...", flush=True)
        res_el = run_qe_scf(
            el_s,
            pseudo_map,
            SSSP_DIR,
            ecutwfc=args.ecutwfc,
            ecutrho=args.ecutrho,
            k_spacing=args.k_spacing,
        )
        el_refs[el] = res_el
        qe_refs_path.write_text(json.dumps(el_refs, indent=2), encoding="utf-8")
        print(
            f"           -> {el:2s}: converged={res_el.get('converged')} | E={res_el.get('energy_eV_atom')} eV/at | P={res_el.get('pressure_GPa')} GPa ({res_el.get('wall_seconds')}s)",
            flush=True,
        )

    # Step 2: Verify all 14 Pareto Frontier Outliers in Quantum ESPRESSO pw.x
    cached_results = {}
    if args.only_new and qe_out_path.exists():
        try:
            for item in json.loads(qe_out_path.read_text(encoding="utf-8")):
                if item.get("qe_converged"):
                    cached_results[str(item["proposal_id"])] = item
        except Exception:
            cached_results = {}

    verified_list = []
    for idx, r in enumerate(sorted(pareto_entries, key=lambda x: float(x.get("polished_signed_hull_meV_atom", 0.0))), 1):
        pid = str(r["proposal_id"])
        formula = r["formula"]
        if args.only_new and pid in cached_results:
            c_rec = cached_results[pid]
            c_dh = c_rec.get("qe_formation_energy_eV_atom")
            c_p = c_rec.get("qe_pressure_GPa")
            c_f = c_rec.get("qe_max_force_eV_A")
            c_rec["qe_dft_verified"] = bool(
                c_rec.get("qe_converged")
                and (c_dh is not None and c_dh < 0.0)
                and (c_f is None or c_f < 1.0)
                and (c_p is not None and abs(c_p) < 20.0)
            )
            print(
                f"[{idx:2d}/{len(pareto_entries)}] [{pid}] {formula:14s} -> Reusing cached QE verification (dH_QE={c_dh:+.4f} eV/at, P={c_p} GPa, PASS={c_rec['qe_dft_verified']})",
                flush=True,
            )
            verified_list.append(c_rec)
            qe_out_path.write_text(json.dumps(verified_list, indent=2), encoding="utf-8")
            continue

        pol_s = structures_by_id[pid]

        print(
            f"[{idx:2d}/{len(pareto_entries)}] [{pid}] {formula:14s} ({r.get('relaxed_sg')}, {len(pol_s)} atoms) -> Running Quantum ESPRESSO pw.x SCF...",
            flush=True,
        )
        qe_res = run_qe_scf(
            pol_s,
            pseudo_map,
            SSSP_DIR,
            ecutwfc=args.ecutwfc,
            ecutrho=args.ecutrho,
            k_spacing=args.k_spacing,
            save_dir=dft_dirs_by_id.get(pid),
        )

        # Compute QE PBE formation enthalpy dH_form_QE (eV/atom)
        dh_qe_ev_atom = None
        if qe_res.get("converged") and qe_res.get("energy_eV_atom") is not None:
            comp_dict = pol_s.composition.fractional_composition.get_el_amt_dict()
            if all(el in el_refs and el_refs[el].get("energy_eV_atom") is not None for el in comp_dict):
                e_elem_sum = sum(frac * float(el_refs[el]["energy_eV_atom"]) for el, frac in comp_dict.items())
                dh_qe_ev_atom = round(float(qe_res["energy_eV_atom"]) - e_elem_sum, 4)

        mace_dh = float(r.get("formation_energy_eV_atom", 0.0))
        max_f = qe_res.get("max_force_eV_A")
        p_gpa = qe_res.get("pressure_GPa")
        qe_pass = bool(
            qe_res.get("converged")
            and (dh_qe_ev_atom is not None and dh_qe_ev_atom < 0.0)
            and (max_f is None or max_f < 1.0)
            and (p_gpa is not None and abs(p_gpa) < 20.0)
        )

        rec = {
            "proposal_id": pid,
            "formula": formula,
            "relaxed_sg": r.get("relaxed_sg"),
            "atoms_primitive": len(pol_s),
            "mace_signed_hull_meV_atom": r.get("polished_signed_hull_meV_atom"),
            "mace_formation_energy_eV_atom": mace_dh,
            "qe_formation_energy_eV_atom": dh_qe_ev_atom,
            "qe_mace_formation_diff_meV_atom": (
                round((dh_qe_ev_atom - mace_dh) * 1000.0, 1) if dh_qe_ev_atom is not None else None
            ),
            "qe_converged": qe_res.get("converged", False),
            "qe_scf_iterations": qe_res.get("scf_iterations"),
            "qe_total_energy_Ry": qe_res.get("total_energy_Ry"),
            "qe_energy_eV_atom": qe_res.get("energy_eV_atom"),
            "qe_fermi_energy_eV": qe_res.get("fermi_energy_eV"),
            "qe_max_force_eV_A": max_f,
            "qe_rms_force_eV_A": qe_res.get("rms_force_eV_A"),
            "qe_pressure_kbar": qe_res.get("pressure_kbar"),
            "qe_pressure_GPa": p_gpa,
            "qe_kgrid": qe_res.get("kgrid"),
            "qe_wall_seconds": qe_res.get("wall_seconds"),
            "qe_dft_verified": qe_pass,
            "d_proto": r.get("d_proto"),
            "B_H_GPa": r.get("elastic_moduli", {}).get("bulk_modulus_GPa"),
            "G_H_GPa": r.get("elastic_moduli", {}).get("shear_modulus_GPa"),
            "E_rho": r.get("elastic_moduli", {}).get("specific_modulus_GPa_cm3_g"),
            "debye_K": r.get("elastic_moduli", {}).get("debye_temp_K"),
            "min_phonon_THz": r.get("phonon_stability", {}).get("min_optical_freq_THz"),
        }
        verified_list.append(rec)
        qe_out_path.write_text(json.dumps(verified_list, indent=2), encoding="utf-8")
        dh_str = f"{dh_qe_ev_atom:+.4f}" if dh_qe_ev_atom is not None else "N/A"
        print(
            f"       -> QE-PBE: conv={rec['qe_converged']} ({rec['qe_scf_iterations']} it, {rec['qe_wall_seconds']}s) | "
            f"dH_QE={dh_str} eV/at (vs MACE {mace_dh:+.4f}) | "
            f"max|F|={max_f} eV/A | P={p_gpa} GPa | PASS={qe_pass}",
            flush=True,
        )

    # Sync QE verification results back into verification_report.json
    qe_by_pid = {str(x["proposal_id"]): x for x in verified_list}
    for r in ver_report:
        pid = str(r.get("proposal_id"))
        if pid in qe_by_pid:
            q = qe_by_pid[pid]
            r["qe_dft_verified"] = q.get("qe_dft_verified", False)
            r["qe_formation_energy_eV_atom"] = q.get("qe_formation_energy_eV_atom")
            r["qe_mace_formation_diff_meV_atom"] = q.get("qe_mace_formation_diff_meV_atom")
            r["qe_pressure_GPa"] = q.get("qe_pressure_GPa")
            r["qe_fermi_energy_eV"] = q.get("qe_fermi_energy_eV")
            r["qe_scf_iterations"] = q.get("qe_scf_iterations")
            r["qe_total_energy_Ry"] = q.get("qe_total_energy_Ry")
    ver_report_path.write_text(json.dumps(ver_report, indent=2), encoding="utf-8")

    n_pass = sum(1 for x in verified_list if x.get("qe_dft_verified"))
    print(f"\n[QE-VERIFY COMPLETE] {n_pass}/{len(verified_list)} Pareto Frontier Outliers verified by Quantum ESPRESSO 7.6 PBE DFT!")


if __name__ == "__main__":
    main()
