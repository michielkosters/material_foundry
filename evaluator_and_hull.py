"""MACE relaxation and a same-model reference hull. No mixed-energy hulls."""
import hashlib
import math
from fractions import Fraction
from pathlib import Path
from sage.all import QQ, Polyhedron
from pymatgen.core import Composition


def decimal_rational(value):
    """Preserve the decimal spelling; Sage QQ does not parse decimal strings."""
    fraction = Fraction(str(value))
    return QQ(fraction.numerator) / fraction.denominator


class ReferenceHull:
    def __init__(self, entries, energy_id):
        if not entries or any(e['energy_id'] != energy_id for e in entries):
            raise ValueError('Hull requires references evaluated with the same energy_id')
        self.entries, self.energy_id = [], energy_id
        for entry in entries:
            composition = Composition(entry['formula'])
            energy = float(entry['energy_per_atom'])
            if not composition or not math.isfinite(energy) or any(v <= 0 for v in composition.values()):
                raise ValueError('References require positive amounts and finite energies')
            self.entries.append((composition, energy))
        self._facets = {}

    def above_hull(self, formula, energy_per_atom, *, energy_id):
        if energy_id != self.energy_id or not math.isfinite(energy_per_atom):
            raise ValueError('Candidate must use the reference energy scale and finite energy')
        composition = Composition(formula)
        if not composition or any(v <= 0 for v in composition.values()):
            raise ValueError('Composition amounts must be positive')
        elements = sorted(str(e) for e in composition.elements)
        def fractions(c):
            amounts = [decimal_rational(c[e]) for e in elements]
            total = sum(amounts)
            return [a/total for a in amounts]
        # Exact rational LP over the supplied floating-energy decimal strings.
        # This does not make model energies physically exact.
        key = tuple(elements)
        if key not in self._facets:
            refs = [(c, e) for c, e in self.entries if set(c.as_dict()) <= set(elements)]
            for element in elements:
                if not any(set(c.as_dict()) == {element} for c, _ in refs):
                    raise ValueError(f'Missing elemental reference: {element}')
            vertices = [fractions(c)[:-1] + [decimal_rational(e)] for c, e in refs]
            poly = Polyhedron(vertices=vertices, rays=[[0] * (len(elements)-1) + [1]], base_ring=QQ)
            self._facets[key] = [(h.b(), tuple(h.A())) for h in poly.inequality_generator() if h.A()[-1] > 0]
        target = fractions(composition)[:-1]
        lower = []
        for b, a in self._facets[key]:
            lower.append(-(b + sum(x*y for x, y in zip(a[:-1], target))) / a[-1])
        if not lower:
            raise ValueError('Reference hull has no lower energy envelope')
        return max(0.0, float(decimal_rational(energy_per_atom) - max(lower)))


class MaceEvaluator:
    def __init__(self, model_path, device='cpu', steps=200, fmax=0.05, default_dtype='float64'):
        from mace.calculators import mace_mp
        self._model_path = str(model_path)
        self._device = device
        self._default_dtype = default_dtype
        self.energy_id = 'mace-mp-0:' + hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
        self.calculator = mace_mp(model=self._model_path, device=device, default_dtype=default_dtype)
        self._fast_calculator = self.calculator if default_dtype == 'float32' else None
        self.steps, self.fmax = steps, fmax

    def get_fast_calculator(self):
        if self._fast_calculator is None:
            from mace.calculators import mace_mp
            self._fast_calculator = mace_mp(model=self._model_path, device=self._device, default_dtype='float32')
        return self._fast_calculator

    def single_point(self, structure):
        import numpy as np
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = self.calculator
        energy = float(atoms.get_potential_energy()) / len(atoms)
        forces = atoms.get_forces()
        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(atoms) else 0.0
        return energy, max_force

    def single_point_fast(self, structure):
        """Fast float32 MACE single-point screening (2.36x faster than float64 with <0.5 meV/at deviation)."""
        import numpy as np
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = self.get_fast_calculator()
        energy = float(atoms.get_potential_energy()) / len(atoms)
        forces = atoms.get_forces()
        max_force = float(np.linalg.norm(forces, axis=1).max()) if len(atoms) else 0.0
        return energy, max_force

    def single_point_batch_fast(self, structures):
        """Vectorized batch float32 MACE single-point screening across multiple crystals simultaneously."""
        from wyckoff_flow_engine import evaluate_mace_batch_fast
        return evaluate_mace_batch_fast(self.get_fast_calculator(), structures)

    def single_point_with_rms(self, structure):
        import numpy as np
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = self.calculator
        energy = float(atoms.get_potential_energy()) / len(atoms)
        forces = atoms.get_forces()
        if len(atoms):
            norms = np.linalg.norm(forces, axis=1)
            max_force = float(norms.max())
            rms_force = float(np.sqrt(np.mean(norms ** 2)))
        else:
            max_force, rms_force = 0.0, 0.0
        return energy, max_force, rms_force

    def quick_prerelax(self, structure, steps=2, use_fast=True):
        """Fast symmetry-preserving MACE BFGS pre-relaxation for ab-initio crystal prescreening."""
        import io
        import numpy as np
        from ase.optimize import BFGS
        from ase.filters import FrechetCellFilter
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor.get_atoms(structure)
        atoms.calc = self.get_fast_calculator() if use_fast else self.calculator
        log_buf = io.StringIO()
        energy = None
        forces = None
        try:
            from ase.constraints import FixSymmetry
            atoms.set_constraint(FixSymmetry(atoms, symprec=1e-2))
            opt = BFGS(FrechetCellFilter(atoms), logfile=log_buf, maxstep=0.14)
            opt.run(fmax=0.08, steps=steps)
            # Read cached energy & forces BEFORE clearing constraint so ASE does not trigger an extra MACE evaluation
            energy = float(atoms.get_potential_energy()) / len(atoms)
            forces = atoms.get_forces()
        except Exception:
            pass
        finally:
            atoms.set_constraint([])
        if energy is None or forces is None:
            energy = float(atoms.get_potential_energy()) / len(atoms)
            forces = atoms.get_forces()
        rms_force = float(np.sqrt(np.mean(np.linalg.norm(forces, axis=1) ** 2))) if len(atoms) else 0.0
        v_atom = atoms.get_volume() / max(1, len(atoms))
        if not np.isfinite(energy) or not (2.0 < v_atom < 140.0):
            raise ValueError('Pre-relaxation out of bounds')
        return AseAtomsAdaptor.get_structure(atoms), energy, rms_force

    def relax(self, structure, output, on_step=None, seed=0, rattle=0.0, sym_warmup=True):
        import io
        import time
        import numpy as np
        from ase.optimize import BFGS
        from ase.filters import FrechetCellFilter
        from ase.io import write
        from pymatgen.io.ase import AseAtomsAdaptor
        atoms = AseAtomsAdaptor.get_atoms(structure)
        # Two-stage float32 -> float64 relaxation: 2.36x faster BFGS trajectory in float32,
        # followed by exact float64 endpoint verification/polish so energy_id is 100% exact.
        atoms.calc = self.get_fast_calculator()
        start = time.perf_counter()
        log_buffer = io.StringIO()
        self.last_steps = 0
        cached_energy = None

        def check_limits(step_count):
            self.last_steps = step_count
            volume_per_atom = atoms.get_volume() / len(atoms)
            if (not np.isfinite(atoms.positions).all() or not 1 < volume_per_atom < 200
                    or np.linalg.cond(atoms.cell.array) > 100):
                raise ValueError('Relaxation exceeded geometry resource limits')
            if on_step:
                on_step(step_count)

        try:
            used_steps = 0
            converged = False
            sym_converged = False
            if sym_warmup and len(atoms) > 1 and self.steps >= 10:
                try:
                    from ase.constraints import FixSymmetry
                    atoms.set_constraint(FixSymmetry(atoms, symprec=1e-2))
                    warmup_budget = min(6, max(4, int(self.steps * 0.12)))
                    opt_sym = BFGS(FrechetCellFilter(atoms), logfile=log_buffer)
                    opt_sym.attach(lambda: check_limits(opt_sym.nsteps), interval=1)
                    sym_converged = bool(opt_sym.run(fmax=self.fmax, steps=warmup_budget))
                    used_steps = opt_sym.nsteps
                except Exception:
                    pass
                finally:
                    atoms.set_constraint([])

            if sym_converged and not rattle:
                converged = True
                self.last_steps = used_steps
            else:
                if rattle:
                    atoms.rattle(stdev=min(float(rattle), 0.004), seed=int(seed))
                remaining_steps = min(5, max(1, self.steps - used_steps)) if sym_converged else max(1, self.steps - used_steps)
                optimizer = BFGS(FrechetCellFilter(atoms), logfile=log_buffer)
                optimizer.attach(lambda: check_limits(used_steps + optimizer.nsteps), interval=1)
                converged = bool(optimizer.run(fmax=self.fmax, steps=remaining_steps))
                self.last_steps = used_steps + optimizer.nsteps

            # Stage B: Exact float64 verification & polish on self.calculator
            atoms.calc = self.calculator
            cached_energy = float(atoms.get_potential_energy()) / len(atoms)
            if converged and self._default_dtype == 'float64' and len(atoms) > 0:
                f64 = atoms.get_forces()
                max_f64 = float(np.linalg.norm(f64, axis=1).max())
                if max_f64 > self.fmax * 1.05 and self.last_steps < self.steps:
                    polish_opt = BFGS(FrechetCellFilter(atoms), logfile=log_buffer)
                    converged = bool(polish_opt.run(fmax=self.fmax, steps=min(2, self.steps - self.last_steps)))
                    self.last_steps += polish_opt.nsteps
                    cached_energy = float(atoms.get_potential_energy()) / len(atoms)
        finally:
            if output is not None:
                Path(str(output) + '.log').write_text(log_buffer.getvalue(), encoding='utf-8')

        energy = cached_energy if cached_energy is not None else (float(atoms.get_potential_energy()) / len(atoms))
        if not math.isfinite(energy):
            raise ValueError('Relaxation returned a nonfinite energy')
        relaxed_struct = AseAtomsAdaptor.get_structure(atoms)
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            from pymatgen.io.cif import CifWriter
            sga = SpacegroupAnalyzer(relaxed_struct, symprec=3e-2)
            std_prim = sga.get_primitive_standard_structure()
            if len(std_prim) == len(relaxed_struct):
                angles = std_prim.lattice.angles
                if max(angles) > 120.0 or min(angles) < 60.0:
                    niggli = std_prim.get_reduced_structure('niggli')
                    if len(niggli) == len(relaxed_struct):
                        relaxed_struct = niggli
                else:
                    relaxed_struct = std_prim
            if output is not None:
                CifWriter(relaxed_struct).write_file(str(output) + '.cif')
        except Exception:
            if output is not None:
                write(str(output) + '.cif', atoms)
        return relaxed_struct, energy, converged, time.perf_counter() - start
