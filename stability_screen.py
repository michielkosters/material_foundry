"""Independent single-checkpoint, fixed-geometry thermodynamic screening.

Not the full LeMat benchmark: evaluates MACE-small-relaxed endpoints with
MACE-MP-0b3-medium against its compatible published reference hull. No DFT claim.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linprog
from ase.data import atomic_numbers
from report_io import atomic_text

HULL_REVISION = '70d505bb294c16658a9551e2b81ace69a28d9790'
BENCHMARK_REVISION = 'fbf1ba4855934acb8fe87135315504899064080d'


class TabulatedHull:
    def __init__(self, records):
        best = {}
        self.skipped = 0
        for row in records:
            species = row['species_at_sites']
            count = int(row['nsites'])
            if count != len(species) or count < 1:
                raise ValueError('Invalid reference atom count')
            value = row['mace_mp_energy']
            if value is None or not math.isfinite(float(value)):
                self.skipped += 1
                continue
            amounts = Counter(species)
            divisor = math.gcd(*amounts.values())
            key = tuple(sorted((str(s), n//divisor) for s, n in amounts.items()))
            energy = float(value)/count
            if key not in best or energy < best[key][0]:
                best[key] = (energy, str(row['immutable_id']))
        self.references = []
        self.known_compositions = set(best.keys())
        self._element_energies = {}
        mask_lo, mask_hi = [], []
        for amounts, (energy, identifier) in best.items():
            mask = 0
            lo, hi = 0, 0
            for s, _ in amounts:
                z = atomic_numbers[s]
                mask |= (1 << z)
                if z < 64:
                    lo |= (1 << z)
                else:
                    hi |= (1 << (z - 64))
            if len(amounts) == 1:
                self._element_energies[amounts[0][0]] = energy
            self.references.append((mask, dict(amounts), energy, identifier))
            mask_lo.append(lo)
            mask_hi.append(hi)
        self._mask_lo = np.array(mask_lo, dtype=np.uint64)
        self._mask_hi = np.array(mask_hi, dtype=np.uint64)
        self.cache = {}

    @classmethod
    def from_parquet(cls, path):
        import pyarrow.parquet as pq
        table = pq.read_table(path, columns=['immutable_id', 'nsites', 'species_at_sites', 'mace_mp_energy'])
        cols = table.to_pydict()
        records = (
            dict(immutable_id=i, nsites=n, species_at_sites=s, mace_mp_energy=e)
            for i, n, s, e in zip(cols['immutable_id'], cols['nsites'], cols['species_at_sites'], cols['mace_mp_energy'])
        )
        return cls(records)

    def has_composition(self, composition):
        amounts = {str(k): int(round(v)) for k, v in composition.reduced_composition.get_el_amt_dict().items()}
        divisor = math.gcd(*amounts.values())
        key = tuple(sorted((s, n // divisor) for s, n in amounts.items()))
        return key in self.known_compositions

    def evaluate(self, composition, energy_per_atom):
        amounts = composition.get_el_amt_dict()
        elements = tuple(sorted(amounts))
        if elements not in self.cache:
            for symbol in elements:
                if symbol not in self._element_energies:
                    raise ValueError(f'Missing elemental reference for {symbol}')
            lo, hi = 0, 0
            for s in elements:
                z = atomic_numbers[s]
                if z < 64:
                    lo |= (1 << z)
                else:
                    hi |= (1 << (z - 64))
            sys_lo = np.uint64(lo)
            sys_hi = np.uint64(hi)
            indices = np.flatnonzero(((self._mask_lo & ~sys_lo) == 0) & ((self._mask_hi & ~sys_hi) == 0))
            all_refs = [self.references[int(i)] for i in indices]
            elem_e = {s: self._element_energies[s] for s in elements}
            # Prune strictly unstable non-elemental phases (above elemental linear combination) to shrink LP
            refs = [
                r for r in all_refs
                if len(r[1]) == 1 or r[2] <= sum(n * elem_e[s] for s, n in r[1].items()) / sum(r[1].values()) + 1e-7
            ]
            vectors = np.array([[r[1].get(s, 0)/sum(r[1].values()) for s in elements] for r in refs]).T
            self.cache[elements] = refs, vectors, np.array([r[2] for r in refs]), len(all_refs)
        refs, vectors, energies, total_ref_count = self.cache[elements]
        target = [amounts[s]/composition.num_atoms for s in elements]
        result = linprog(energies, A_eq=vectors, b_eq=target, bounds=(0, None), method='highs')
        if not result.success or not np.allclose(vectors @ result.x, target, atol=1e-8, rtol=0):
            raise ValueError(f'Hull LP failed: {result.message}')
        signed = energy_per_atom-float(result.fun)
        elem_ref = sum(amounts[s] * self._element_energies[s] for s in elements) / composition.num_atoms
        formation_e = float(energy_per_atom - elem_ref)
        phases = [dict(reference_id=r[3], composition=r[1], atom_fraction=float(weight))
                  for r, weight in zip(refs, result.x) if weight > 1e-8]
        return dict(signed_decomposition_energy_eV_atom=signed,
                    energy_above_reference_hull_eV_atom=max(0.0, signed),
                    formation_energy_eV_atom=formation_e,
                    within_0p1=signed <= .1, on_or_below_reference_hull=signed <= 0,
                    competing_phases=phases, reference_count=total_ref_count)


def sha256(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/pilot-001')
    parser.add_argument('--output', default='runs/pilot-001/stability')
    parser.add_argument('--model', default='models/mace-mp-0b3-medium.model')
    parser.add_argument('--hull', default='data/lemat-mace-mp-hull.parquet')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    import pandas as pd
    from mace.calculators import mace_mp
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pilot import deadline
    original = [json.loads(line) for line in (Path(args.run)/'telemetry/events.jsonl').read_text().splitlines()]
    candidates = [row for row in original if row.get('converged') and row.get('relaxed_cif')]
    manifest = dict(model='MACE-MP-0b3-medium', model_sha256=sha256(args.model),
                    hull_sha256=sha256(args.hull), hull_revision=HULL_REVISION,
                    matching_model_evidence_revision=BENCHMARK_REVISION,
                    energy_column='mace_mp_energy', energy_units='total eV divided by nsites',
                    geometry='Fixed endpoints from MACE-MP-0-small relaxation; no medium re-relaxation',
                    novelty='Original MP-20-train matching only',
                    source_sha256=sha256(__file__), input_events_sha256=sha256(Path(args.run)/'telemetry/events.jsonl'),
                    started_utc=datetime.now(timezone.utc).isoformat(), record_verified=False)
    atomic_text(output/'manifest.json', json.dumps(manifest, indent=2))
    table = pd.read_parquet(args.hull, columns=['immutable_id', 'nsites', 'species_at_sites', 'mace_mp_energy'])
    hull = TabulatedHull(table.to_dict('records'))
    calculator = mace_mp(model=args.model, device='cpu', default_dtype='float64')
    results = []
    status = 'COMPLETE'
    for index, candidate in enumerate(candidates):
        if (Path(args.run)/'STOP').exists() or (output/'STOP').exists() or time.monotonic()-start > 600:
            status = 'STOPPED: user request or 600-second budget'
            break
        result = dict(proposal_id=candidate['proposal_id'], formula=candidate['formula'],
                      unique=candidate.get('unique'), novel=candidate.get('novel'))
        try:
            with deadline(30):
                structure = Structure.from_file(candidate['relaxed_cif'])
                atoms = AseAtomsAdaptor.get_atoms(structure)
                atoms.calc = calculator
                energy = float(atoms.get_potential_energy())/len(atoms)
                forces = atoms.get_forces()
                if not math.isfinite(energy) or not np.isfinite(forces).all():
                    raise ValueError('Nonfinite model output')
                result.update(energy_per_atom=energy, max_force_eV_A=float(np.linalg.norm(forces, axis=1).max()),
                              **hull.evaluate(structure.composition, energy), status='screened')
        except Exception as error:
            result.update(status='failed', reason=f'{type(error).__name__}: {error}')
        results.append(result)
        with (output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(result, allow_nan=False)+'\n')
        atomic_text(output/'progress.json', json.dumps(dict(completed=index+1, planned=len(candidates),
                    elapsed_seconds=time.monotonic()-start, latest=result), indent=2), required=False)
        print(f'Stability screening {index+1}/{len(candidates)}: {result}', flush=True)
    scored = [row for row in results if row['status'] == 'screened']
    joint = [row for row in scored if row['within_0p1'] and row['unique'] and row['novel']]
    summary = dict(status=status, candidates_planned=len(candidates), assessed=len(scored),
                   failed=len(results)-len(scored), original_proposals=len(original),
                   predicted_within_0p1=sum(row['within_0p1'] for row in scored),
                   predicted_on_or_below_reference_hull=sum(row['on_or_below_reference_hull'] for row in scored),
                   predicted_joint_unique_training_novel_within_0p1=len(joint),
                   diagnostic_joint_yield_per_original_proposal=len(joint)/len(original),
                   mean_predicted_above_reference_hull_eV_atom=(float(np.mean([row['energy_above_reference_hull_eV_atom'] for row in scored])) if scored else None),
                   elapsed_seconds=time.monotonic()-start, reference_formulas=len(hull.references),
                   omitted_nonfinite_reference_rows=hull.skipped, record_verified=False,
                   qualification='Single-model fixed-geometry screening; not DFT or full external benchmark')
    atomic_text(output/'summary.json', json.dumps(summary, indent=2))
    lines = ['# Stability screening — pilot-001', '', summary['qualification'], '',
             f"Assessed: {len(scored)}/{len(candidates)} converged endpoints.",
             f"Predicted within 0.1 eV/atom: {summary['predicted_within_0p1']}.",
             f"Also unique and unmatched against MP-20 train: {len(joint)}/{len(original)} original proposals.",
             '', 'No record verified. No DFT, independent-model ensemble, full novelty corpus, or medium-model re-relaxation.', '',
             '| Proposal | Formula | Above reference hull (eV/atom) | Max force (eV/A) | Training novel |',
             '| --- | --- | ---: | ---: | --- |']
    for row in sorted(scored, key=lambda r: r['energy_above_reference_hull_eV_atom']):
        lines.append(f"| {row['proposal_id']} | {row['formula']} | {row['energy_above_reference_hull_eV_atom']:.4f} | {row['max_force_eV_A']:.3f} | {row['novel']} |")
    atomic_text(output/'report.md', '\n'.join(lines)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
