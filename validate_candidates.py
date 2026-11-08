"""Bounded, resumable matched-model relaxation and training-novelty recheck.

Preserves pilot outcomes. This is a local validation protocol, not DFT or a full
LeMat benchmark. Stop-Pilot.cmd stops it through the original run's STOP file.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import fcntl
import html
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import time
import traceback

from report_io import atomic_text
from stability_screen import TabulatedHull, HULL_REVISION, BENCHMARK_REVISION, sha256


class StopRequested(Exception):
    pass


def joint_pass(row):
    return (row.get('status') == 'validated' and row.get('converged') is True
            and row.get('within_0p1') is True and row.get('unique') is True
            and row.get('novel') is True)


def build_summary(results, original_count, planned, elapsed, cpu_seconds, status):
    valid = [r for r in results if r['status'] == 'validated']
    joint = [r for r in valid if joint_pass(r)]
    return dict(status=status, planned=planned, completed=len(results),
                validated=len(valid), failed=len(results)-len(valid),
                unresolved=max(0, planned-len(results)), original_proposals=original_count,
                predicted_within_0p1=sum(r['within_0p1'] for r in valid),
                recomputed_unique=sum(r.get('unique') is True for r in valid),
                recomputed_training_novel=sum(r.get('novel') is True for r in valid),
                predicted_joint_training_novel_count=len(joint),
                diagnostic_joint_yield=(len(joint)/original_count if len(results) == planned else None),
                joint_yield_bounds=[len(joint)/original_count,
                                   (len(joint)+planned-len(results))/original_count],
                lead_ids=[r['proposal_id'] for r in joint],
                elapsed_this_session_seconds=elapsed, cpu_this_session_seconds=cpu_seconds,
                reasons=dict(Counter(r.get('reason', r['status']) for r in results)),
                record_verified=False,
                qualification='Single MACE checkpoint; matched hull; MP-20 training novelty only; no DFT')


class Report:
    def __init__(self, root, original_count, planned, results, started, cpu_started):
        self.root, self.original_count, self.planned, self.results = root, original_count, planned, results
        self.started, self.cpu_started, self.last = started, cpu_started, 0

    def update(self, stage, proposal=None, step=None, force=False):
        if not force and time.monotonic()-self.last < 5:
            return
        self.last = time.monotonic()
        summary = build_summary(self.results, self.original_count, self.planned,
                                time.monotonic()-self.started, time.process_time()-self.cpu_started, stage)
        progress = dict(summary, active_proposal=proposal, optimizer_step=step,
                        updated_utc=datetime.now(timezone.utc).isoformat())
        atomic_text(self.root/'progress.json', json.dumps(progress, indent=2), required=False)
        message = (f'{len(self.results)}/{self.planned} complete | {stage} | proposal={proposal} | step={step}\n'
                   f'Validated: {summary["validated"]} | within 0.1 eV/atom: {summary["predicted_within_0p1"]} | '
                   f'unique + training-novel + within threshold: {summary["predicted_joint_training_novel_count"]}\n'
                   f'Elapsed this session: {summary["elapsed_this_session_seconds"]:.1f}s | '
                   f'Updated: {progress["updated_utc"]}\nNo DFT or external benchmark record verified.\n')
        atomic_text(self.root/'progress.txt', message, required=False)
        page = f'''<!doctype html><html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>Matched-model validation</title><style>body{{background:#101923;color:#e4edf7;font:17px system-ui;max-width:1000px;margin:40px auto;padding:20px}}
progress{{width:100%;height:28px}}pre{{white-space:pre-wrap;background:#1c2b3b;padding:22px;border-radius:12px}}a{{color:#77e1be}}</style></head>
<body><h1>Matched-model validation</h1><progress value="{len(self.results)}" max="{self.planned}"></progress>
<pre>{html.escape(message)}</pre><p>Relaxing endpoints with MACE-MP-0b3-medium, checking the compatible reference hull,
and recomputing uniqueness and novelty against MP-20 training structures.</p>
<p>Four CPU threads · 15-minute session budget · 120-second candidate budget · stop control: Stop-Pilot.cmd</p>
<p>Partial counts are provisional. Later candidates may merge into earlier endpoints. Final yield counts all
100 original proposals, including early rejections and failed relaxation.</p>
<p><a href="summary.json">Summary</a> · <a href="report.md">Final report</a> · <a href="manifest.json">Frozen protocol</a></p></body></html>'''
        atomic_text(self.root/'progress.html', page, required=False)
        print(message, flush=True)
        return summary


def compare_and_match(relaxed, accepted, references, matcher):
    key = relaxed.composition.reduced_formula
    unique = not any(matcher.fit(relaxed, other) for other in accepted
                     if other.composition.reduced_formula == key)
    novel = not any(matcher.fit(relaxed, other) for other in references)
    return bool(unique), bool(novel)


def run(args):
    started, cpu_started = time.monotonic(), time.process_time()
    run_root, root = Path(args.run), Path(args.output)
    root.mkdir(parents=True, exist_ok=args.resume)
    lock = (root/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs_path = run_root/'telemetry/events.jsonl'
    original = [json.loads(line) for line in inputs_path.read_text().splitlines()]
    candidates = [row for row in original if row.get('converged') and row.get('relaxed_cif')]
    source_paths = ['validate_candidates.py', 'stability_screen.py', 'evaluator_and_hull.py', 'pilot.py', 'report_io.py']
    config = vars(args).copy()
    config.pop('resume')
    config.update(model_sha256=sha256(args.model), hull_sha256=sha256(args.hull),
                  training_sha256=sha256(args.data), input_events_sha256=sha256(inputs_path),
                  input_cif_sha256={r['proposal_id']: sha256(r['relaxed_cif']) for r in candidates},
                  source_sha256={p: sha256(p) for p in source_paths},
                  matcher=dict(ltol=.2, stol=.3, angle_tol=5, primitive_cell=True, scale=True))
    if args.resume:
        previous = json.loads((root/'manifest.json').read_text())
        if previous['configuration'] != config:
            raise ValueError('Resume requires identical code, inputs and configuration')
    else:
        (root/'source').mkdir()
        for name in source_paths:
            shutil.copyfile(name, root/'source'/name)
        manifest = dict(configuration=config, model='MACE-MP-0b3-medium', hull_revision=HULL_REVISION,
                        matching_model_evidence_revision=BENCHMARK_REVISION,
                        packages={p: importlib.metadata.version(p) for p in ['torch', 'mace-torch', 'ase', 'pymatgen', 'scipy']},
                        original_proposals=len(original), candidates=len(candidates),
                        geometry_protocol='Unconstrained BFGS cell+position relaxation; no added rattle; fmax 0.03',
                        selection='All 37 originally converged endpoints, not just the four leads',
                        failure_policy='Failures remain in denominator; original 63 failures/rejections retained',
                        created_utc=datetime.now(timezone.utc).isoformat(), record_verified=False)
        atomic_text(root/'manifest.json', json.dumps(manifest, indent=2))
    ledger = root/'events.jsonl'
    results = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    if len({r['proposal_id'] for r in results}) != len(results):
        raise ValueError('Duplicate outcomes in validation ledger')
    seen = {r['proposal_id'] for r in results}
    report = Report(root, len(original), len(candidates), results, started, cpu_started)
    report.update('Loading matched model and reference data', force=True)
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    import pandas as pd
    import numpy as np
    from pymatgen.core import Composition, Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from evaluator_and_hull import MaceEvaluator
    from pilot import deadline
    hull = TabulatedHull(pd.read_parquet(args.hull, columns=['immutable_id', 'nsites', 'species_at_sites', 'mace_mp_energy']).to_dict('records'))
    evaluator = MaceEvaluator(args.model, steps=args.steps, fmax=args.fmax)
    matcher = StructureMatcher(ltol=.2, stol=.3, angle_tol=5)
    cifs, training_cache = defaultdict(list), {}
    with open(args.data, newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            cifs[Composition(row['pretty_formula']).reduced_formula].append(row['cif'])
    accepted = [Structure.from_file(r['relaxed_cif']) for r in results if r['status'] == 'validated']
    (root/'structures').mkdir(exist_ok=True)
    status = 'COMPLETE'
    def check_stop():
        if (root/'STOP').exists() or (run_root/'STOP').exists():
            raise StopRequested('STOP file requested')
    try:
        for candidate in candidates:
            key = candidate['proposal_id']
            if key in seen:
                continue
            remaining = args.budget_seconds-(time.monotonic()-started)
            if remaining <= 1:
                status = 'STOPPED: session budget exhausted'
                break
            try:
                check_stop()
            except StopRequested:
                status = 'STOPPED: user request'
                break
            result = dict(proposal_id=key, formula=candidate['formula'])
            phase, begin = 'relaxation', time.monotonic()
            report.update('Relaxing with matched MACE model', key, 0, force=True)
            def on_step(step):
                check_stop()
                report.update('Relaxing with matched MACE model', key, step)
            try:
                with deadline(min(args.candidate_seconds, remaining)):
                    initial = Structure.from_file(candidate['relaxed_cif'])
                    base = root/'structures'/key
                    relaxed, energy, converged, seconds = evaluator.relax(initial, base, on_step=on_step)
                    result.update(converged=converged, energy_per_atom=energy, relaxation_seconds=seconds,
                                  optimizer_steps=evaluator.last_steps, relaxed_cif=str(base)+'.cif')
                    if not converged:
                        result.update(status='failed', reason='NOT_CONVERGED_WITHIN_STEP_BUDGET')
                    else:
                        phase = 'assessment'
                        report.update('Rechecking hull, uniqueness and training novelty', key, force=True)
                        atoms = AseAtomsAdaptor.get_atoms(relaxed)
                        atoms.calc = evaluator.calculator
                        forces, stress = atoms.get_forces(), atoms.get_stress()
                        if not np.isfinite(forces).all() or not np.isfinite(stress).all():
                            raise ValueError('Nonfinite force/stress after relaxation')
                        formula = relaxed.composition.reduced_formula
                        if formula not in training_cache:
                            training_cache[formula] = [Structure.from_str(text, fmt='cif') for text in cifs[formula]]
                        unique, novel = compare_and_match(relaxed, accepted, training_cache[formula], matcher)
                        result.update(hull.evaluate(relaxed.composition, energy))
                        result.update(unique=unique, novel=novel, status='validated',
                                      max_force_eV_A=float(np.linalg.norm(forces, axis=1).max()),
                                      max_abs_stress_eV_A3=float(np.abs(stress).max()),
                                      original_unique=candidate.get('unique'), original_novel=candidate.get('novel'))
                        accepted.append(relaxed)
            except (StopRequested, KeyboardInterrupt):
                status = 'STOPPED: resumable interruption; current candidate remains pending'
                with (root/'attempts.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(proposal_id=key, status='interrupted', seconds=time.monotonic()-begin))+'\n')
                break
            except Exception as error:
                if isinstance(error, TimeoutError) and time.monotonic()-started >= args.budget_seconds:
                    status = 'STOPPED: session budget; current candidate remains pending'
                    break
                result.update(status='failed', reason=f'{phase}:{type(error).__name__}:{error}',
                              attempted_seconds=time.monotonic()-begin)
                (root/'structures'/f'{key}.error.txt').write_text(traceback.format_exc())
            result['completed_utc'] = datetime.now(timezone.utc).isoformat()
            with ledger.open('a') as stream:
                stream.write(json.dumps(result, allow_nan=False)+'\n')
            results.append(result)
            seen.add(key)
            report.update('Candidate completed', key, force=True)
    except BaseException:
        status = 'FAILED: see validation console/error log; completed outcomes are resumable'
        raise
    finally:
        summary = report.update(status, force=True)
        atomic_text(root/'summary.json', json.dumps(summary, indent=2))
        with (root/'sessions.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(status=status, elapsed_seconds=time.monotonic()-started,
                                        cpu_seconds=time.process_time()-cpu_started))+'\n')
        lines = ['# Matched-model validation', '', summary['qualification'], '',
                 f"Status: {status}. Validated {summary['validated']}/{len(candidates)} endpoints.",
                 f"Predicted within 0.1 eV/atom: {summary['predicted_within_0p1']}.",
                 f"Also recomputed unique and MP-20-training novel: {summary['predicted_joint_training_novel_count']}/{len(original)} original proposals.",
                 '', 'No external benchmark or DFT record established.', '',
                 '| ID | Formula | Above hull (eV/atom) | Max force (eV/A) | Unique | Training novel |',
                 '| --- | --- | ---: | ---: | --- | --- |']
        for row in sorted((r for r in results if r['status'] == 'validated'), key=lambda r: r['energy_above_reference_hull_eV_atom']):
            lines.append(f"| {row['proposal_id']} | {row['formula']} | {row['energy_above_reference_hull_eV_atom']:.4f} | {row['max_force_eV_A']:.4f} | {row['unique']} | {row['novel']} |")
        atomic_text(root/'report.md', '\n'.join(lines)+'\n')
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='runs/pilot-001')
    parser.add_argument('--output', default='runs/pilot-001/validation')
    parser.add_argument('--data', default='data/mp20-train.csv')
    parser.add_argument('--model', default='models/mace-mp-0b3-medium.model')
    parser.add_argument('--hull', default='data/lemat-mace-mp-hull.parquet')
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--fmax', type=float, default=.03)
    parser.add_argument('--candidate-seconds', type=float, default=120)
    parser.add_argument('--budget-seconds', type=float, default=900)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not all(math.isfinite(x) and x > 0 for x in [args.steps, args.fmax, args.candidate_seconds, args.budget_seconds]):
        parser.error('Budgets and convergence thresholds must be positive and finite')
    run(args)


if __name__ == '__main__':
    main()
