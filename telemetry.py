"""Single-writer, resumable terminal outcomes with incremental aggregation.

Record one immutable result per proposal_id, including failures. The scheduler
must track in-flight proposals separately. Call checkpoint() at shutdown.
"""
import json
import math
from collections import Counter
from pathlib import Path
from report_io import atomic_text


class Telemetry:
    STATUSES = {'evaluated', 'rejected', 'generation_failed', 'relaxation_failed',
                'missing_reference', 'evaluation_failed'}

    def __init__(self, directory, resume=False, refresh_every=25):
        if refresh_every < 1:
            raise ValueError('refresh_every must be positive')
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=resume)
        self.refresh_every = refresh_every
        self.seen, self.counts, self.reasons = set(), Counter(), Counter()
        self.relax_seconds = self.oracle_total_ms = self.oracle_max_ms = 0.0
        self.oracle_count = 0
        self.ledger = self.directory / 'events.jsonl'
        if resume and self.ledger.exists():
            # Corruption is an error, never an excuse to silently lose an outcome.
            with self.ledger.open(encoding='utf-8') as stream:
                for line in stream:
                    event = json.loads(line)
                    self._validate(event)
                    self._accumulate(event)

    def _validate(self, event):
        json.dumps(event, allow_nan=False)
        key = event.get('proposal_id')
        if not isinstance(key, str) or not key or key in self.seen:
            raise ValueError('proposal_id must be a new, nonempty string')
        if event.get('status') not in self.STATUSES:
            raise ValueError('Unknown terminal outcome status')
        if event['status'] != 'evaluated' and event.get('above_hull') is not None:
            raise ValueError('Only evaluated outcomes may carry above_hull')
        if 'reason' in event and not isinstance(event['reason'], str):
            raise ValueError('reason must be a string')
        for field in ('relax_seconds', 'oracle_ms', 'above_hull'):
            value = event.get(field)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or (field != 'above_hull' and value < 0)):
                raise ValueError(f'{field} must be finite; timings must be nonnegative')
        for field in ('relax_seconds', 'oracle_ms'):
            if field in event and event[field] is None:
                raise ValueError(f'{field} cannot be null')
        for field in ('novel', 'unique', 'converged'):
            if event.get(field) is not None and not isinstance(event[field], bool):
                raise ValueError(f'{field} must be boolean or null')

    def _accumulate(self, event):
        self.seen.add(event['proposal_id'])
        self.counts['proposed'] += 1
        self.counts[event['status']] += 1
        self.reasons[event.get('reason', event['status'])] += 1
        self.relax_seconds += event.get('relax_seconds', 0)
        self.counts['converged'] += event.get('converged') is True
        self.counts['unique_endpoints'] += event.get('converged') is True and event.get('unique') is True
        self.counts['novel_endpoints'] += event.get('converged') is True and event.get('novel') is True
        if 'oracle_ms' in event:
            self.oracle_count += 1
            self.oracle_total_ms += event['oracle_ms']
            self.oracle_max_ms = max(self.oracle_max_ms, event['oracle_ms'])
        assessed = (event['status'] == 'evaluated' and event.get('converged') is True
                    and event.get('above_hull') is not None)
        if assessed:
            self.counts['energy_assessed'] += 1
            self.counts['within_0p1'] += event['above_hull'] <= .1
        if assessed and event.get('unique') is not None and event.get('novel') is not None:
            self.counts['joint_assessed'] += 1
            joint = event['unique'] and event['novel']
            self.counts['un_at_0'] += joint and event['above_hull'] <= 0
            self.counts['un_at_0p1'] += joint and event['above_hull'] <= .1
        elif event['status'] not in {'rejected', 'generation_failed', 'relaxation_failed'}:
            # A known duplicate, known training match or known unstable endpoint
            # cannot be a joint discovery even if another assessment is missing.
            ruled_out = (event.get('unique') is False or event.get('novel') is False
                          or (assessed and event['above_hull'] > .1))
            self.counts['unresolved'] += not ruled_out

    def record(self, event):
        self._validate(event)
        encoded = json.dumps(event, allow_nan=False)
        with self.ledger.open('a', encoding='utf-8') as stream:
            stream.write(encoded + '\n')
        self._accumulate(event)
        if self.counts['proposed'] % self.refresh_every == 0:
            self.checkpoint()
            self.display()

    def summary(self):
        n = self.counts['proposed']
        unresolved = self.counts['unresolved']
        found = self.counts['un_at_0p1']
        complete = n > 0 and unresolved == 0
        return dict(
            counts=dict(self.counts), outcomes_recorded=n,
            early_rejection_rate=self.counts['rejected']/n if n else None,
            generation_failure_rate=self.counts['generation_failed']/n if n else None,
            un_yield_at_0=self.counts['un_at_0']/n if complete else None,
            un_yield_at_0p1=found/n if complete else None,
            un_at_0p1_yield_bounds=[found/n, (found+unresolved)/n] if n else None,
            survivor_within_0p1_rate=(self.counts['within_0p1']/self.counts['energy_assessed']
                                      if self.counts['energy_assessed'] else None),
            relaxation_seconds=self.relax_seconds,
            oracle_mean_ms=self.oracle_total_ms/self.oracle_count if self.oracle_count else None,
            oracle_max_ms=self.oracle_max_ms if self.oracle_count else None,
            reasons=dict(self.reasons), record_verified=False,
            status='LOCAL SCREENING; NOT A VERIFIED BENCHMARK RECORD')

    def checkpoint(self):
        atomic_text(self.directory / 'summary.json',
                    json.dumps(self.summary(), indent=2, allow_nan=False), required=False)

    def display(self):
        print('\n================ LIVE DISCOVERY SCOREBOARD ================')
        print(json.dumps(self.summary(), indent=2))
        print('===========================================================', flush=True)
