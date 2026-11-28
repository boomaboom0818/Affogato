"""Verify the frozen 004 trial and append only the remaining opening/color games."""
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from arena import ROOT
from openings import symmetry_key
from report import report

RUN = Path('.runtime/experiment-004')
CANDIDATE = RUN / 'affogato-004/model.bin.gz'
BASELINE = Path('.runtime/model.bin.gz')


def verify_model(path, digest):
    with gzip.open(path, 'rb') as model:
        if hashlib.file_digest(model, 'sha256').hexdigest() != digest:
            raise ValueError(f'Digest mismatch for {path}')


def read_records(path):
    data = path.read_bytes() if path.exists() else b''
    lines = data.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        tail = lines[-1]
        try:
            json.loads(tail)
        except (ValueError, UnicodeError):
            # A power loss can tear the last append. Save its exact bytes first.
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + '.torn-', delete=False) as backup:
                backup.write(tail)
                backup.flush()
                os.fsync(backup.fileno())
            with path.open('rb+') as stream:
                stream.truncate(len(data) - len(tail))
            lines.pop()
        else:
            with path.open('ab') as stream:
                stream.write(b'\n')
    return [json.loads(line) for line in lines]


def validate(records, book, visits, candidate=CANDIDATE):
    if len(records) > 2 * len(book):
        raise ValueError('Excess games')
    for i, g in enumerate(records):
        opening = book[i // 2]
        moves = g['moves']
        if not (Path(g['candidate']).resolve() == candidate.resolve()
                and Path(g['opponent']).resolve() == BASELINE.resolve()
                and g['candidate_color'] == 'BW'[i % 2] and g['opening'] == opening
                and g['visits'] == visits and g['komi'] == 7.5
                and g['move_seconds'] is None and g['game_seconds'] is None
                and [m[1] for m in moves[:len(opening)]] == opening
                and all(m[0] == 'BW'[j % 2] for j, m in enumerate(moves))
                and [m[1].lower() for m in moves[-2:]] == ['pass', 'pass']
                and re.fullmatch(r'[BW]\+\d+(\.\d+)?', g['result'])):
            raise ValueError(f'Unexpected game {i+1} at {visits} visits')


def play_trial(run, prefix, candidate, openings, book, visits):
    path = run / (prefix + 'games.jsonl')
    records = read_records(path)
    validate(records, book, visits, candidate)
    if len(records) < 2 * len(book):
        print(f'Resuming {visits} visits after {len(records)} games', flush=True)
        with path.open('ab') as out, (run / (prefix + 'arena.log')).open('ab') as err:
            subprocess.run([sys.executable, 'training/arena.py', str(candidate), '--openings',
                            str(openings), '--pairs', str(len(book)), '--visits', str(visits),
                            '--skip-games', str(len(records))], stdout=out, stderr=err, check=True)
        validate(read_records(path), book, visits, candidate)
    summary = report(path, len(book))
    if not summary['complete']:
        raise RuntimeError(f'Incomplete {visits}-visit arena')
    dest = run / (prefix + 'result.json')
    tmp = dest.with_suffix('.tmp')
    tmp.write_text(json.dumps(summary, indent=2) + '\n')
    tmp.replace(dest)
    print(visits, summary, flush=True)
    return summary


def main():
    os.chdir(ROOT)
    with (RUN / 'arena.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = json.loads(Path('models/affogato-004/manifest.json').read_text())
        verify_model(CANDIDATE, manifest['model_sha256'])
        verify_model(BASELINE, manifest['base_sha256'])
        seen = set()
        for prefix, visits in [('', 1024), ('replication-', 4096)]:
            openings = RUN / (prefix + 'openings.json')
            book = json.loads(openings.read_text())
            keys = {symmetry_key(moves) for moves in book}
            if len(book) != 64 or len(keys) != 64 or keys & seen:
                raise ValueError(f'Invalid or overlapping book: {openings}')
            seen.update(keys)
            play_trial(RUN, prefix, CANDIDATE, openings, book, visits)


if __name__ == '__main__':
    main()
