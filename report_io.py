"""Atomic reports with bounded retries for Windows/OneDrive reader locks."""
import sys
import time
from pathlib import Path


def atomic_text(path, content, *, required=True):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    for attempt in range(6):
        try:
            temporary.write_text(content, encoding='utf-8')
            temporary.replace(path)
            return True
        except OSError as error:
            if attempt == 5:
                if required:
                    raise
                print(f'Report temporarily unavailable: {path}: {error}', file=sys.stderr, flush=True)
                return False
            time.sleep(min(.02 * 2**attempt, .2))
