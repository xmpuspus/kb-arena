#!/usr/bin/env python3
"""Print an evidence bundle the way a reader reads it.

Used by `docs/tapes/evidence.tape` so the recording shows real output rather
than a screenshot of prose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/run_422209dd/evidence.json")
    bundle = json.loads(path.read_text())
    print(f"citable:   {bundle['citable']}")
    if not bundle["citable"]:
        print(f"reason:    {bundle['why_not_citable']}")
    print(f"command:   {' '.join(bundle['command'])}")
    env = bundle["environment"]
    print(f"version:   {env['kb_arena']}  commit {str(env.get('git_sha'))[:12]}")
    print(f"seed:      {bundle['seed']}")
    counts = bundle["review"]["counts"]
    print(f"reviewed:  {counts['human-reviewed']} of {bundle['review']['questions']} questions")
    print(f"run set:   {bundle['question_set_fingerprint']}")
    print(f"review of: {bundle['review_question_set']}  split {bundle['review_split']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
