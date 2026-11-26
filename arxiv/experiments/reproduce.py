#!/usr/bin/env python3
"""Inspect or execute the paper's recorded experiment recipes, serially."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('group', nargs='?', default='list', help='list, all, or a group shown by list')
    parser.add_argument('--only', help='select a single job ID')
    parser.add_argument('--execute', action='store_true', help='run commands; otherwise print the plan')
    parser.add_argument('--resume', action='store_true', help='skip commands successfully completed by this runner')
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error('--threads must be positive')
    catalog = json.loads((ROOT/'configs/experiments.json').read_text())
    if args.group == 'list':
        for group in catalog['group_order']:
            jobs = [j for j in catalog['jobs'] if j['group'] == group]
            print(f'{group:20s} {len(jobs):4d} steps')
        return 0
    if args.group not in ['all', *catalog['group_order']]:
        parser.error('unknown experiment group')
    jobs = [j for j in catalog['jobs'] if (args.group == 'all' or j['group'] == args.group)
            and (args.only is None or j['id'] == args.only)]
    if not jobs:
        parser.error('no matching jobs')
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
        env[key] = str(args.threads)
    logdir = ROOT/'build/reproduction'
    if args.execute:
        logdir.mkdir(parents=True, exist_ok=True)
    statefile = logdir/'completed.json'
    completed = json.loads(statefile.read_text()) if statefile.exists() else {}
    for job in jobs:
        command = [sys.executable, *job['command']]
        # Keep numerical choices unchanged; cap only the CPU worker count.
        if '--threads' in command:
            command[command.index('--threads')+1] = str(args.threads)
        print(f"\n[{job['group']}] {job['id']}\n{shlex.join(command)}", flush=True)
        if not args.execute or (args.resume and completed.get(job['id'], {}).get('command') == job['command']):
            continue
        jobenv = {**env, **job.get('env', {})}
        name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in job['id'])[:180]
        with (logdir/(name+'.log')).open('w') as log:
            result = subprocess.run(command, cwd=ROOT, env=jobenv, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print(f'Failed; see {logdir/(name+".log")}. No later command was started.', file=sys.stderr)
            return result.returncode
        completed[job['id']] = {'command': job['command'], 'finished': time.time()}
        temp = statefile.with_suffix('.tmp');temp.write_text(json.dumps(completed, indent=2)+'\n');temp.replace(statefile)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
