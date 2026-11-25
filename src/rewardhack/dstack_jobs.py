"""dstack task configurations for the pod jobs: a prototype beside `ops/drive_runpod.sh`, unproven until one paid run.

dstack deploys the RunPod pod, starts `ops/dstack_job.sh` in it, and removes the pod when the job's commands end or
`max_duration` runs out. Both of those are done by the dstack server calling RunPod's API, so they hold only while that
server runs (docs/notes-2026-11-21.md, section 62). The cap that does not depend on this machine is the one the bash
path already has: `ops/dstack_job.sh` starts `ops/remote_job.sh` with REQUIRE_AUTOSTOP=1, which arms the pod's own
removal with RunPod's CLI before the job starts and refuses to run if it cannot.

A launch is identified as for the bash driver, by remote script and deadline minutes, and must be covered by the launch
record that `entrypoints/gate.py ready` writes. The configuration is rendered from that record's settings, so the pod
gets exactly the settings that were checked.
"""
import json
import math
import re
from pathlib import Path

from .gates import DRIVER_GRACE_SECONDS, DRIVER_JUDGE_MAX_USD, RECORD, git, launch_problems

# What ops/drive_runpod.sh hands the pod when a setting is not given (its `${NAME:-value}` forms; a test reads them
# from the driver so the two cannot drift). Here every one is written into the task's env by name.
DRIVER_DEFAULTS = {'JUDGE_KIND': 'api', 'LEAN_ISOLATION': 'required', 'JUDGE_MAX_USD': f'{DRIVER_JUDGE_MAX_USD:g}',
                   'AUTOSTOP_GRACE_SECONDS': str(DRIVER_GRACE_SECONDS)}
SETTINGS = re.compile(r'SMOKE_[A-Z_]+|LEAN_ISOLATION|JUDGE_KIND|JUDGE_MAX_USD|FORWARD_JUDGE_KEY|AUTOSTOP_GRACE_SECONDS')
# remote_job.sh stops the job at the deadline with `timeout --kill-after=30` and then writes EXIT_CODE. dstack's own stop
# comes after that, so it never cuts the runner's record short: 30 s plus up to 90 s for the exit trap, rounded to 2 minutes.
AFTER_DEADLINE_MINUTES = 2
# dstack's stop_duration: once max_duration is reached it sends the job a stop and waits this long before killing it.
# remote_job.sh's TERM trap kills the job's group and writes two files, which takes seconds.
STOP_MINUTES = 2
# The dstack server terminates a job itself 2 minutes past max_duration if the runner on the pod did not
# (MAX_DURATION_ENFORCEMENT_GRACE in dstack's server/background/pipeline_tasks/jobs_running.py, read 2026-11-25).
SERVER_BACKSTOP_MINUTES = 2
# Billing before the job's first command, which max_duration does not count ("excluding provisioning and pulling") and
# only the dstack server bounds: pod boot measured at 1 to 1.5 min, plus dstack's runner download and the repo clone,
# not measured, so 3x an estimated 5 min. The same 15 minutes gates.py counts for the bash driver's provisional timer.
PROVISIONING_MINUTES = 15
GPU = 'RTXPRO6000:96GB:1'  # gpuhunt's name for RunPod's "RTX PRO 6000" (spaces dropped); confirm with `dstack offer` first
DISK = '150GB..'  # the smoke job's self-test wants 40 GB free under /workspace; model and judge weights are about 30 GB
RESULTS_PATH = '/results'

# The two jobs prototyped so far. `files` are copied beside the repo clone; dstack limits each entry to 2 MB unless the
# server's DSTACK_SERVER_CODE_UPLOAD_LIMIT is raised, and the benchmark's adapter is about 30 MB.
JOBS = {
    'smoke': dict(remote='ops/remote_runpod_smoke.sh', files=()),
    'genbench': dict(remote='ops/remote_runpod_genbench.sh', files=('payload/adapter',)),
}

# Where the results go. `volume`: a RunPod network volume that dstack mounts, which outlives the pod whoever removes it.
# `hold`: on the pod's own disk, kept for the grace after the job so they can be copied with `dstack attach`; gone with
# the pod if nobody copies them, as with the bash driver.
RESULTS = {
    'volume': lambda volume: dict(volumes=[dict(name=volume, path=RESULTS_PATH)]),
    'hold': lambda volume: {},
}


def driver_env(settings):
    """The environment the pod job gets from these driver settings, every value named. Raises ValueError on an unknown setting."""
    unknown = sorted(name for name in settings if not SETTINGS.fullmatch(name))
    if unknown:
        raise ValueError(f"settings a dstack launch does not pass to the pod: {', '.join(unknown)} (KEEP_POD has no meaning here)")
    return {**DRIVER_DEFAULTS, **settings}


def forwards_judge_key(env):
    """The API judge's key goes to the pod only on request, as with the bash driver."""
    return env['JUDGE_KIND'] == 'api' and 'FORWARD_JUDGE_KEY' in env and env['FORWARD_JUDGE_KEY'] == '1'


def hold_seconds(env, results):
    """How long the results wait on the pod after the job: the pod's own grace when they must be copied off it, else none."""
    return {'hold': int(env['AUTOSTOP_GRACE_SECONDS']), 'volume': 0}[results]


def max_duration_minutes(deadline, hold_seconds):
    """dstack's max_duration: the job deadline, the runner's time to stop it and record its exit, and the results hold if any."""
    return deadline + AFTER_DEADLINE_MINUTES + math.ceil(hold_seconds / 60)


def worst_case_minutes(deadline, grace_seconds, hold_seconds):
    """Billed minutes at worst: before the job, the server's bound; after it starts, the later of dstack's stop and the pod's own timer.

    The pod's timer (remote_job.sh) removes it the deadline plus 30 s plus the grace after the job starts, with or without the server.
    """
    by_server = max_duration_minutes(deadline, hold_seconds) + STOP_MINUTES + SERVER_BACKSTOP_MINUTES
    by_pod = deadline + (30 + grace_seconds) / 60
    return PROVISIONING_MINUTES + math.ceil(max(by_server, by_pod))


def task(root, job, deadline, settings, hourly, results, volume):
    """The dstack task configuration for one launch, as a dict. Raises ValueError for an unknown job or results mode."""
    if job not in JOBS:
        raise ValueError(f"no dstack job {job!r}; known: {', '.join(sorted(JOBS))}")
    if results not in RESULTS:
        raise ValueError(f"results must be one of {', '.join(sorted(RESULTS))}, not {results!r}")
    if (results == 'volume') != (volume is not None):
        raise ValueError('a network volume name is needed exactly when results go to a volume')
    spec = JOBS[job]
    env = driver_env(settings)
    hold = hold_seconds(env, results)
    head = git(root, 'rev-parse', 'HEAD').strip()
    config = dict(
        type='task',
        name=f'rh-{job}-{head[:7]}',
        python='3.12',
        repos=[dict(local_path='.')],
        files=[dict(local_path=str(Path(root)/path), path=path) for path in spec['files']],
        env=[f'{name}={value}' for name, value in sorted(env.items())]
            + [f'RESULTS_MODE={results}', f'RESULTS_HOLD_SECONDS={hold}']
            + (['OPENAI_API_KEY'] if forwards_judge_key(env) else []),  # no value: dstack takes it from the launcher's environment
        commands=[f"bash ops/dstack_job.sh {spec['remote']} {deadline * 60}"],
        backends=['runpod'],
        resources=dict(gpu=GPU, disk=DISK),
        spot_policy='on-demand',  # dstack refuses RunPod spot offers anyway ("Runpod spot offers are not supported")
        max_price=hourly,
        retry=False,  # a retry is a new pod, paid again, that nobody asked for
        max_duration=f'{max_duration_minutes(deadline, hold)}m',
        stop_duration=f'{STOP_MINUTES}m',
        **RESULTS[results](volume),
    )
    if not config['files']:
        del config['files']
    return config


def render(config, basis):
    """YAML text for `dstack apply -f -`: one `key: value` line per property, each value written as JSON (YAML's flow form)."""
    head = ['# Rendered by entrypoints/dstack_launch.py from the launch record; launch it only through that script.',
            f'# max_duration: the job deadline, sized as {basis}, plus {AFTER_DEADLINE_MINUTES} min for ops/remote_job.sh to stop the job',
            '# and record its exit, plus RESULTS_HOLD_SECONDS when the results wait on the pod to be copied.']
    return '\n'.join(head + [f'{key}: {json.dumps(value)}' for key, value in config.items()]) + '\n'


def pushed_problems(root):
    """dstack clones the repo from its remote and applies local changes up to 2 MB: the commit must be on a remote branch."""
    if not git(root, 'branch', '-r', '--contains', 'HEAD').strip():
        return ['HEAD is on no remote-tracking branch, and dstack clones the repo from its remote: push it first '
                '(judged from the remote branches as last fetched; nothing is fetched here)']
    return []


def configuration(root, job, deadline, settings, results, volume):
    """(task dict, YAML text) for this launch, with the price and cap basis taken from the launch record."""
    record = json.loads((Path(root)/RECORD).read_text())
    config = task(root, job, deadline, settings, record['hourly_usd'], results, volume)
    return config, render(config, record['basis'])


def launch_check(root, job, deadline, settings, results, volume, environ):
    """Why this dstack launch may not start: the launch record (as for the bash driver), the commit's reach, dstack's worst case
    against the one the budget check counted, and the judge key. Returns (problems, task dict, YAML text); the task is None with problems.
    """
    if job not in JOBS:
        return [f"no dstack job {job!r}; known: {', '.join(sorted(JOBS))}"], None, None
    problems = launch_problems(root, JOBS[job]['remote'], deadline, settings) + pushed_problems(root)
    if problems:
        return problems, None, None
    env = driver_env(settings)
    record = json.loads((Path(root)/RECORD).read_text())
    worst = worst_case_minutes(deadline, int(env['AUTOSTOP_GRACE_SECONDS']), hold_seconds(env, results))
    if worst > record['worst_case_minutes']:
        problems.append(f"under dstack this launch can bill {worst} min, and the budget check counted {record['worst_case_minutes']} "
                        '(gates.py counts the bash driver: its provisional timer plus the grace)')
    if forwards_judge_key(env) and ('OPENAI_API_KEY' not in environ or not environ['OPENAI_API_KEY']):
        problems.append('FORWARD_JUDGE_KEY=1 sends the judge key, and OPENAI_API_KEY is not set in this environment')
    if problems:
        return problems, None, None
    return [], *configuration(root, job, deadline, settings, results, volume)
