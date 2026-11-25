"""Free deterministic pilot fixtures; never imports a trainer or calls an API."""
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path

from . import jsonl
from .budget import FIELDS
from .pilot import SCHEMA, digest, load_spec

SCENARIOS = ('null', 'useful', 'delayed', 'harm', 'regression', 'measurement_failure')


def synthetic_rows(spec, checkpoint, scenario='delayed'):
    if scenario not in SCENARIOS:
        raise ValueError('unknown scenario')
    x = checkpoint/spec['checkpoints'][-1]
    rows = []
    for seed in spec['seeds']:
        rng = random.Random(f'{scenario}:{seed}')
        offset = rng.gauss(0, .025)
        benefit = {'null': 0, 'useful': .45*(1-math.exp(-5*x)),
                   'delayed': .6/(1+math.exp(-16*(x-.7))), 'harm': .4*x,
                   'regression': .5*math.sin(math.pi*x), 'measurement_failure': .2*x}[scenario]
        effects = {e['name']: dict(mean=max(-1, min(1, (benefit if i == 0 else -.35*x if scenario == 'harm' else 0)+offset)), se=.04)
                   for i, e in enumerate(spec['endpoints'])}
        rows.append(dict(spec_hash=digest(spec), seed=seed, checkpoint=checkpoint,
                         baseline_compute=checkpoint, intervention_compute=checkpoint, effects=effects,
                         quality=dict(valid=scenario != 'measurement_failure', invalid_rate=1. if scenario == 'measurement_failure' else 0., reason='synthetic fixture'),
                         sources=[digest(dict(synthetic=True, scenario=scenario, seed=seed, checkpoint=checkpoint))]))
    return rows


def write_stage(spec_path, checkpoint, scenario):
    spec = load_spec(spec_path)
    stage = next((s for s in spec['stages'] if s['checkpoint'] == checkpoint), None)
    if stage is None:
        raise ValueError('unplanned checkpoint')
    path = Path(stage['evidence'])
    with path.open('x') as handle:
        jsonl.write_rows(handle, synthetic_rows(spec, checkpoint, scenario), ensure_ascii=True)


def example_spec(directory):
    """Materialize an explicitly requested $0 local demo, refusing overwrite."""
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    spec_path = directory/'spec.json'
    ledger = directory/'synthetic-ledger.csv'
    with ledger.open('w') as handle:
        writer = csv.writer(handle); writer.writerow(FIELDS)
    script = Path(__file__).resolve().parents[2]/'entrypoints'/'pilot_simulate.py'
    spec = dict(schema=SCHEMA, id='free-delayed-learning-demo', question='Describe a synthetic benefit curve; no confirmatory margins selected.',
                context=dict(population='synthetic fixed cohort', compute_axis='synthetic compute units', report_mode='typed', baseline='synthetic baseline', intervention='synthetic intervention', evaluation_protocol='synthetic-v1'),
                seeds=['s1','s2','s3'], checkpoints=[1,2,4,8],
                endpoints=[dict(name='exploit_reduction', target=None, plateau_delta=.02),dict(name='proof_retention', target=None, plateau_delta=.02)],
                model=dict(prior_scales=[.25,.5], noise_floors=[.05,.1], seed_sds=[.15,.3], delayed_knees=[.5,.9]),
                stop=dict(confidence=.95, min_checkpoints=3, min_compute_fraction=.5, max_invalid_rate=.05),
                budget=dict(max_usd=0., max_seconds=60., line='api', ledger=str(ledger)),
                resource=dict(kind='local_process', note='Free synthetic Python only; no model, API, GPU, or paid resource.'))
    spec['stages'] = [dict(id=f'compute-{p}', checkpoint=p,
                           argv=[sys.executable,str(script),'--spec',str(spec_path),'--checkpoint',str(p),'--scenario','delayed'],
                           evidence=str(directory/f'evidence-{p}.jsonl'), max_seconds=10.,max_usd=0.,
                           inputs={str(script): hashlib.sha256(script.read_bytes()).hexdigest()}) for p in spec['checkpoints']]
    spec_path.write_text(json.dumps(spec, indent=2)+'\n')
    return spec_path
