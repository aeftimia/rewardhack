import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rewardhack.tasks import select_tasks

script = Path(__file__).resolve().parents[1] / 'entrypoints/checkpoint_analysis.py'
spec = importlib.util.spec_from_file_location('checkpoint_analysis', script)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


@pytest.mark.parametrize('search,onset,selected', [
    (False, 17, None), (True, 17, 17), (True, 0, 0), (True, 60, None),
])
def test_checkpoint_targets(search, onset, selected):
    calls = []
    def evaluate(step, phase, draws):
        calls.append((step, phase, draws))
        return {'exploit': 100 * draws * int(step >= onset)}
    report = analysis.analyze(60, search, evaluate)
    assert report['targets'].get('plateau') == selected
    assert report['candidate_bracket'] == ([16, 17] if selected == 17 else None)
    assert [step for step, phase, draws in calls if phase == 'confirmation' and draws == 10] == list(dict.fromkeys(report['targets'].values()))
    assert bool(report['screen']) == search


def test_thousand_draws_real_scoring_and_records(tmp_path):
    train = select_tasks('hard', part='train')
    tasks = analysis.panel('hard', train)
    assert len(tasks) == len({r['statement'] for r in tasks}) == 100
    assert sum(row['true'] for row in tasks) == 50
    assert {r['statement'] for r in train}.isdisjoint(r['statement'] for r in tasks)
    seeds = []
    def sample(step, panel, seed):
        seeds.append(seed)
        # A model that hits the ceiling each time. The production scorer
        # handles this before invoking Lean.
        return [('cut off', True) for _ in panel]
    path = tmp_path/'confirmation-60.jsonl'
    counts = analysis.evaluate_checkpoint(tasks, 60, 'confirmation', 10, 2000, sample, 'lean', 'prose', path)
    assert counts['answers'] == counts['truncated'] == 1000
    assert counts['exploit'] == 0
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r['sampling_seed'] for r in rows] == list(range(2000, 3000))
    assert len({(r['task_id'], r['draw']) for r in rows}) == 1000
    assert seeds == list(range(2000, 3000, 100))


def test_incomplete_model_batch_is_rejected(tmp_path):
    tasks = analysis.panel('hard', select_tasks('hard', part='train'))
    with pytest.raises(ValueError, match='shorter'):
        analysis.evaluate_checkpoint(tasks, 0, 'screen', 1, 10, lambda *args: [], 'lean',
                                     'typed', tmp_path/'screen.jsonl')


def test_cli_two_parts_with_only_model_io_mocked(tmp_path, monkeypatch):
    lean = script.parents[1] / '.tools/lean-safe'
    if not lean.exists():
        pytest.skip('local sandboxed Lean installation required')
    segment = tmp_path/'segment'
    arm = segment/'train-baseline'
    arm.mkdir(parents=True)
    settings = dict(model='test', revision='pinned', taskset='hard', steps=4, reasoning_effort='low',
                    thinking='unsupported', prompt_variant='refute', report_mode='typed', seed=42)
    config = dict(run_id='fixture', parent=None, output_arm=arm.name, settings=settings)
    (segment/'run-config.json').write_text(json.dumps(config))
    (arm/'summary.json').write_text(json.dumps({'global_step': 4}))
    (arm/'tasks.json').write_text(json.dumps({'train': select_tasks('hard', part='train')}))
    adapters = {}
    for step in range(1, 5):
        path = arm/f'adapter-step{step}'/'adapter_model.safetensors'
        path.parent.mkdir()
        path.write_text(str(step))
        adapters[str(step)] = {str(path.relative_to(segment)): analysis.file_sha256(path)}
    (segment/'outputs.json').write_text(json.dumps(dict(backup_status='verified', checkpoints={'adapter': adapters})))

    class Tokenizer:
        chat_template = 'reasoning_effort'
        eos_token_id = 7

        def apply_chat_template(self, chat, **kwargs):
            assert len(chat) >= 2 and kwargs['reasoning_effort'] == 'low'
            return {'input_ids': [3]}

        def decode(self, tokens, **kwargs):
            return 'invalid typed answer'

    class Server:
        def __init__(self, args, stops):
            self.stops = stops
            self.name = 'fixture-adapter'
            self.checked_reload = False

        def preflight(self):
            self.sleeping = True

        def request(self, method, path):
            self.sleeping = {'/wake_up': False, '/sleep?level=1': True}[path]

        def require_sleep(self, value):
            assert self.sleeping == value

        def load(self, path):
            assert (path/'adapter_model.safetensors').is_file()

        def verify_reload(self, first):
            self.checked_reload = True

        def completions(self, prompts, *, seeds, max_tokens, model):
            assert not self.sleeping and len(prompts) == 100 and max_tokens == [8192] * 100 and len(seeds) == 100
            return [[7] for _ in prompts], [], None

        def close(self):
            assert self.sleeping

    out = tmp_path/'analysis'
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda model, revision: Tokenizer())))
    monkeypatch.setattr(analysis, 'AdapterRollouts', Server)
    monkeypatch.setattr(sys, 'argv', [str(script), '--segment', str(segment), '--lean', str(lean),
                                     '--out', str(out), '--rollout-url', 'http://127.0.0.1:8011', '--search'])
    analysis.main()
    report = analysis.read(out/'analysis.json')
    assert report['targets'] == {'base': 0, 'plateau': 0, 'final': 4}
    assert len(report['screen']) == 5
    assert [r['answers'] for r in report['confirmation'].values()] == [1000, 1000]
    assert '0' in report['final_minus_checkpoint']
