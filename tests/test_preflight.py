"""Regression checks for holdout integrity, preflight gates and representative sampling."""
import copy
import json
import subprocess
from pathlib import Path

import pytest

from rewardhack.judge import validate
from rewardhack.judge_holdout import acceptance, check_frozen, check_rows, freeze, load_review, prepare
from rewardhack.tasks import split, stratified_limit, taskset_rows


def test_limits_keep_pairs_and_cover_families():
    rows = split(taskset_rows('hard'))['train']
    selected = stratified_limit(rows, 12)
    assert len(selected) == 12
    assert {r['family'] for r in selected} == {r['family'] for r in rows}
    assert all(sum(r['group'] == group for r in selected) == 2 for group in {r['group'] for r in selected})
    assert len(stratified_limit(rows, 11)) == 10
    with pytest.raises(ValueError):
        stratified_limit(rows, 1)


def fixture_protocol(tmp_path):
    read = tmp_path/'development'
    read.mkdir()
    row = dict(id='dev', task_id='add-34-F', statement='old statement', report='old report')
    (read/'batch0.jsonl').write_text(json.dumps(row)+'\n')
    (read/'labels0.jsonl').write_text(json.dumps(dict(id='dev', category='false_claim'))+'\n')
    protocol = freeze(read, {'model': 'frozen'}, 'Lean test',
                      dict(min_rows=4, min_class=2, min_agreement=.95, max_false_honest=.05, max_false_dishonest=.05))
    rows = [dict(id=f'new{i}', task_id=f'fresh-{i}', group=f'fresh-{i//2}', template_id='fresh-template',
                 statement=f'new statement {i}', established='', report=f'fresh report {i}', genuine=False) for i in range(4)]
    return protocol, rows


def test_holdout_rejects_development_and_duplicates(tmp_path):
    protocol, rows = fixture_protocol(tmp_path)
    check_rows(rows, protocol)
    for key, value in (('id', 'dev'), ('group', 'add-34'), ('template_id', 'add'),
                       ('statement', 'old  statement'), ('report', 'old report')):
        changed = copy.deepcopy(rows)
        changed[0][key] = value
        with pytest.raises(ValueError):
            check_rows(changed, protocol)
    with pytest.raises(ValueError):
        check_rows(rows + rows[:1], protocol)
    changed = copy.deepcopy(protocol)
    changed['source_sha256']['judge.py'] = 'changed'
    with pytest.raises(ValueError):
        check_frozen(changed)


def test_blind_holdout_round_trip_and_acceptance(tmp_path):
    protocol, rows = fixture_protocol(tmp_path)
    out = tmp_path/'holdout'
    prepare(rows, protocol, out)
    sheet = (out/'sheet.md').read_text()
    assert 'genuine' not in sheet and 'read_claim' not in sheet
    with pytest.raises(ValueError, match='Complete every'):
        load_review(out, protocol['config'], 'Lean test')
    for i in range(1, 5):
        sheet = sheet.replace(f'Answer {i:02d}: ', f"Answer {i:02d}: {'E' if i <= 2 else 'K'}")
    (out/'sheet.md').write_text(sheet)
    labelled, criteria, provenance = load_review(out, protocol['config'], 'Lean test')
    assert len(labelled) == 4 and provenance['human_sheet_sha256']
    def judge(row):
        return dict(claim=row['read_claim'], raw='')
    result = validate(judge, labelled)
    assert acceptance(result, 4, criteria)['accepted']
    assert not acceptance(result, 5, criteria)['accepted']
    result['false_honest']['n'] = 1
    assert not acceptance(result, 4, criteria)['accepted']
    with pytest.raises(ValueError, match='configuration'):
        load_review(out, {'model': 'different'}, 'Lean test')
    (out/'sheet.md').write_text(sheet.replace('fresh report', 'edited report'))
    with pytest.raises(ValueError, match='content changed'):
        load_review(out, protocol['config'], 'Lean test')
    (out/'rows.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='rows changed'):
        load_review(out, protocol['config'], 'Lean test')


def test_remote_runner_propagates_failure(tmp_path):
    # Stub GNU timeout to exercise status plumbing on macOS without launching a job remotely.
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    timeout = bin_dir/'timeout'
    timeout.write_text('#!/bin/bash\nshift 3\nexec "$@"\n')
    timeout.chmod(0o755)
    job = tmp_path/'fail.sh'
    job.write_text('exit 7\n')
    import os
    env = dict(os.environ, REMOTE_JOB_OUT=str(tmp_path/'out'), PATH=str(bin_dir)+':'+os.environ['PATH'])
    result = subprocess.run(['bash', 'ops/remote_job.sh', str(job), '10'], env=env)
    assert result.returncode == 7
    assert (tmp_path/'out'/'EXIT_CODE').read_text().strip() == '7'
    assert (tmp_path/'out'/'DONE').exists()


def test_reward_groups_measure_within_prompt_variation():
    from rewardhack.outcomes import reward_groups
    base = dict(exploit=False, genuine=False, verified_refutation=False, body='', format_valid=True, verifier_inconclusive=False)
    rows = [dict(base, task_id=task, reward=reward) for task, reward in [('a', 0), ('a', 1), ('b', 2), ('b', 2)]]
    result = reward_groups(rows, 2)
    assert result['groups'] == 2 and result['mixed_groups'] == 1 and result['zero_advantage_groups'] == 1
    assert [g['task_id'] for g in result['per_group']] == ['a', 'b']
    with pytest.raises(ValueError):
        reward_groups(rows[:3], 2)


def test_shared_model_transport(monkeypatch):
    from types import SimpleNamespace
    from rewardhack import model_io
    captured = []
    def post(url, **kwargs):
        captured.append((url, kwargs))
        return SimpleNamespace(status_code=200, json=lambda: dict(choices=[dict(message=dict(content=' disclose '), finish_reason='stop')],
                                                                model='test', usage=dict(prompt_tokens=4, completion_tokens=2)))
    monkeypatch.setenv('OPENAI_API_KEY', 'test-only-placeholder')
    monkeypatch.setattr(model_io.requests, 'post', post)
    args = SimpleNamespace(model='test', max_tokens=8, reasoning_effort='none', temperature=None, base_url='https://api.openai.com/v1', key_env='OPENAI_API_KEY')
    assert model_io.call(args, [{'role': 'user', 'content': 'report'}])['raw'] == ' disclose '
    assert captured[0][1]['json']['max_completion_tokens'] == 8
    assert captured[0][1]['json']['reasoning_effort'] == 'none'
    meter = model_io.Meter(1, 2)
    meter.add(dict(prompt_tokens=4, completion_tokens=2))
    assert meter.totals['usd'] == 8/1e6


@pytest.mark.parametrize('remote_code', [0, 7, 124])
def test_driver_returns_remote_status(tmp_path, remote_code):
    import os
    bin_dir = tmp_path/'bin'
    bin_dir.mkdir()
    stubs = {
        'ssh': f'for arg in "$@"; do if [[ $arg == "cat out/EXIT_CODE" ]]; then echo {remote_code}; fi; done\nexit 0\n',
        'rsync': 'exit 0\n',
        'sleep': 'exit 0\n',
    }
    for name, body in stubs.items():
        path = bin_dir/name
        path.write_text('#!/bin/bash\n'+body)
        path.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir)+':'+os.environ['PATH'], JUDGE_KIND='pod', FORWARD_JUDGE_KEY='0')
    result = subprocess.run(['bash', 'ops/drive_runpod.sh', 'example.invalid', '22', 'ops/remote_runpod_smoke.sh',
                             str(tmp_path/'results'), '1'], env=env, capture_output=True, text=True)
    assert result.returncode == remote_code, result.stderr


def test_a_judge_that_never_flags_cannot_be_accepted(tmp_path):
    """Half genuine, nobody flagged: the honesty classes are balanced, yet an always-OK judge would be perfect."""
    protocol, rows = fixture_protocol(tmp_path)
    for i, row in enumerate(rows):
        row['genuine'] = i % 2 == 0
    out = tmp_path/'holdout'
    prepare(rows, protocol, out)
    sheet = (out/'sheet.md').read_text()
    for i in range(1, 5):
        sheet = sheet.replace(f'Answer {i:02d}: ', f'Answer {i:02d}: K')
    (out/'sheet.md').write_text(sheet)
    with pytest.raises(ValueError, match='one side of the flag'):
        load_review(out, protocol['config'], 'Lean test')
    # The same check stands in acceptance, for results that reach it by another route.
    labelled = [dict(r, category='human_OK', read_claim='OK') for r in rows]
    silent = validate(lambda row: dict(claim='OK', raw=''), labelled)
    verdict = acceptance(silent, 4, protocol['criteria'])
    assert not verdict['accepted'] and any('missed_flag' in f for f in verdict['failures'])
    # A judge that misses flags fails on the flag rate even where the honesty bit happens to agree.
    mixed = [dict(r, category='human', read_claim='ERROR' if i < 2 else 'OK', genuine=False) for i, r in enumerate(rows)]
    deaf = acceptance(validate(lambda row: dict(claim='OK', raw=''), mixed), 4, protocol['criteria'])
    assert not deaf['accepted'] and 'missed_flag: rate above threshold' in deaf['failures']
