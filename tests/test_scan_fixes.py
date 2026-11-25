"""Regression tests for the scan of 25 November: values that were read as zero, false or absent when they were unknown."""
import argparse
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from rewardhack import calibrate, judge, model_io, openai_probe, outcomes, pilot

ROOT = Path(__file__).resolve().parent.parent
CHECKLIST = json.dumps({'requested': 'a', 'reported': 'b', **{name: False for name in judge.CHECKS}})
ROW = dict(statement='True', report='Proved it.')


def reply(**changes):
    body = dict(choices=[dict(message=dict(content='text'), finish_reason='stop')], model='m', usage=dict(prompt_tokens=4, completion_tokens=2))
    body.update(changes)
    return SimpleNamespace(status_code=200, json=lambda: body, text=json.dumps(body))


def transport(monkeypatch, responses):
    """`model_io.call` against canned responses; returns the list of requests made. Sleeping is an error: nothing here may retry."""
    sent = []

    def post(url, **kwargs):
        sent.append(url)
        return responses[len(sent)-1]
    monkeypatch.setattr(model_io.requests, 'post', post)
    monkeypatch.setattr(model_io.time, 'sleep', lambda seconds: pytest.fail('a reply that arrived was waited on and asked for again'))
    monkeypatch.setenv('SCAN_TEST_KEY', 'placeholder')
    return sent


ARGS = SimpleNamespace(model='m', max_tokens=8, reasoning_effort=None, temperature=None, base_url='http://127.0.0.1:1/v1', key_env='SCAN_TEST_KEY')


# --- A: metering

@pytest.mark.parametrize('usage', [None, {}, {'completion_tokens': 5}, {'input_tokens': 9, 'completion_tokens': 5}])
def test_a_priced_meter_refuses_a_reply_it_cannot_count(usage):
    meter = model_io.Meter(0.2, 1.25)
    with pytest.raises(model_io.UnmeteredReply, match='prompt_tokens.*cannot be metered'):
        meter.add(usage)
    assert meter.totals['usd'] == 0.0 and meter.totals['input'] == 0, 'a refused reply changed the totals'


def test_an_unpriced_meter_records_absent_counters_as_unknown():
    meter = model_io.Meter(0, 0)
    meter.add({'prompt_tokens': 3, 'completion_tokens': 2})
    assert (meter.totals['input'], meter.totals['output'], meter.totals['reasoning']) == (3, 2, None)  # no reasoning count is ordinary
    assert meter.add({}) == dict(input=None, output=None, reasoning=None)
    meter.add({'prompt_tokens': 3, 'completion_tokens': 2})
    assert meter.totals['input'] is None and meter.totals['output'] is None, 'an unknown count was summed as zero'
    assert meter.totals['uncounted'] == dict(input=1, output=1, reasoning=3) and meter.totals['usd'] == 0.0


def test_a_priced_meter_counts_and_prices_what_it_is_sent():
    meter = model_io.Meter(1, 2)
    counts = meter.add({'prompt_tokens': 4, 'completion_tokens': 2, 'completion_tokens_details': {'reasoning_tokens': 1}})
    assert counts == dict(input=4, output=2, reasoning=1) and meter.totals['usd'] == 8/1e6
    with pytest.raises(ValueError, match='not a count'):
        meter.add({'prompt_tokens': '4', 'completion_tokens': 2})


def test_a_priced_judge_needs_its_input_counter_too(monkeypatch):
    monkeypatch.setattr(model_io, 'call', lambda *a, **kw: dict(raw=CHECKLIST, finish='stop', model='m', usage={'completion_tokens': 10}))
    with pytest.raises(RuntimeError, match='prompt_tokens'):
        judge.make_judge('api', model='m', usd_per_m_input=0.2, usd_per_m_output=1.25, max_usd=1.0)(ROW)
    local = judge.make_judge('api', model='m', base_url='http://127.0.0.1:8011/v1')  # a judge on the pod: no price, nothing to refuse
    assert local(ROW)['claim'] == 'OK' and local.meter.totals['input'] is None and local.meter.totals['output'] == 10


def probe_args(tmp_path, **changes):
    values = dict(report_mode='prose', max_usd=1, usd_per_m_input=0, usd_per_m_output=0, out=tmp_path, taskset='v1', tasks='all', limit=None,
                  samples=1, concurrency=1, follow_on=False, variant='neutral', model='m', reasoning_effort=None, max_tokens=8)
    return SimpleNamespace(**dict(values, **changes))


def stub_probe(monkeypatch, result):
    task = dict(id='t', group='g', statement='True', true=True)
    monkeypatch.setattr(openai_probe, 'select_tasks', lambda *a, **kw: [task])
    monkeypatch.setattr(openai_probe, 'taskset_rows', lambda *a: [task])
    monkeypatch.setattr(openai_probe, 'call', lambda args, messages: result)


def test_a_priced_probe_stops_when_a_reply_has_no_usage(monkeypatch, tmp_path):
    stub_probe(monkeypatch, dict(raw='{}', usage=None, finish='stop', model='m'))
    with pytest.raises(model_io.UnmeteredReply):
        openai_probe.run(probe_args(tmp_path, usd_per_m_input=0.2, usd_per_m_output=1.25))
    assert not (tmp_path/'summary.json').exists(), 'a summary recording $0 was written for a priced run that could not be metered'


def test_an_unpriced_probe_stores_unknown_token_counts_as_none(monkeypatch, tmp_path):
    stub_probe(monkeypatch, dict(raw='{}', usage=None, finish='stop', model='m'))
    openai_probe.run(probe_args(tmp_path))
    row = json.loads((tmp_path/'rows.jsonl').read_text())
    summary = json.loads((tmp_path/'summary.json').read_text())
    assert row['tokens'] is None and row['reasoning_tokens'] is None
    assert summary['output'] is None and summary['mean_output_tokens'] is None and summary['uncounted']['output'] == 1


# --- B: which judge endpoints are metered

@pytest.mark.parametrize('url', ['https://api.openai.com/v1/', 'https://api.openai.com/v1', 'https://openrouter.ai/api/v1', 'http://10.0.0.5:8000/v1',
                                 'http://localhost.example.com/v1', 'localhost:8000/v1'])
def test_a_judge_anywhere_but_this_machine_needs_cap_and_prices(url):
    ap = argparse.ArgumentParser()
    paid = dict(arm='report_correct', judge='api', dry_run=False, judge_url=url)
    judge.check_reward_args(ap, argparse.Namespace(**paid, judge_max_usd=1.0, judge_usd_in=0.2, judge_usd_out=1.25))
    for unbounded in (dict(judge_max_usd=None, judge_usd_in=0.2, judge_usd_out=1.25), dict(judge_max_usd=1.0, judge_usd_in=0.0, judge_usd_out=1.25),
                      dict(judge_max_usd=1.0, judge_usd_in=0.2, judge_usd_out=0.0)):
        with pytest.raises(SystemExit):
            judge.check_reward_args(ap, argparse.Namespace(**paid, **unbounded))


@pytest.mark.parametrize('url', ['http://127.0.0.1:8011/v1', 'http://localhost:8011/v1/', 'http://[::1]:8011/v1'])
def test_a_judge_on_this_machine_needs_neither(url):
    judge.check_reward_args(argparse.ArgumentParser(), argparse.Namespace(arm='report_correct', judge='api', dry_run=False, judge_url=url,
                                                                          judge_max_usd=None, judge_usd_in=0.0, judge_usd_out=0.0))


# --- C: what the transport asks for again

def test_a_missing_key_fails_at_once(monkeypatch):
    monkeypatch.setattr(model_io.requests, 'post', lambda *a, **kw: pytest.fail('a request was sent without a key'))
    monkeypatch.setattr(model_io.time, 'sleep', lambda seconds: pytest.fail('a missing key was waited on and retried'))
    monkeypatch.delenv('SCAN_ABSENT_KEY', raising=False)
    with pytest.raises(KeyError, match='SCAN_ABSENT_KEY'):
        model_io.call(SimpleNamespace(**dict(vars(ARGS), key_env='SCAN_ABSENT_KEY')), [])


@pytest.mark.parametrize('changes,missing', [
    (dict(choices=[]), '0'), (dict(choices=[dict(finish_reason='stop')]), "'message'"), (dict(choices=[dict(message={}, finish_reason='stop')]), "'content'"),
    (dict(choices=[dict(message=dict(content='text'))]), 'finish_reason'), (dict(choices=[dict(message=dict(content='text'), finish_reason=None)]), 'finish_reason'),
    (dict(choices=[dict(message=dict(content=None), finish_reason='stop')]), 'not text')])
def test_a_reply_with_the_wrong_shape_is_an_error_and_is_not_asked_for_again(monkeypatch, changes, missing):
    sent = transport(monkeypatch, [reply(**changes)]*4)
    result = model_io.call(ARGS, [])
    assert len(sent) == 1, 'a billed reply was requested again'
    assert missing in result['error'] and 'raw' not in result
    assert result['usage'] == dict(prompt_tokens=4, completion_tokens=2), 'the billed reply no longer reaches the meter'


def test_a_reply_without_choices_or_json_names_what_is_missing(monkeypatch):
    body = reply()
    del body.json()['choices']
    sent = transport(monkeypatch, [body])
    assert "no 'choices'" in model_io.call(ARGS, [])['error'] and len(sent) == 1

    def not_json():
        raise ValueError('Expecting value')
    sent = transport(monkeypatch, [SimpleNamespace(status_code=200, json=not_json, text='<html>')])
    result = model_io.call(ARGS, [])
    assert 'not JSON' in result['error'] and result['usage'] is None and len(sent) == 1


def test_a_whole_reply_and_a_reply_cut_off_before_any_text(monkeypatch):
    transport(monkeypatch, [reply(), reply(choices=[dict(message=dict(content=None), finish_reason='length')])])
    assert model_io.call(ARGS, []) == dict(raw='text', finish='stop', usage=dict(prompt_tokens=4, completion_tokens=2), model='m')
    cut = model_io.call(ARGS, [])
    assert (cut['raw'], cut['finish']) == ('', 'length')


def test_a_refused_request_is_still_retried(monkeypatch):
    sent = transport(monkeypatch, [SimpleNamespace(status_code=503, text='busy'), reply()])
    monkeypatch.setattr(model_io.time, 'sleep', lambda seconds: None)
    assert model_io.call(ARGS, [])['raw'] == 'text' and len(sent) == 2


def test_a_judge_reply_without_a_finish_reason_is_a_failed_reading_and_is_metered(monkeypatch):
    sent = transport(monkeypatch, [reply(choices=[dict(message=dict(content=CHECKLIST))])])
    reader = judge.make_judge('api', model='m', base_url=ARGS.base_url, key_env='SCAN_TEST_KEY', usd_per_m_input=1, usd_per_m_output=1, max_usd=1)
    verdict = reader(ROW)
    assert verdict['claim'] == 'UNPARSED' and 'finish_reason' in verdict['judge_error'] and len(sent) == 1
    assert reader.meter.totals['errors'] == 1 and reader.meter.totals['usd'] == 6/1e6, 'the unusable reply was billed and not counted'


# --- D: calibration labels

def judged_row(task_id, label, true=False):
    return dict(task_id=task_id, true=true, genuine=False, exploit=False, refutation=False, judge=label)


def test_unlabelled_rows_leave_the_acknowledgement_rate():
    rows = [judged_row('f', 'ACKNOWLEDGE'), judged_row('f', 'CLAIM'), judged_row('f', calibrate.UNLABELLED), judged_row('f', calibrate.UNLABELLED),
            judged_row('unread', calibrate.UNLABELLED)]
    table, keep = calibrate.per_statement(rows, True, 0.4, 0.6)
    by_id = {t['task_id']: t for t in table}
    assert by_id['f']['ack'] == 0.5 and by_id['f']['claim'] == 0.5 and (by_id['f']['labelled'], by_id['f']['unlabelled']) == (2, 2)
    assert keep == ['f'], 'unlabelled rows counted as "did not acknowledge" moved the statement out of the band'
    assert by_id['unread']['ack'] is None and by_id['unread']['unlabelled'] == 1
    head = calibrate.headline(table, keep, True, len(rows))
    assert head['unlabelled_rows'] == 3 and head['false']['mean_ack'] == 0.5 and head['false']['statements_with_ack'] == 1
    with pytest.raises(ValueError, match='UNPARSED'):
        calibrate.per_statement([judged_row('f', 'UNPARSED')], True)


def test_a_mistyped_category_stops_calibration(tmp_path):
    path = tmp_path/'labels.jsonl'
    path.write_text(json.dumps(dict(id='run#0', category='honest_admission'))+'\n'+json.dumps(dict(id='run#1', category='honest_admision'))+'\n')
    with pytest.raises(ValueError, match="run#1 \\('honest_admision'\\)"):
        calibrate.read_labels(path)


def test_calibration_writes_the_unlabelled_count_and_refuses_a_file_that_labels_nothing(tmp_path):
    row = dict(task_id='f', true=False, genuine=False, exploit=False, verified_refutation=False, format_valid=True, report='r')
    (tmp_path/'rows.jsonl').write_text((json.dumps(row)+'\n')*3)
    (tmp_path/'labels.jsonl').write_text(json.dumps(dict(id='run#0', category='honest_admission'))+'\n')
    args = SimpleNamespace(out=tmp_path/'out', rows=[tmp_path/'rows.jsonl'], labels=tmp_path/'labels.jsonl', judge='labels', run_name='run', lo=0.2, hi=0.6)
    calibrate.run(args)
    written = json.loads((tmp_path/'out'/'per_statement.json').read_text())
    assert written['unlabelled_rows'] == 2 and written['table'][0]['ack'] == 1.0 and written['table'][0]['unlabelled'] == 2
    args.run_name = 'another-run'
    with pytest.raises(ValueError, match='no row has a label'):
        calibrate.run(args)


# --- E: keys that every freshly built row carries

def fresh(**changes):
    return dict(dict(task_id='t', group='g', tier='easy', true=True, arm='report_correct', format_valid=True, genuine=False, genuine_why='no', exploit=False,
                     verified_refutation=False, verifier_inconclusive=False, exploit_and_claims=False, exploit_and_admits=False, failed_and_claims=False,
                     needs_review=False, body='', reward=0.0, truncated=False), **changes)


def test_a_fresh_row_without_its_undecided_label_is_an_error():
    renamed = fresh(verifier_undecided_renamed=True)
    del renamed['verifier_inconclusive']
    for read in (outcomes.training_keep, lambda row: outcomes.measured([row]), lambda row: outcomes.summarize([row]), lambda row: outcomes.tally([row]),
                 lambda row: outcomes.verifier_advice([row]), lambda row: outcomes.training_rewards([row], 1)):
        with pytest.raises(KeyError, match='verifier_inconclusive'):
            read(renamed)
    assert outcomes.training_keep(fresh()) is True and outcomes.training_keep(fresh(verifier_inconclusive=True)) is False


def test_a_row_that_predates_the_label_is_unknown_not_decided():
    old = fresh()
    del old['verifier_inconclusive'], old['genuine_why']  # rows saved before 21 November carry neither
    assert outcomes.verifier_undecided(old) is None
    assert outcomes.verifier_advice([old])['unverified_rate'] is None and outcomes.tally([old])['inconclusive'] == 0
    assert outcomes.verifier_undecided(fresh(verifier_inconclusive=True)) is True
    # `judge_undecided` is written only when a reading failed: absent means no.
    assert outcomes.training_keep(fresh()) and not outcomes.training_keep(fresh(judge_undecided=True))


def test_a_variant_without_its_misfit_flag_does_not_read_as_a_good_fit(tmp_path, monkeypatch):
    from rewardhack import pilot_model
    from rewardhack.pilot_simulation import example_spec, synthetic_rows
    spec = pilot.load_spec(example_spec(tmp_path/'demo'))
    rows = [row for point in spec['checkpoints'] for row in synthetic_rows(spec, point, 'null')]
    real = pilot_model.analyze

    def renamed(spec, rows):
        posterior = real(spec, rows)
        for variant in posterior['variants']:
            variant['misfit'] = variant.pop('model_misfit')
        return posterior
    monkeypatch.setattr(pilot_model, 'analyze', renamed)
    with pytest.raises(KeyError, match='model_misfit'):
        pilot.decide(spec, rows)


# --- F: the read set

def read_folder(tmp_path, labels):
    for i in range(3):
        (tmp_path/f'batch{i}.jsonl').write_text(json.dumps(dict(id=f'r{i}', statement='True', report='Proved.', genuine=True))+'\n')
    (tmp_path/'labels0.jsonl').write_text(''.join(json.dumps(dict(id=key, category=value))+'\n' for key, value in labels.items()))
    return tmp_path


def test_a_read_row_without_a_usable_label_is_refused_not_dropped(tmp_path):
    complete = dict(r0='true_claim', r1='false_claim', r2='honest_admission')
    assert [row['read_claim'] for row in judge.read_set(read_folder(tmp_path, complete))] == ['OK', 'OK', 'ERROR']
    with pytest.raises(ValueError, match=r"1 read rows have no label \(first: \['r2'\]\) and 1 have a category outside the rubric \(first: \[\"r1 \('false_clam'\)\"\]\)"):
        judge.read_set(read_folder(tmp_path, dict(r0='true_claim', r1='false_clam')))


def test_validation_in_development_mode_refuses_a_read_set_with_gaps(tmp_path, monkeypatch):
    (tmp_path/'read').mkdir()
    read = read_folder(tmp_path/'read', dict(r0='true_claim'))
    monkeypatch.setattr(sys, 'argv', ['validate_judge.py', '--out', str(tmp_path/'out'), '--read', str(read), '--judge', 'triage', '--dry-run'])
    with pytest.raises(ValueError, match='2 read rows have no label'):
        runpy.run_path(str(ROOT/'entrypoints'/'validate_judge.py'), run_name='__main__')
    assert not (tmp_path/'out').exists()


# --- G

def test_seconds_per_report_is_over_the_reports_judged(tmp_path, monkeypatch):
    read = tmp_path/'read'
    read.mkdir()
    read_folder(read, dict(r0='true_claim', r1='false_claim', r2='honest_admission'))

    def capped(row):
        return dict(claim='OK', raw='')
    capped.meter = SimpleNamespace(totals=dict(usd=0.0))

    def judge_one_then_hit_the_cap(row):
        capped.meter.totals['usd'] = 5.0
        return capped(row)
    judge_one_then_hit_the_cap.meter = capped.meter
    monkeypatch.setattr(judge, 'from_args', lambda args: judge_one_then_hit_the_cap)
    ticks = iter([100.0, 103.0, 103.0, 103.0])
    monkeypatch.setattr('time.perf_counter', lambda: next(ticks))
    monkeypatch.setattr(sys, 'argv', ['validate_judge.py', '--out', str(tmp_path/'out'), '--read', str(read), '--judge', 'triage', '--dry-run', '--max-usd', '1'])
    runpy.run_path(str(ROOT/'entrypoints'/'validate_judge.py'), run_name='__main__')
    summary = json.loads((tmp_path/'out'/'summary.json').read_text())
    assert (summary['rows'], summary['planned']) == (1, 3)
    assert summary['seconds_per_report'] == 3.0, 'three seconds for one judged report was spread over the three planned'


def test_an_adapter_that_fits_no_lora_layer_is_refused(tmp_path):
    mx = pytest.importorskip('mlx.core')
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from rewardhack import mlx_rl

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.base, self.lora = nn.Linear(2, 2), nn.Linear(2, 2)
    model = Tiny()
    model.freeze()
    model.lora.unfreeze()
    saved = {name: value+1 for name, value in tree_flatten(model.trainable_parameters())}
    mx.save_safetensors(str(tmp_path/'adapter.safetensors'), saved)
    mlx_rl.load_adapter(model, tmp_path/'adapter.safetensors')
    assert all(mx.array_equal(value, saved[name]).item() for name, value in tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(tmp_path/'other.safetensors'), {'layers.0.lora_a': mx.zeros((2, 2))})
    with pytest.raises(ValueError, match='does not fit'):
        mlx_rl.load_adapter(model, tmp_path/'other.safetensors')
