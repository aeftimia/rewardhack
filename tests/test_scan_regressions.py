"""Absent evidence must not become an ordinary successful measurement."""
import json
from types import SimpleNamespace

import pytest

from rewardhack import budget, export_transcripts, hf_grpo, rates, reporting, rescore


def test_no_exposure_is_no_measurement():
    assert rates.observed_ratio(0, 0) is None
    assert rates.observed_ratio(6, 2) == 3
    for total, exposure in [(1, 0), (0, -1), (float('nan'), 1), (1, float('inf'))]:
        with pytest.raises(ValueError):
            rates.observed_ratio(total, exposure)


def test_missing_ledger_cannot_mean_no_spend(tmp_path):
    with pytest.raises(FileNotFoundError):
        budget.read_ledger(tmp_path/'missing.csv')


def test_missing_approved_spend_is_not_zero(tmp_path):
    ledger = tmp_path/'ledger.csv'
    budget.reservation_path(ledger).write_text(json.dumps([
        dict(key='stage', line='storage', total_usd=10, status='pilot worst-case reservation')]))
    with pytest.raises(ValueError, match='reservation registry'):
        budget.read_reservations(ledger)


@pytest.mark.parametrize('template', [None, '', {}, False])
def test_missing_chat_template_stops_before_generation(template):
    with pytest.raises(ValueError, match='chat template'):
        hf_grpo.settle_template(SimpleNamespace(chat_template=template), None, 'unsupported')


@pytest.mark.parametrize('row', [
    {'termination_schema': 'other', 'truncated': False},
    {'termination_schema': 'cutoff-v1', 'truncated': 0},
    {'termination_schema': 'cutoff-v1', 'truncated': None}])
def test_cutoff_schema_cannot_silently_be_interpreted_as_complete(row):
    with pytest.raises(ValueError):
        rescore.scored_as_cut_off(row)


def test_missing_reward_penalty_stops_rescore(tmp_path):
    path = tmp_path/'rows.jsonl'
    path.write_text(json.dumps(dict(task_id='t', raw='{}', arm='report_correct', report_mode='typed'))+'\n')
    with pytest.raises(KeyError, match='penalty'):
        rescore.rescore_file(path, 'unused', {'t': {}})
    assert not path.with_suffix('.rescored.jsonl').exists()


def test_missing_report_is_distinct_from_malformed_report():
    assert reporting.report_text(None) == ''
    for value in [0, False, [], {}]:
        with pytest.raises(TypeError):
            reporting.report_text(value)
    with pytest.raises(ValueError):
        reporting.stored_mode({'report_mode': None})


def test_broken_git_provenance_is_not_reported_as_no_repository(tmp_path):
    import subprocess
    assert export_transcripts.git_commit(tmp_path) is None
    (tmp_path/'.git').mkdir()
    with pytest.raises(subprocess.CalledProcessError):
        export_transcripts.git_commit(tmp_path)


def test_missing_conversation_turn_is_not_promoted_to_first_turn():
    assert export_transcripts.turn_number({}) == 1
    assert export_transcripts.turn_number({'conversation': 'c', 'turn': 2}) == 2
    with pytest.raises(KeyError):
        export_transcripts.turn_number({'conversation': 'c'})
    with pytest.raises(ValueError):
        export_transcripts.turn_number({'turn': 2})


def test_unknown_export_stream_is_rejected():
    with pytest.raises(KeyError):
        export_transcripts.update_count({}, {}, {}, 'unknown')
    with pytest.raises(ValueError):
        export_transcripts.update_count({}, {}, {}, ('unknown', 1))


def test_absent_summary_metadata_is_reported_not_fabricated():
    meta = export_transcripts.run_meta({'variant': 'neutral'}, 'standalone')
    assert meta['missing_summary_metadata'] == ['argv', 'template_kwargs']
    assert meta['thinking'] is None and meta['model'] is None
    for name, value in [('argv', {}), ('template_kwargs', [])]:
        with pytest.raises(TypeError):
            export_transcripts.run_meta({name: value, 'variant': 'neutral'}, 'standalone')
    with pytest.raises(ValueError, match='no recorded prompt variant'):
        export_transcripts.run_meta({}, 'standalone')


def test_probe_requires_explicit_claim_state():
    from rewardhack.probe_labels import probe_labels
    row = dict(format_valid=True, verifier_inconclusive=False, verified_refutation=False)
    with pytest.raises(KeyError, match='claim'):
        probe_labels(row)
    with pytest.raises(ValueError, match='unknown report claim'):
        probe_labels(dict(row, claim='misspelled'))
    assert probe_labels(dict(row, claim=None))['report_error'] is None
