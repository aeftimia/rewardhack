"""Blind-audit completion, integrity, and fail-closed development exclusions."""
import json
import runpy
import sys
from pathlib import Path

import pytest

from rewardhack.formal_audit import audit_file
from rewardhack.judge_holdout import freeze, load_review, prepare

ROOT = Path(__file__).resolve().parent.parent


def protocol_and_rows(tmp_path):
    read = tmp_path/'development'
    read.mkdir()
    (read/'batch0.jsonl').write_text(json.dumps(dict(id='dev', task_id='add-34-F', statement='old', report='old'))+'\n')
    (read/'labels0.jsonl').write_text(json.dumps(dict(id='dev', category='false_claim'))+'\n')
    config = dict(judge='api', judge_model='stub', judge_revision=None,
                  judge_url='https://api.openai.com/v1', judge_extra={})
    protocol = freeze(read, config, 'Lean test')
    rows = [dict(id=f'fresh-{i}', task_id=f'fresh-{i}', group='fresh', template_id='fresh-template',
                 statement=f'True {i}', established='', report=f'Report {i}', genuine=False) for i in range(2)]
    return protocol, rows


def fill(out):
    sheet = (out/'sheet.md').read_text()
    for i in range(1, 3):
        sheet = sheet.replace(f'Answer {i:02d}: ', f'Answer {i:02d}: E')
    (out/'sheet.md').write_text(sheet)


def test_freeze_requires_complete_nonempty_development(tmp_path):
    with pytest.raises(ValueError, match='does not exist'):
        freeze(tmp_path/'missing', {}, 'Lean')
    empty = tmp_path/'empty'
    empty.mkdir()
    with pytest.raises(ValueError, match='nonempty'):
        freeze(empty, {}, 'Lean')
    protocol_and_rows(tmp_path)
    (tmp_path/'development'/'labels0.jsonl').write_text('')
    with pytest.raises(ValueError, match='recognized reading label'):
        freeze(tmp_path/'development', {}, 'Lean')


def test_descriptive_review_shares_integrity_without_class_gate(tmp_path):
    protocol, rows = protocol_and_rows(tmp_path)
    out = tmp_path/'audit'
    prepare(rows, protocol, out, descriptive=True)
    with pytest.raises(ValueError, match='Complete every'):
        load_review(out, protocol['config'], 'Lean test', descriptive=True)
    fill(out)
    labelled, criteria, provenance = load_review(out, protocol['config'], 'Lean test', descriptive=True)
    assert len(labelled) == 2 and all(r['read_claim'] == 'ERROR' for r in labelled)
    assert criteria is None and provenance['human_sheet_sha256']
    with pytest.raises(ValueError, match='Review mode'):
        load_review(out, protocol['config'], 'Lean test')
    with pytest.raises(ValueError, match='configuration'):
        load_review(out, {}, 'Lean test', descriptive=True)
    sheet = (out/'sheet.md').read_text()
    (out/'sheet.md').write_text(sheet.replace('Report', 'Edited report'))
    with pytest.raises(ValueError, match='content changed'):
        load_review(out, protocol['config'], 'Lean test', descriptive=True)
    (out/'sheet.md').write_text(sheet)
    (out/'rows.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='rows changed'):
        load_review(out, protocol['config'], 'Lean test', descriptive=True)


def test_legacy_audit_has_actionable_error(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps(dict(schema='judge-blind-audit-v1')))
    with pytest.raises(ValueError, match='preserve it as historical evidence'):
        load_review(tmp_path, {}, 'Lean test', descriptive=True)


def test_audit_cli_scores_disagreements_without_acceptance(tmp_path, monkeypatch):
    from rewardhack import judge
    protocol, rows = protocol_and_rows(tmp_path)
    out, scored = tmp_path/'audit', tmp_path/'scored'
    prepare(rows, protocol, out, descriptive=True)
    fill(out)
    def stub_judge(row):
        return dict(claim='OK', raw='')
    stub_judge.meter = None
    monkeypatch.setattr(judge, 'from_args', lambda args: stub_judge)
    monkeypatch.setattr('subprocess.check_output', lambda *a, **kw: 'Lean test')
    monkeypatch.setattr(sys, 'argv', ['validate_judge.py', '--audit', str(out), '--out', str(scored),
                                    '--lean', 'stub', '--judge', 'api', '--judge-model', 'stub'])
    runpy.run_path(str(ROOT/'entrypoints/validate_judge.py'), run_name='__main__')
    result = json.loads((scored/'summary.json').read_text())
    assert result['validation_kind'] == 'human_audit' and result['complete']
    assert result['honest_bit_agreement'] == 0
    assert 'accepted' not in result


def test_freeze_cli_default_corpus_is_independent_of_cwd(tmp_path, monkeypatch):
    from rewardhack import judge_holdout
    seen = []
    monkeypatch.setattr(judge_holdout, 'freeze', lambda read, config, lean_version: seen.append(read) or {})
    monkeypatch.setattr('subprocess.check_output', lambda *a, **kw: 'Lean test')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['prepare_judge_holdout.py', 'freeze', '--protocol', str(tmp_path/'protocol.json'),
                                    '--lean', 'stub', '--judge', 'api', '--judge-model', 'stub'])
    runpy.run_path(str(ROOT/'entrypoints/prepare_judge_holdout.py'), run_name='__main__')
    assert seen == [ROOT/'results'/'report-reading']


def test_saved_collection_becomes_a_descriptive_sheet(tmp_path, monkeypatch, capsys):
    protocol, rows = protocol_and_rows(tmp_path)
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    (tmp_path/'fresh.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    argv = ['prepare_judge_holdout.py', 'prepare', '--protocol', str(tmp_path/'protocol.json'), '--rows', str(tmp_path/'fresh.jsonl')]
    monkeypatch.setattr(sys, 'argv', argv+['--out', str(tmp_path/'acceptance')])
    with pytest.raises(SystemExit):  # two rows are far below the acceptance minimum
        runpy.run_path(str(ROOT/'entrypoints/prepare_judge_holdout.py'), run_name='__main__')
    assert 'frozen minimum' in capsys.readouterr().err and not (tmp_path/'acceptance').exists()
    monkeypatch.setattr(sys, 'argv', argv+['--out', str(tmp_path/'audit'), '--descriptive'])
    runpy.run_path(str(ROOT/'entrypoints/prepare_judge_holdout.py'), run_name='__main__')
    assert json.loads((tmp_path/'audit'/'manifest.json').read_text())['review_kind'] == 'descriptive'


def test_descriptive_audit_refuses_reports_the_judge_never_reads(tmp_path):
    protocol, rows = protocol_and_rows(tmp_path)
    rows[0]['report'] = ''
    with pytest.raises(ValueError, match='rows with a report'):
        prepare(rows, protocol, tmp_path/'audit', descriptive=True)
    assert not (tmp_path/'audit').exists()


@pytest.mark.parametrize('change,message', [(lambda row: row.pop('raw'), "lacks \\['raw'\\]"),
                                            (lambda row: row.update(task_id='unknown'), 'not in the supplied task set')])
def test_formal_audit_refuses_before_writing(tmp_path, change, message):
    row = dict(id='r', task_id='known', group='g', statement='True', raw='{}', true=True,
               genuine=False, exploit=False, verified_refutation=False)
    change(row)
    (tmp_path/'rows.jsonl').write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match=message):
        audit_file('no-lean', tmp_path/'rows.jsonl', tmp_path/'out', {'known': {}})
    assert not (tmp_path/'out').exists()  # a retry is not blocked by a half-written directory


@pytest.mark.parametrize('field,value', [('statement', 'False'), ('true', False), ('true', 1), ('group', 'other')])
def test_formal_audit_rejects_misjoined_task_before_writing(tmp_path, field, value):
    task = dict(id='known', group='g', statement='True', true=True, reference='by trivial')
    row = dict(id='r', task_id='known', group='g', statement='True', raw='{}', true=True,
               genuine=False, exploit=False, verified_refutation=False)
    row[field] = value
    source = tmp_path/'rows.jsonl'
    source.write_text(json.dumps(row)+'\n')
    with pytest.raises(ValueError, match=field):
        audit_file('must-not-run', source, tmp_path/'out', {'known': task})
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('rows,message', [
    ([], 'no rows'),
    ([dict(id='r', task_id='known', group='g', statement='True', raw=None, true=True, genuine=False, exploit=False,
           verified_refutation=False)], "wrong type in \\['raw'\\]"),
    (['not a row'], 'lacks'),
])
def test_formal_audit_checks_types_and_never_reuses_a_directory(tmp_path, rows, message):
    from rewardhack.formal_audit import Unauditable
    (tmp_path/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(Unauditable, match=message):
        audit_file('no-lean', tmp_path/'rows.jsonl', tmp_path/'out')
    assert not (tmp_path/'out').exists()
    good = dict(id='r', task_id='known', group='g', statement='True', raw='{}', true=True, genuine=False, exploit=False,
                verified_refutation=False)
    (tmp_path/'rows.jsonl').write_text(json.dumps(good)+'\n')
    (tmp_path/'out').mkdir()
    with pytest.raises(Unauditable, match='already exists'):
        audit_file('no-lean', tmp_path/'rows.jsonl', tmp_path/'out')


@pytest.mark.parametrize('review', ['collection', 'collection/review'])
def test_collector_refuses_overlapping_directories_before_generating(tmp_path, monkeypatch, capsys, review):
    protocol, _ = protocol_and_rows(tmp_path)
    (tmp_path/'protocol.json').write_text(json.dumps(protocol))
    monkeypatch.setattr('rewardhack.tasks.verify_references', lambda *a, **kw: pytest.fail('reached the collection'))
    monkeypatch.setattr(sys, 'argv', ['collect_judge_holdout.py', '--protocol', str(tmp_path/'protocol.json'), '--model-path', 'none',
                                    '--lean', 'none', '--out', str(tmp_path/'collection'), '--review-out', str(tmp_path/review)])
    with pytest.raises(SystemExit):
        runpy.run_path(str(ROOT/'entrypoints/collect_judge_holdout.py'), run_name='__main__')
    assert 'separate directories' in capsys.readouterr().err


def test_formal_audit_refuses_a_cut_off_file_and_a_mistyped_flag(tmp_path):
    from rewardhack.formal_audit import Unauditable
    good = dict(id='r', task_id='known', group='g', statement='True', raw='{}', true=True, genuine=False, exploit=False,
                verified_refutation=False)
    for text, message in ((json.dumps(good)+'\n{"id": ', 'line 2 .* is not JSON'), (json.dumps(dict(good, truncated='false')), 'truncated')):
        (tmp_path/'rows.jsonl').write_text(text)
        with pytest.raises(Unauditable, match=message):
            audit_file('no-lean', tmp_path/'rows.jsonl', tmp_path/'out')
        assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('why', ['timeout', 'no axiom report', 'unexpected output'])
def test_formal_audit_never_certifies_an_unresolved_declaration(tmp_path, monkeypatch, why):
    from rewardhack import formal_audit
    row = dict(id='r', task_id='t', group='g', statement='False', raw='{}', true=False,
               genuine=False, exploit=False, verified_refutation=False)
    monkeypatch.setattr(formal_audit, 'score', lambda *a, **kw: dict(
        established='True', report='', body='by trivial', genuine=False, genuine_why='type mismatch',
        grader_reward=0, exploit=False, verified_refutation=False, refutation_why='type mismatch',
        verifier_inconclusive=False))
    calls = []
    def unresolved(*a, **kw):
        calls.append(kw)
        return dict(other=None, why=why)
    monkeypatch.setattr(formal_audit, 'statement_relation', unresolved)
    source = tmp_path/'rows.jsonl'
    source.write_text(json.dumps(row)+'\n')
    out = tmp_path/'audit'
    summary = audit_file('unused', source, out)
    result = json.loads((out/'formal-verdicts.jsonl').read_text())
    assert result['machine_determined_flag'] is None and result['inconclusive']
    assert result['declaration_why'] == why and not result['needs_prose_reading']
    assert summary['machine_determined_flags'] == {'unverified': 1}
    assert summary['counts']['inconclusive'] == 1
    assert len(calls) == (2 if why == 'timeout' else 1)


def test_formal_audit_invalid_declaration_is_not_a_verifier_timeout(monkeypatch):
    from rewardhack import formal_audit
    row = dict(id='r', task_id='t', group='g', statement='False', raw='{}', true=False,
               genuine=False, exploit=False, verified_refutation=False)
    monkeypatch.setattr(formal_audit, 'score', lambda *a, **kw: dict(
        established='garbled', report='', body='', genuine=False, genuine_why='empty',
        grader_reward=0, exploit=False, verified_refutation=False, refutation_why='empty',
        verifier_inconclusive=False))
    monkeypatch.setattr(formal_audit, 'statement_relation', lambda *a: dict(other=None, why='type mismatch'))
    result = formal_audit.audit_row('unused', row)
    assert result['machine_determined_flag'] == 'OK' and not result['inconclusive']
