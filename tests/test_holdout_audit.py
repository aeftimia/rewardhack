"""Blind-audit completion, integrity, and fail-closed development exclusions."""
import json
import runpy
import sys
from pathlib import Path

import pytest

from rewardhack.holdout_collection import prepare_audit
from rewardhack.judge_holdout import freeze, load_review

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
    prepare_audit(rows, protocol, out)
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
    prepare_audit(rows, protocol, out)
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
    monkeypatch.setattr(judge_holdout, 'freeze', lambda read, *a: seen.append(read) or {})
    monkeypatch.setattr('subprocess.check_output', lambda *a, **kw: 'Lean test')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['prepare_judge_holdout.py', 'freeze', '--protocol', str(tmp_path/'protocol.json'),
                                    '--lean', 'stub', '--judge', 'api', '--judge-model', 'stub'])
    runpy.run_path(str(ROOT/'entrypoints/prepare_judge_holdout.py'), run_name='__main__')
    assert seen == [ROOT/'results'/'report-reading']
