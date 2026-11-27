"""DRY candidates need positive examples and counterexamples before enforcement."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from rewardhack.dry_audit import audit, read_repository


def findings(sources, rule):
    return [item for item in audit(sources, min_nodes=1)['findings'] if item['rule'] == rule]


def test_cross_language_constants_and_drift():
    sources = {'policy.py': 'SERVER_PORT = 8123\nREQUEST_SECONDS = 80\n',
               'pins.env': 'SERVER_PORT=8123\nREQUEST_SECONDS=90\n'}
    repeated = findings(sources, 'constant-repeat')
    drift = findings(sources, 'constant-drift')
    assert len(repeated) == len(drift) == 1
    assert {item['role'] for item in repeated[0]['occurrences']} == {'SERVER_PORT'}
    assert drift[0]['values'] == [('int', '80'), ('int', '90')]
    assert {item['line'] for item in drift[0]['occurrences']} == {2}


def test_unrelated_equal_numbers_and_state_transitions_are_not_constants():
    sources = {'a.py': 'TIMEOUT = 600\nBATCH_SIZE = 600\nSTATE = 0\nSTATE = 1\n',
               'b.sh': 'STATE=0\nSTATE=1\n'}
    assert not audit(sources)['findings']


def test_nonliteral_reassignment_is_also_mutable():
    assert not audit({'a.py': 'LIMIT = 60\nLIMIT = read_limit()\n', 'b.py': 'LIMIT = 60\n'})['findings']
    assert not audit({'a.sh': 'LIMIT=60\nLIMIT=$USER_LIMIT\n', 'b.sh': 'LIMIT=60\n'})['findings']


def test_local_definitions_have_separate_scopes():
    rows = findings({'a.py': 'def a():\n    timeout = 60\n\ndef b():\n    timeout = 60\n'}, 'constant-repeat')
    assert len(rows) == 1
    assert len(rows[0]['occurrences']) == 2


def test_strings_link_differently_named_definitions_and_defaults():
    sources = {'a.py': 'MODEL = "acme/large-model"\nparser.add_argument("--model", default="acme/large-model")\n',
               'b.env': 'TRAIN_MODEL="acme/large-model"\n'}
    rows = findings(sources, 'literal-repeat')
    assert len(rows) == 1
    assert len(rows[0]['occurrences']) == 3


def test_literals_are_typed_and_expressions_are_not_evaluated():
    sources = {'a.py': 'LIMIT = 1\nOTHER = True\nSIGNED = -12\n',
               'b.py': 'LIMIT = "1"\nOTHER = 1\nSIGNED = -12\nresult = dangerous()\n'}
    assert len(findings(sources, 'constant-drift')) == 1
    assert len(findings(sources, 'constant-repeat')) == 1


def test_different_callees_do_not_share_a_keyword_policy():
    assert not audit({'a.py': 'connect(timeout=60)\nsleep(timeout=60)\n'})['findings']
    assert len(findings({'a.py': 'connect(timeout=60)\nconnect(timeout=60)\n'}, 'constant-repeat')) == 1


def test_shell_subset_is_explicit():
    sources = {'a.sh': 'MODEL="acme/large-model"\nPORT=${PORT:-8123}\nX=1; Y=2\n',
               'b.py': 'MODEL = "acme/large-model"\nPORT = 8123\n'}
    report = audit(sources)
    assert len(report['findings']) == 1
    assert report['coverage_limits'][0]['path'] == 'a.sh'
    assert report['constants'] == 5


def test_shell_sequences_quotes_and_reused_model_variables():
    sources = {'a.sh': 'M=acme/large-model; R=0123456789abcdef\nM=acme/other-model; R=fedcba9876543210\n',
               'b.env': "MODEL='acme/large-model'\nREV=0123456789abcdef\nTEXT='semi; ASSIGN=inside'\n"}
    rows = findings(sources, 'literal-repeat')
    assert len(rows) == 2
    assert not findings(sources, 'constant-drift')
    assert audit(sources)['constants'] == 7


def test_renamed_functions_preserve_repeated_bindings():
    sources = {'a.py': 'def first(x):\n    y = x + x\n    return y * x\n',
               'b.py': 'def second(a):\n    b = a + a\n    return b * a\n',
               'c.py': 'def third(a):\n    b = a + a\n    return b * b\n'}
    rows = findings(sources, 'function-clone')
    assert len(rows) == 1 and rows[0]['match'] == 'renamed'
    assert {item['path'] for item in rows[0]['occurrences']} == {'a.py', 'b.py'}


@pytest.mark.parametrize('body', ['return external_b(x)', 'return external_a(x + 1)', 'return external_a(x - 2)',
                                 'return local_0(x)', 'return x(x)'])
def test_external_calls_operators_and_literals_are_not_erased(body):
    assert not findings({'a.py': 'def a(x):\n    return external_a(x + 2)\n',
                         'b.py': f'def b(x):\n    {body}\n'}, 'function-clone')


def test_canonical_names_cannot_collide_with_external_names():
    assert not findings({'a.py': 'def a(x):\n    return x\n',
                         'b.py': 'def b(y):\n    return local_0\n'}, 'function-clone')


def test_defaults_annotations_and_decorators_keep_external_bindings():
    sources = {'a.py': '@cache\ndef a(x: x = x):\n    return x\n',
               'b.py': '@cache\ndef b(y: y = y):\n    return y\n',
               'c.py': '@other_cache\ndef c(x: x = x):\n    return x\n'}
    assert not findings(sources, 'function-clone')


def test_exact_functions_ignore_docstrings_and_source_positions():
    sources = {'a.py': 'def first(x):\n    "doc one"\n    return x + 3\n',
               'b.py': '\n\ndef second(x):\n    "doc two"\n    return x + 3\n'}
    rows = findings(sources, 'function-clone')
    assert len(rows) == 1 and rows[0]['match'] == 'exact'
    assert [item['line'] for item in rows[0]['occurrences']] == [1, 3]


def test_complex_scopes_are_reported_and_still_match_exactly():
    report = audit({'a.py': 'def a(x):\n    return [x for x in x]\n',
                    'b.py': 'def b(x):\n    return [x for x in x]\n',
                    'c.py': 'def c(y):\n    return [y for y in y]\n'}, min_nodes=1)
    assert len(report['coverage_limits']) == 3
    rows = report['findings']
    assert len(rows) == 1 and rows[0]['match'] == 'exact'


def test_atomic_snapshot_can_remove_both_copies():
    before = {'a.py': 'LIMIT = 8123\n', 'b.py': 'LIMIT = 8123\n'}
    after = {'policy.py': 'LIMIT = 8123\n', 'a.py': 'from policy import LIMIT\n',
             'b.py': 'from policy import LIMIT\n'}
    assert audit(before)['findings']
    assert not audit(after)['findings']


def test_invalid_input_never_looks_like_a_clean_audit():
    with pytest.raises(SyntaxError, match='broken.py'):
        audit({'broken.py': 'def broken('})
    with pytest.raises(ValueError, match='unsupported'):
        audit({'config.toml': 'port = 42'})
    with pytest.raises(ValueError, match='positive'):
        audit({}, min_nodes=0)


def test_size_threshold_and_deterministic_order():
    sources = {'b.py': 'def b(x):\n    return x + 3\n', 'a.py': 'def a(x):\n    return x + 3\n'}
    assert not audit(sources)['findings']
    assert audit(sources, min_nodes=1) == audit(dict(reversed(list(sources.items()))), min_nodes=1)


def test_repository_reader_and_cli(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'a.py').write_text('LIMIT = 8123\n')
    (tmp_path / 'b.py').write_text('LIMIT = 8123\n')
    (tmp_path / 'test_expected.py').write_text('LIMIT = 8123\n')
    (tmp_path / '.gitignore').write_text('ignored.py\n')
    (tmp_path / 'ignored.py').write_text('broken syntax(')
    sources, excluded = read_repository(tmp_path)
    assert set(sources) == {'a.py', 'b.py'}
    assert excluded == ['test_expected.py']
    assert 'test_expected.py' in read_repository(tmp_path, include_tests=True)[0]
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'entrypoints/audit_dry.py'),
               '--root', str(tmp_path), '--json']
    done = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert done.returncode == 0, done.stderr
    assert len(json.loads(done.stdout)['findings']) == 1
    assert subprocess.run([*command, '--check'], capture_output=True, timeout=30).returncode == 1
    (tmp_path / 'a.py').write_text('broken syntax(')
    done = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert done.returncode != 0 and 'SyntaxError' in done.stderr


def test_repository_rejects_source_symlinks(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'a.py').symlink_to('/missing/source.py')
    with pytest.raises(ValueError, match='symlink'):
        read_repository(tmp_path)


def test_subdirectories_cannot_be_mistaken_for_a_complete_repository(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    child = tmp_path / 'src'
    child.mkdir()
    with pytest.raises(ValueError, match='repository root'):
        read_repository(child)
