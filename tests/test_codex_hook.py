"""Verify the adapter checks proposed edits without changing working files."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
from rewardhack import codex_hooks as adapter

ROOT = Path(__file__).resolve().parents[1]


def patch(body):
    return '*** Begin Patch\n' + body + '*** End Patch\n'


def test_multi_file_patch_and_move_do_not_write(tmp_path):
    source = tmp_path / 'old.py'
    source.write_text('def f():\n    return 1\n\nx = 2\n')
    text = patch('*** Update File: old.py\n*** Move to: new.py\n@@ def f():\n'
                 '-    return 1\n+    return 3\n@@\n-x = 2\n+x = 4\n*** End of File\n'
                 '*** Add File: added.py\n+y = 5\n')
    proposed = list(adapter.proposed_files(text, tmp_path))
    assert proposed == [(tmp_path / 'new.py', source.read_text(), 'def f():\n    return 3\n\nx = 4\n'),
                        (tmp_path / 'added.py', '', 'y = 5\n')]
    assert source.read_text() == 'def f():\n    return 1\n\nx = 2\n'
    assert not (tmp_path / 'new.py').exists()


@pytest.mark.parametrize('body', [
    '*** Update File: x.py\n@@\n-no such context\n+new\n',
    '*** Update File: x.py\n@@\n-x = 1\n+x = 2\n',
    '*** Add File: x.py\n+x = 2\n',
    '*** Delete File: missing.py\n',
])
def test_unreadable_or_ambiguous_patch_is_refused(tmp_path, body):
    (tmp_path / 'x.py').write_text('x = 1\nx = 1\n')
    with pytest.raises(ValueError):
        list(adapter.proposed_files(patch(body), tmp_path))


def test_delete_and_append(tmp_path):
    (tmp_path / 'x.py').write_text('x = 1\n')
    assert list(adapter.proposed_files(patch('*** Delete File: x.py\n'), tmp_path)) == []
    edits = list(adapter.proposed_files(patch('*** Update File: x.py\n@@\n+y = 2\n'), tmp_path))
    assert edits[0][2] == 'x = 1\ny = 2\n'


@pytest.fixture
def checker(monkeypatch):
    dependency = Path.home() / '.claude/hooks/no-silent-fallbacks/no_silent_fallbacks.py'
    if not dependency.exists():
        pytest.skip('integration test requires the installed shared checker')
    # Network/update behavior has its own tests; these exercise the installed scanner.
    monkeypatch.setattr(adapter, 'load_hook_updater',
                        lambda: SimpleNamespace(source_module=lambda name: dependency))
    return adapter.load_checker()


def test_uv_upgrades_runtime_before_checker_is_loaded(tmp_path, monkeypatch):
    events = []
    checker = object()
    def source(name):
        events.append(('source', name))
        return tmp_path / 'no_silent_fallbacks.py'
    def load(name, path):
        events.append(('load', name))
        if name == 'shared_hook_updater':
            return SimpleNamespace(source_module=source)
        assert path == tmp_path / 'no_silent_fallbacks.py'
        return checker
    monkeypatch.setattr(adapter, 'load_source_module', load)
    def run(arguments, **kwargs):
        events.append(('command', tuple(arguments[:3])))
        if arguments[:3] == ['uv', 'tool', 'dir']:
            return SimpleNamespace(returncode=0, stdout=str(tmp_path), stderr='')
        return SimpleNamespace(returncode=0, stdout=str(tmp_path / 'updater.py'), stderr='')
    monkeypatch.setattr(adapter.subprocess, 'run', run)
    assert adapter.load_checker() is checker
    assert events[0] == ('command', ('uv', 'tool', 'upgrade'))
    assert events[-3:] == [('load', 'shared_hook_updater'), ('source', 'no-silent-fallbacks'),
                          ('load', 'shared_no_silent_fallbacks')]


def test_missing_updater_is_an_error(tmp_path, monkeypatch):
    def run(arguments, **kwargs):
        if arguments[0] == 'uv':
            return SimpleNamespace(returncode=0, stdout=str(tmp_path), stderr='')
        raise FileNotFoundError(arguments[0])
    monkeypatch.setattr(adapter.subprocess, 'run', run)
    with pytest.raises(FileNotFoundError):
        adapter.load_hook_updater()


def test_actual_checker_refuses_new_swallowed_error(tmp_path, checker):
    call = dict(tool_name='apply_patch', cwd=str(tmp_path), tool_input=dict(command=patch(
        '*** Add File: bad.py\n+try:\n+    work()\n+except Exception:\n+    pass\n')))
    assert adapter.check(call, checker) == 2
    assert not (tmp_path / 'bad.py').exists()


def test_actual_checker_allows_clean_edit_and_checks_shell(tmp_path, checker):
    call = dict(tool_name='apply_patch', cwd=str(tmp_path), tool_input=dict(command=patch(
        '*** Add File: good.py\n+value = data["value"]\n')))
    assert adapter.check(call, checker) == 0
    # The checker deliberately permits scratch writes under /tmp; this checkout may itself be a temporary worktree.
    # Only inspect the command, using a non-temporary synthetic source directory; nothing is executed or written.
    call.update(tool_name='Bash', cwd=str(Path.home()/'hook-source-fixture'), tool_input=dict(command='echo "x = 1" > bad.py'))
    assert adapter.check(call, checker) == 2


def test_malformed_input_fails_closed(checker):
    done = subprocess.run(['bash', str(ROOT / 'ops/gate.sh'), 'codex-hook'],
                          input=json.dumps({'tool_name': 'apply_patch'}), text=True,
                          capture_output=True, timeout=10)
    assert done.returncode == 2
    assert 'check itself failed' in done.stderr


def test_project_argument_rule_applies_to_proposed_content(tmp_path, checker):
    (tmp_path / '.no-silent-fallbacks.json').write_text(json.dumps({
        'explicit_arguments': {'model.generate': ['max_new_tokens']}}))
    call = dict(tool_name='apply_patch', cwd=str(tmp_path), tool_input=dict(command=patch(
        '*** Add File: generate.py\n+model.generate(inputs)\n')))
    assert adapter.check(call, checker) == 2
    call['tool_input']['command'] = patch(
        '*** Add File: generate.py\n+model.generate(inputs, max_new_tokens=100)\n')
    assert adapter.check(call, checker) == 0
