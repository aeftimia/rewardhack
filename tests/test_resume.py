"""A run resumed from a checkpoint continues exactly where it stopped (owner, 26 Nov: Part 1 runs may be extended).

The real TRL trainer on a tiny CPU model, as in test_training_integration. The reward depends on the sampled text, so a
resume that lost the random state would sample other completions, and one that lost the optimizer state would take
other steps: both are shown to break the match, so the match is evidence.
"""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import run_options

from rewardhack import hf_grpo, outcomes
from test_training_integration import REWARDS, train

torch = pytest.importorskip('torch')
REWARD = REWARDS['depends on the text']
ROOT = Path(__file__).resolve().parent.parent


def checkpoint(run, step):
    return run.out/'trainer'/f'checkpoint-{step}'


def lines(run, name):
    return [json.loads(line) for line in (run.out/name).read_text().splitlines()]


def resumed_from(tmp_path, monkeypatch, path, steps, reward=REWARD):
    return train(tmp_path, monkeypatch, reward, steps=steps, save_every=2, resume_from=path)


@pytest.fixture
def runs(tmp_path, monkeypatch):
    whole = train(tmp_path, monkeypatch, REWARD, steps=4, save_every=2)
    first_half = train(tmp_path, monkeypatch, REWARD, steps=2, save_every=2)
    return whole, first_half


def test_a_resumed_run_ends_with_the_weights_of_an_uninterrupted_one(tmp_path, monkeypatch, runs):
    whole, first_half = runs
    resumed = resumed_from(tmp_path, monkeypatch, checkpoint(first_half, 2), 4)
    assert float(whole.change.abs().max()) > 1e-4, 'the run moved nothing, so the comparison below would be empty'
    assert torch.equal(resumed.change, whole.change)
    for key in ('loss', 'grad_norm', 'learning_rate', 'reward'):
        assert [entry[key] for entry in resumed.log] == [entry[key] for entry in whole.log], key  # the history is restored too


def unnamed(rows):
    """`rows` without the run name in their provenance, which differs between an uninterrupted run and its resumption."""
    return [dict(row, meta={k: v for k, v in row['meta'].items() if k != 'run'}) if 'meta' in row else row for row in rows]


def test_a_resumed_run_carries_the_records_and_counters_of_the_steps_before_it(tmp_path, monkeypatch, runs):
    whole, first_half = runs
    resumed = resumed_from(tmp_path, monkeypatch, checkpoint(first_half, 2), 4)
    # The same completions, the same per-step records and the same run totals as the uninterrupted run.
    copied = {'copied_from': str(first_half.out)}
    expected_rows = [dict(row, **copied) if i < 2*4 else row for i, row in enumerate(whole.rows)]
    expected_lines = [dict(line, **copied) if line['step'] <= 2 else line for line in lines(whole, 'steps.jsonl')]
    # Each run stamps its own name in its rows' provenance; everything else must match.
    assert unnamed(resumed.rows) == unnamed(expected_rows) and unnamed(lines(resumed, 'steps.jsonl')) == unnamed(expected_lines)
    assert [line['step'] for line in lines(resumed, 'steps.jsonl')] == [1, 2, 3, 4]
    for key in ('rollouts', 'first_seen_step', 'mixed_reward_groups', 'reward_groups', 'global_step'):
        assert resumed.summary[key] == whole.summary[key], key
    assert sorted(p.name for p in resumed.out.glob('adapter-step*')) == ['adapter-step4']


def test_the_no_variation_stop_counts_the_steps_before_the_checkpoint(tmp_path, monkeypatch):
    """Variation in steps 1-2, none after: a resume from step 2 must not stop at step 3 as if nothing had varied."""
    reward = lambda i, raw, step: float(i == 0) if step <= 2 else 1.0
    first = train(tmp_path, monkeypatch, reward, steps=2, save_every=2, stop_no_variation=3)
    resumed = train(tmp_path, monkeypatch, lambda i, raw, step: 1.0, steps=4, save_every=2, stop_no_variation=3,
                    resume_from=checkpoint(first, 2))
    assert 'early_stop' not in resumed.summary and resumed.summary['global_step'] == 4


def test_a_finished_run_extends_from_its_last_checkpoint(tmp_path, monkeypatch, runs):
    whole, _ = runs
    longer = train(tmp_path, monkeypatch, REWARD, steps=6, save_every=2)
    extended = resumed_from(tmp_path, monkeypatch, checkpoint(whole, 4), 6)
    assert [line['step'] for line in lines(extended, 'steps.jsonl')] == [1, 2, 3, 4, 5, 6]
    assert extended.summary['global_step'] == 6 and torch.equal(extended.change, longer.change)
    # Only the latest two checkpoints are kept: a resume needs one.
    assert sorted(p.name for p in (longer.out/'trainer').glob('checkpoint-*')) == ['checkpoint-4', 'checkpoint-6']


def test_a_run_stopped_early_keeps_a_checkpoint_of_its_last_step(tmp_path, monkeypatch):
    stopped = train(tmp_path, monkeypatch, REWARD, steps=4, save_every=2, train_deadline=time.monotonic()-1)
    assert stopped.summary['early_stop'] == 'deadline' and stopped.summary['global_step'] == 1
    assert (checkpoint(stopped, 1)/'trainer_state.json').is_file()


@pytest.mark.parametrize('lost', ['optimizer.pt', 'rng_state.pth'])
def test_the_match_depends_on_what_the_checkpoint_restores(tmp_path, monkeypatch, runs, lost):
    """Control: without the optimizer state or the random state, the resumed run no longer matches."""
    whole, first_half = runs
    damaged = tmp_path/'damaged'/'trainer'/'checkpoint-2'
    shutil.copytree(first_half.out, damaged.parent.parent, dirs_exist_ok=True)
    (damaged/lost).unlink()
    resumed = resumed_from(tmp_path, monkeypatch, damaged, 4)
    assert not torch.equal(resumed.change, whole.change)


SETTINGS = dict(model='m', revision='r', arm='report_correct', penalty=0.5, report_mode='prose', taskset='hard', train_limit=None,
                prompt_variant='neutral', reasoning_effort='low', thinking='unsupported', mxfp4='dequantize', quant='none',
                attn_implementation='eager', experts_implementation=None, lora_rank=16, learning_rate=1e-4, lr_schedule='constant',
                scale_rewards='none', max_grad_norm=1.0, router_aux_loss_coef=0.0, beta=0.0, num_generations=4, prompts_per_step=1,
                max_completion_length=2048, seed=7, save_every=2, stop_no_variation=99, stop_saturated=True, rollout_backend='vllm', judge='api',
                judge_model='gpt-5.4-nano', judge_revision=None, judge_url='https://api.openai.com/v1', judge_extra={},
                verifier_retry_seconds=120.0, adapter=None, eval_limit=12, eval_samples=2, training_sampler='uniform')


@pytest.mark.parametrize('saved_step', [2, 6])
def test_stratified_real_trainer_resume_matches_uninterrupted(tmp_path, monkeypatch, saved_step):
    import test_training_integration as integration
    rows = [dict(row, family='first' if index < 2 else 'second')
            for index, row in enumerate(integration.TASKS)]
    monkeypatch.setattr(integration, 'TASKS', rows)
    options = dict(save_every=2, training_sampler='family-truth')
    whole = train(tmp_path, monkeypatch, REWARD, steps=8, **options)
    partial = train(tmp_path, monkeypatch, REWARD, steps=saved_step, **options)
    resumed = train(tmp_path, monkeypatch, REWARD, steps=8,
                    resume_from=checkpoint(partial, saved_step), **options)
    assert torch.equal(resumed.change, whole.change)
    assert [row['task_id'] for row in resumed.rows] == [row['task_id'] for row in whole.rows]
    assert [row['raw'] for row in resumed.rows] == [row['raw'] for row in whole.rows]


@pytest.fixture
def saved(tmp_path):
    """A checkpoint's folder layout without training: what `check_resume` reads, written as `run` writes it."""
    assert set(SETTINGS) == set(hf_grpo.RESUME_SETTINGS)
    original = tmp_path/'original'
    (original/'trainer'/'checkpoint-2').mkdir(parents=True)
    (original/'trainer'/'checkpoint-2'/'trainer_state.json').write_text(json.dumps({'global_step': 2, 'epoch': 0.5}))
    for name in hf_grpo.CHECKPOINT_FILES:
        (original/'trainer'/'checkpoint-2'/name).write_bytes(b'')
    (original/'summary.json').write_text(json.dumps({'status': 'completed', 'global_step': 2}))
    row = dict(__import__('test_scan_fixes').fresh(), claim='OK', honest=True)
    (original/'training-rollouts.jsonl').write_text(''.join(json.dumps(row)+'\n' for _ in range(2*4)))
    (original/'steps.jsonl').write_text(''.join(json.dumps({'step': n}).strip()+'\n' for n in (1, 2)))
    (original/'before-evaluation.jsonl').write_text(''.join(json.dumps(row)+'\n' for _ in range(2)))  # one task, two samples
    (original/hf_grpo.RUN_SETTINGS).write_text(json.dumps(hf_grpo.run_settings(SimpleNamespace(**SETTINGS, steps=4))))
    return original


def resume_args(saved, tmp_path, **changes):
    args = run_options(**SETTINGS, steps=4, resume_from=saved/'trainer'/'checkpoint-2', out=tmp_path/'resumed')
    vars(args).update(changes)
    return args


def test_the_same_settings_pass(saved, tmp_path):
    hf_grpo.check_resume(resume_args(saved, tmp_path))


@pytest.mark.parametrize('changes, refusal', [
    (dict(arm='baseline'), "arm 'report_correct' -> 'baseline'"),
    (dict(penalty=0.0), 'penalty 0.5 -> 0.0'),
    (dict(learning_rate=1e-3), 'learning_rate'),
    (dict(seed=8), 'seed 7 -> 8'),
    (dict(judge_model='gpt-5.4-mini'), "judge_model 'gpt-5.4-nano' -> 'gpt-5.4-mini'"),
    (dict(judge_extra={'temperature': 0}), 'judge_extra'),
    (dict(verifier_retry_seconds=0.0), 'verifier_retry_seconds'),
    (dict(stop_no_variation=24), 'stop_no_variation'),
    (dict(stop_saturated=False), 'stop_saturated'),
    (dict(attn_implementation='sdpa'), 'attn_implementation'),
    (dict(steps=2), 'must exceed the 2 steps'),
    (dict(eval_limit=24), 'eval_limit 12 -> 24'),
    (dict(eval_samples=4), 'eval_samples 2 -> 4'),
    (dict(lr_schedule='linear'), 'lr_schedule'),
])
def test_a_resume_with_other_settings_is_refused(saved, tmp_path, changes, refusal):
    with pytest.raises(ValueError) as raised:
        hf_grpo.check_resume(resume_args(saved, tmp_path, **changes))
    assert refusal in str(raised.value)


def test_a_folder_that_is_not_a_checkpoint_is_refused(saved, tmp_path):
    with pytest.raises(ValueError, match='not a trainer checkpoint'):
        hf_grpo.check_resume(resume_args(saved, tmp_path, resume_from=saved/'trainer'))


def test_a_resume_into_the_original_folder_is_refused(saved, tmp_path):
    with pytest.raises(ValueError, match='would be overwritten'):
        hf_grpo.check_resume(resume_args(saved, tmp_path, out=saved))


def test_a_run_that_started_from_an_adapter_cannot_be_resumed(saved, tmp_path):
    (saved/hf_grpo.RUN_SETTINGS).write_text(json.dumps(hf_grpo.run_settings(SimpleNamespace(**dict(SETTINGS, adapter=Path('a')), steps=4))))
    with pytest.raises(ValueError, match='started from --adapter a'):
        hf_grpo.check_resume(resume_args(saved, tmp_path))


@pytest.mark.parametrize('changes, refusal', [
    (dict(out='original'), 'would be overwritten'),
    (dict(arm='baseline'), 'other settings'),
])
def test_a_refused_resume_writes_nothing(saved, tmp_path, changes, refusal):
    """`run` checks first: neither the original run's folder nor a new --out is touched by a refusal."""
    before = {p.relative_to(saved): p.read_bytes() for p in saved.rglob('*') if p.is_file()}
    changes = dict(changes, out=saved) if 'out' in changes else changes
    args = resume_args(saved, tmp_path, mode='train', **changes)
    with pytest.raises(ValueError, match=refusal):
        hf_grpo.run(args)
    assert {p.relative_to(saved): p.read_bytes() for p in saved.rglob('*') if p.is_file()} == before
    assert not (tmp_path/'resumed').exists()


def test_a_resume_after_a_code_change_is_refused(saved, tmp_path):
    settings = json.loads((saved/hf_grpo.RUN_SETTINGS).read_text())
    settings['code_sha256']['outcomes.py'] = '0'*64
    (saved/hf_grpo.RUN_SETTINGS).write_text(json.dumps(settings))
    with pytest.raises(ValueError, match='written by other code; changed since: outcomes.py'):
        hf_grpo.check_resume(resume_args(saved, tmp_path))


def test_the_verifier_retry_allowance_continues_from_the_checkpoint(tmp_path, monkeypatch, runs):
    """The allowance is the whole run's: a resumed run starts with what the checkpointed step left, not a full one."""
    _, first_half = runs
    records = lines(first_half, 'steps.jsonl')
    records[-1].update(retries_left=7.5, retries_denied=3)
    (first_half.out/'steps.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    resumed = resumed_from(tmp_path, monkeypatch, checkpoint(first_half, 2), 4)
    after = lines(resumed, 'steps.jsonl')[2:]
    assert [(r['retries_left'], r['retries_denied']) for r in after] == [(7.5, 3), (7.5, 3)]
    assert resumed.summary['training_verifier']['retries_denied'] == 3


def test_a_run_that_fails_keeps_a_resumable_checkpoint_of_its_last_step(tmp_path, monkeypatch):
    """A stop by exception (here the scoring step) still leaves the last completed step resumable, and the resume
    leaves out the failed batch's partial rows."""
    def failing(i, raw, step):
        if step == 3 and i == 1:  # mid-batch: the first rollout of step 3 is scored and saved as a partial row
            raise RuntimeError('scoring failed')
        return REWARD(i, raw, step)
    summary = dict(rollouts={})
    with pytest.raises(RuntimeError, match='scoring failed'):
        train(tmp_path, monkeypatch, failing, steps=4, save_every=5, summary=summary)
    assert summary['untrained_step'] == 3 and summary['global_step'] == 2  # the step whose rows did not update the model
    failed = SimpleNamespace(out=tmp_path/'run0')
    assert [row.get('partial_batch') for row in lines(failed, 'training-rollouts.jsonl')][-1] is True
    assert (checkpoint(failed, 2)/'optimizer.pt').is_file() and not checkpoint(failed, 3).exists()
    resumed = resumed_from(tmp_path, monkeypatch, checkpoint(failed, 2), 4)
    assert resumed.summary['global_step'] == 4 and not any(row.get('partial_batch') for row in resumed.rows)


@pytest.mark.parametrize('extra, said', [
    (['--adapter', 'a', '--resume-from', 'b'], 'not allowed with argument'),
    (['--resume-from', 'b'], '--resume-from continues training'),
])
def test_the_command_line_refuses_a_resume_it_cannot_honour(tmp_path, extra, said):
    mode = 'train' if '--adapter' in extra else 'probe'
    done = subprocess.run([sys.executable, str(ROOT/'entrypoints'/'run.py'), mode, '--out', str(tmp_path), '--lean', 'x', *extra],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 2 and said in done.stderr


def test_run_totals_take_the_largest_proof_length_not_the_sum():
    total = {}
    outcomes.accumulate(total, dict(n=4, inconclusive_longest_proof_chars=900))
    outcomes.accumulate(total, dict(n=4, inconclusive_longest_proof_chars=40))
    assert total == dict(n=8, inconclusive_longest_proof_chars=900)


def test_a_linear_schedule_run_cannot_be_extended(saved, tmp_path):
    """Its restored scheduler decays over the original --steps, so a longer total would not match an uninterrupted run."""
    (saved/hf_grpo.RUN_SETTINGS).write_text(json.dumps(hf_grpo.run_settings(SimpleNamespace(**dict(SETTINGS, lr_schedule='linear'), steps=4))))
    with pytest.raises(ValueError, match='cannot be extended'):
        hf_grpo.check_resume(resume_args(saved, tmp_path, lr_schedule='linear', steps=6))
    hf_grpo.check_resume(resume_args(saved, tmp_path, lr_schedule='linear', steps=4))  # resuming the same total is fine



def tasks_per_step(run):
    return [row['task_id'] for row in run.rows[::4]]


@pytest.mark.parametrize('stop', [2, 4, 6])
def test_a_resumed_run_trains_on_the_tasks_the_uninterrupted_run_would(tmp_path, monkeypatch, stop):
    """The task order is a pass-by-pass shuffle from TRL's own generator. Four tasks make a pass of four steps, so these
    resume inside the first pass, at its end, and inside the second. The tiny model's completions barely depend on the
    prompt, so equal weights alone would not show the order; the task of every step is compared."""
    uninterrupted = train(tmp_path, monkeypatch, REWARD, steps=8, save_every=2)
    first = train(tmp_path, monkeypatch, REWARD, steps=stop, save_every=2)
    resumed = resumed_from(tmp_path, monkeypatch, checkpoint(first, stop), 8)
    assert len(set(tasks_per_step(uninterrupted)[:4])) == 4 and tasks_per_step(uninterrupted)[:4] != tasks_per_step(uninterrupted)[4:]
    assert tasks_per_step(resumed) == tasks_per_step(uninterrupted)
    assert torch.equal(resumed.change, uninterrupted.change)


def test_a_resume_with_other_training_libraries_is_refused(saved, tmp_path):
    settings = json.loads((saved/hf_grpo.RUN_SETTINGS).read_text())
    settings['libraries']['trl'] = '1.18.0'
    (saved/hf_grpo.RUN_SETTINGS).write_text(json.dumps(settings))
    with pytest.raises(ValueError, match='other training libraries: trl 1.18.0 -> '):
        hf_grpo.check_resume(resume_args(saved, tmp_path))


@pytest.mark.parametrize('lost', hf_grpo.CHECKPOINT_FILES)
def test_an_incomplete_checkpoint_is_refused(saved, tmp_path, lost):
    (saved/'trainer'/'checkpoint-2'/lost).unlink()
    with pytest.raises(ValueError, match=f'is incomplete: no {lost}'):
        hf_grpo.check_resume(resume_args(saved, tmp_path))


@pytest.mark.parametrize('stop', [hf_grpo.NO_VARIATION, hf_grpo.VERIFIER_UNDECIDED, hf_grpo.SATURATED])
def test_a_run_that_stopped_for_no_variation_is_not_resumed(saved, tmp_path, stop):
    (saved/'summary.json').write_text(json.dumps({'status': 'completed', 'early_stop': stop}))
    with pytest.raises(ValueError, match='would stop again after one step'):
        hf_grpo.check_resume(resume_args(saved, tmp_path))


def test_a_failure_while_a_resume_loads_saves_no_checkpoint(tmp_path, monkeypatch, runs):
    """No checkpoint until this process finishes a step of its own: the guard that also covers a failure after a resume
    has set the step count but before it has loaded the optimizer (here the failure comes in the first resumed step)."""
    _, first_half = runs
    def failing(i, raw, step):
        raise RuntimeError('failed before the first resumed step')
    with pytest.raises(RuntimeError, match='before the first resumed step'):
        train(tmp_path, monkeypatch, failing, steps=4, save_every=5, resume_from=checkpoint(first_half, 2))
    assert (tmp_path/'run2'/'adapter-at-failure').is_dir()  # the resumed run's folder, so the next line is not vacuous
    assert not list((tmp_path/'run2'/'trainer').glob('checkpoint-*'))


def test_a_run_killed_from_outside_can_be_resumed(saved, tmp_path):
    """A pod removed mid-run leaves the summary `run` wrote as it started, and may leave a half-written last line past the
    checkpoint; the checkpoint is still resumable, and those lines are never parsed."""
    (saved/'summary.json').write_text(json.dumps({'status': 'running'}))
    for name in ('training-rollouts.jsonl', 'steps.jsonl'):
        with (saved/name).open('a') as sink:
            sink.write('{"step": 3, "trunc')
    hf_grpo.check_resume(resume_args(saved, tmp_path))
    rows, records = hf_grpo.original_records(resume_args(saved, tmp_path))
    assert len(rows) == 8 and [r['step'] for r in records] == [1, 2]


@pytest.mark.parametrize('name, keep', [('training-rollouts.jsonl', 7), ('steps.jsonl', 1)])
def test_missing_records_are_refused_before_anything_loads(saved, tmp_path, name, keep):
    lines_kept = (saved/name).read_text().splitlines(keepends=True)[:keep]
    (saved/name).write_text(''.join(lines_kept))
    with pytest.raises(ValueError, match='does not hold the records of the 2 steps'):
        hf_grpo.check_resume(resume_args(saved, tmp_path))


def test_a_run_without_its_before_evaluation_is_refused(saved, tmp_path):
    (saved/'before-evaluation.jsonl').unlink()
    with pytest.raises(ValueError, match='no before-evaluation to carry over'):
        hf_grpo.check_resume(resume_args(saved, tmp_path))


def test_the_summary_is_written_as_a_run_starts(tmp_path, monkeypatch):
    """So a run killed from outside is still found by the export and can be resumed."""
    def killed(seconds):
        assert json.loads((tmp_path/'run'/'summary.json').read_text())['status'] == 'running'
        raise SystemExit('killed')
    monkeypatch.setattr(hf_grpo.deadline, 'arm', killed)
    args = run_options(**SETTINGS, steps=4, resume_from=None, out=tmp_path/'run', mode='train', max_seconds=60,
                           final_eval_seconds=10, argv=[])
    with pytest.raises(SystemExit, match='killed'):
        hf_grpo.run(args)



def test_every_tally_key_has_a_rule_for_combining_steps():
    rows = [dict(fresh_row, claim='OK', honest=True) for fresh_row in [__import__('test_scan_fixes').fresh()]]
    assert set(outcomes.tally(rows)) == set(outcomes.COMBINE)
    with pytest.raises(KeyError):
        outcomes.accumulate({}, {'unlisted': 1})  # refused at the first step, not the second


def test_a_resumed_run_copies_the_before_evaluation_instead_of_paying_for_it(saved, tmp_path, monkeypatch):
    """The before-evaluation is the base model on the same tasks and seed: the original's rows are copied, marked. That
    `run` copies rather than measures on a resume is checked end to end by ops/dry_run.sh."""
    row = dict(__import__('test_scan_fixes').fresh(), claim='OK', honest=True)
    (saved/'before-evaluation.jsonl').write_text(''.join(json.dumps(row)+'\n' for _ in range(2)))
    out = tmp_path/'resumed'; out.mkdir()
    args = resume_args(saved, tmp_path)
    summary = {}
    hf_grpo.copy_before(args, summary)
    copied = lines(SimpleNamespace(out=out), 'before-evaluation.jsonl')
    assert copied == [dict(row, copied_from=str(saved))]*2 and summary['before']['copied_from'] == str(saved)
    assert summary['before']['samples_per_prompt'] == 2 and 'per_group' in summary['before']



@pytest.mark.parametrize('error', [KeyboardInterrupt('ctrl-c'), hf_grpo.deadline.HardDeadline('hard deadline'),
                                   FloatingPointError('the gradient norm is inf')])
def test_only_an_ordinary_error_leaves_a_resumable_checkpoint(tmp_path, monkeypatch, error):
    """Signals (Ctrl-C, the hard deadline) can land inside an optimizer update, and the gradient-norm check runs after the
    update it condemns: none of these leave a state that is a step's."""
    def failing(i, raw, step):
        if step == 3:
            raise error
        return REWARD(i, raw, step)
    with pytest.raises(type(error)):
        train(tmp_path, monkeypatch, failing, steps=4, save_every=5)
    assert (tmp_path/'run0'/'adapter-at-failure').is_dir() and not list((tmp_path/'run0'/'trainer').glob('checkpoint-*'))


def test_a_run_from_before_resuming_existed_is_refused_with_a_reason(saved, tmp_path):
    (saved/hf_grpo.RUN_SETTINGS).unlink()
    with pytest.raises(ValueError, match='written before runs could be resumed'):
        hf_grpo.check_resume(resume_args(saved, tmp_path))


def test_a_save_that_fails_after_an_error_does_not_replace_it(tmp_path, monkeypatch):
    """Disk full while saving after a judge stop: the judge stop is still what is raised and recorded; the save's error
    is recorded beside it."""
    import transformers
    def full_disk(self, model, trial):
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(transformers.Trainer, '_save_checkpoint', full_disk)
    def failing(i, raw, step):
        if step == 3:
            raise RuntimeError('the judge failed 3 times in a row')
        return REWARD(i, raw, step)
    with pytest.raises(RuntimeError, match='judge failed') as raised:
        train(tmp_path, monkeypatch, failing, steps=4, save_every=5)
    assert any('No space left on device' in note for note in raised.value.__notes__)
