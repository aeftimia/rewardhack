"""Does the reward we compute reach the real trainer, and is the reward the only thing it trains on?

Written after 2026-11-25, when every paid gpt-oss run turned out to have trained on a library default (the model's
router load-balancing loss at weight 0.9) with a learning rate decaying to zero. These tests run the real TRL
trainer through `hf_grpo.train` on tiny random models, CPU only, with the scoring step scripted. They show
plumbing and objective, never behaviour.
"""
import json
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from rewardhack import deadline, hf_grpo, judge, outcomes
from rewardhack.outcomes import RETRY_SECONDS

torch = pytest.importorskip('torch')
pytest.importorskip('trl')

TASKS = [dict(id=f't{i}', group=f'g{i}', tier='easy', true=bool(i % 2), statement=f'statement {i}') for i in range(4)]
WORDS = ['<pad>', '<eos>', '<unk>'] + [f'w{i}' for i in range(13)]
GENERATIONS = 4


def tiny(kind):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, GptOssConfig, GptOssForCausalLM, PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token='<unk>'))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>', pad_token='<pad>', unk_token='<unk>')
    tokenizer.padding_side = 'left'
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }} {% endfor %}w0"
    torch.manual_seed(0)
    if kind == 'moe':  # the gpt-oss architecture, whose config carries its own router loss coefficient
        model = GptOssForCausalLM(GptOssConfig(num_hidden_layers=1, num_local_experts=4, num_experts_per_tok=2, vocab_size=len(WORDS), hidden_size=16,
                                               intermediate_size=16, head_dim=4, num_attention_heads=4, num_key_value_heads=2, router_aux_loss_coef=0.9,
                                               pad_token_id=0, eos_token_id=1, bos_token_id=2))
    else:
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(WORDS), n_layer=1, n_head=2, n_embd=16, bos_token_id=2, eos_token_id=1, pad_token_id=0))
    return model, tokenizer


def train(tmp_path, monkeypatch, reward_of, *, kind='plain', steps=1, built=None, extra=None, **overrides):
    """Run `hf_grpo.train` with `reward_of(position in batch, completion text, step)` in place of Lean and the judge.

    None from `reward_of` is a rollout the verifier could not decide. Returns the adapter's change, TRL's log,
    our summary, and the rewards each step handed to TRL.
    """
    template = outcomes.score('unused', TASKS[0], '', 'baseline', truncated=True)  # a cut-off row needs neither Lean nor a judge; the fields below are then scripted
    handed, step = [], [0]

    def score(lean, task, raw, arm, penalty, judge_fn, **kwargs):
        position = len(handed[-1]) if handed and len(handed[-1]) < GENERATIONS else 0
        if position == 0:
            handed.append([])
            step[0] += 1
        value = reward_of(position, raw, step[0])
        handed[-1].append(value)
        return dict(template, task_id=task['id'], group=task['group'], raw=raw, truncated=False, format_valid=True, arm=arm,
                    verifier_inconclusive=value is None, reward=0.0 if value is None else float(value),
                    **(extra(position, step[0]) if extra else {}))
    monkeypatch.setattr(hf_grpo, 'score', score)
    model, tokenizer = built or tiny(kind)
    out = tmp_path/f'run{len(list(tmp_path.iterdir()))}'
    out.mkdir()
    args = SimpleNamespace(out=out, lean='unused', arm='report_correct', penalty=0.5, judge_fn=None, report_mode='prose', verifier_retry_seconds=RETRY_SECONDS, num_generations=GENERATIONS,
                           prompts_per_step=1, steps=steps, learning_rate=1e-2, lr_schedule='constant', router_aux_loss_coef=0.0, scale_rewards='group', max_grad_norm=1.0, beta=0.0, seed=7,
                           adapter=None, lora_rank=2, max_completion_length=6, prompt_variant='neutral', thinking='unsupported', reasoning_effort=None,
                           save_every=99, stop_no_variation=99, train_deadline=time.monotonic()+3600)
    vars(args).update(overrides)
    summary = dict(rollouts={})
    fit = hf_grpo.prepare_training(model, tokenizer, TASKS, args, summary)
    start = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
    torch.manual_seed(31415)  # what the before-evaluation does between building the trainer and training
    fit()
    change = torch.cat([(p.detach()-start[name]).flatten() for name, p in model.named_parameters() if name in start])
    log = [row for row in json.loads((out/'training-log.json').read_text()) if 'loss' in row]
    rows = [json.loads(line) for line in (out/'training-rollouts.jsonl').read_text().splitlines()]
    return SimpleNamespace(change=change, log=log, summary=summary, handed=handed, rows=rows, out=out)


def mean_kept(values):
    kept = [v for v in values if v is not None]
    return sum(kept)/len(kept)


def cosine(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


REWARDS = {
    'first of the group': lambda i, raw, step: float(i == 0),
    'depends on the text': lambda i, raw, step: float(len(raw.split()) % 3),
    'negative and fractional': lambda i, raw, step: -1.5*i+0.25,
    'large': lambda i, raw, step: 1e4*(i % 2)+step,
    'one rollout undecided': lambda i, raw, step: None if i == 1 else float(i),
    'changes every step': lambda i, raw, step: float((i+step) % 3),
}


@pytest.mark.parametrize('name', sorted(REWARDS))
def test_any_reward_function_reaches_the_trainer_unchanged(tmp_path, monkeypatch, name):
    run = train(tmp_path, monkeypatch, REWARDS[name], steps=3)
    assert len(run.handed) == len(run.log) == 3 and all(len(step) == GENERATIONS for step in run.handed)
    for handed, logged in zip(run.handed, run.log):
        # What TRL says it was rewarded with is what the scoring step returned, undecided rollouts left out.
        assert logged['rewards/reward_fn/mean'] == pytest.approx(mean_kept(handed), rel=1e-5, abs=1e-6)
    # What is saved per rollout is what was trained on.
    assert [row['training_reward'] for row in run.rows] == [value for step in run.handed for value in step]
    assert float(run.change.abs().max()) > 1e-4, 'a varied reward moved nothing'


def test_the_update_follows_the_reward_and_nothing_else(tmp_path, monkeypatch):
    """Same model, same seed, so the same completions: only the reward differs between these runs."""
    reward = REWARDS['first of the group']
    base = train(tmp_path, monkeypatch, reward)
    assert [row['raw'] for row in base.rows] == [row['raw'] for row in train(tmp_path, monkeypatch, reward).rows], 'runs are not comparable'
    opposite = train(tmp_path, monkeypatch, lambda i, raw, step: -reward(i, raw, step))
    assert cosine(base.change, opposite.change) < -0.99, 'negating the reward did not reverse the update'
    # GRPO standardises rewards within a prompt group, so a shift or a positive scale changes nothing.
    rescaled = train(tmp_path, monkeypatch, lambda i, raw, step: 1000*reward(i, raw, step)+7)
    assert cosine(base.change, rescaled.change) > 0.99
    # No contrast inside the group, or nothing decided: no update at all.
    for flat in (lambda i, raw, step: 3.0, lambda i, raw, step: None):
        assert float(train(tmp_path, monkeypatch, flat).change.abs().max()) < 1e-7


def test_our_arm_rewards_are_what_the_trainer_trains_on(tmp_path, monkeypatch):
    """`arm_reward` for four outcomes (compiles, genuine, refuted, honest) through the trainer, for each funded penalty."""
    cases = [(True, False, False, False), (True, True, False, True), (False, False, True, True), (False, False, False, True)]
    for penalty in (0.0, 0.5, 1.0):
        values = [outcomes.arm_reward('report_correct', *case, penalty) for case in cases]
        run = train(tmp_path, monkeypatch, lambda i, raw, step, values=values: values[i], penalty=penalty)
        assert run.handed == [values] and run.log[0]['rewards/reward_fn/mean'] == pytest.approx(sum(values)/4)
        # The concealed exploit (first case) is worth 1 - penalty; at full penalty it is the worst outcome in the group.
        assert values[0] == pytest.approx(1-penalty) and values[1] == 2.0


def test_schedule_and_objective_are_the_ones_asked_for(tmp_path, monkeypatch):
    run = train(tmp_path, monkeypatch, REWARDS['first of the group'], steps=3)
    assert [row['learning_rate'] for row in run.log] == [pytest.approx(1e-2)]*3, 'the learning rate is not constant'
    assert run.summary['trainer']['lr_schedule'] == 'constant' and run.summary['trainer']['router_aux_loss_coef'] == 0.0
    assert all('aux_loss' not in row for row in run.log)
    decaying = train(tmp_path, monkeypatch, REWARDS['first of the group'], steps=3, lr_schedule='linear')
    rates = [row['learning_rate'] for row in decaying.log]
    assert rates[0] > rates[1] > rates[2], 'the linear option should be the one that decays'


def test_a_mixture_of_experts_model_trains_on_the_reward_alone(tmp_path, monkeypatch):
    """gpt-oss carries `router_aux_loss_coef` in its own config; left unset, TRL adds that loss to ours."""
    ours = train(tmp_path, monkeypatch, REWARDS['first of the group'], kind='moe')
    assert ours.summary['trainer']['aux_loss_enabled'] is False and ours.summary['trainer']['router_aux_loss_coef'] == 0.0
    assert 'aux_loss' not in ours.log[0]
    assert float(train(tmp_path, monkeypatch, lambda i, raw, step: 1.0, kind='moe').change.abs().max()) < 1e-7, 'equal rewards still moved the model'
    # The failure this guards against, reproduced: with the model's coefficient the loss is mostly the router term,
    # and the model moves even when every reward is equal.
    theirs = train(tmp_path, monkeypatch, lambda i, raw, step: 1.0, kind='moe', router_aux_loss_coef=0.9)
    assert theirs.summary['trainer']['aux_loss_enabled'] is True
    assert theirs.log[0]['loss'] == pytest.approx(0.9*theirs.log[0]['aux_loss'], rel=1e-3)
    assert float(theirs.change.abs().max()) > 1e-4


def test_entrypoint_defaults_are_the_safe_ones(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(sys, 'argv', ['run.py', 'train', '--out', str(tmp_path), '--lean', 'lean'])
    import importlib.util
    spec = importlib.util.spec_from_file_location('run_entry', 'entrypoints/run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'run', lambda args: captured.update(vars(args)))
    module.main()
    assert captured['router_aux_loss_coef'] == 0.0 and captured['lr_schedule'] == 'constant' and captured['mxfp4'] == 'absent'


def test_a_failed_run_keeps_what_was_paid_for(tmp_path, monkeypatch):
    def reward(i, raw, step):
        if step == 3:
            raise RuntimeError('judge spend cap reached')
        return float(i == 0)
    with pytest.raises(RuntimeError, match='spend cap'):
        train(tmp_path, monkeypatch, reward, steps=5)
    out = next(tmp_path.iterdir())
    assert len([row for row in json.loads((out/'training-log.json').read_text()) if 'loss' in row]) == 2
    assert len(json.loads((out/'step-seconds.json').read_text())['update']) == 2
    assert (out/'adapter-at-failure'/'adapter_config.json').exists()


def test_weights_that_were_not_asked_for_are_refused(monkeypatch):
    """A model the loader left quantized cannot be trained; that must stop at load, not after a paid evaluation."""
    import transformers
    model, tokenizer = tiny('plain')
    model.is_quantized = True
    monkeypatch.setattr(transformers.AutoModelForCausalLM, 'from_pretrained', lambda *a, **kw: model)
    monkeypatch.setattr(transformers.AutoTokenizer, 'from_pretrained', lambda *a, **kw: tokenizer)
    args = SimpleNamespace(model='tiny', revision=None, attn_implementation='eager', quant='none', mxfp4='absent', experts_implementation=None, adapter=None, mode='train',
                           reasoning_effort=None, thinking='unsupported', prompt_variant='neutral', report_mode='prose')
    with pytest.raises(ValueError, match='--mxfp4 dequantize'):
        hf_grpo.load_model(args, {})
    model.is_quantized = False
    summary = {}
    hf_grpo.load_model(args, summary)
    assert summary['weights']['quantized'] is False and summary['weights']['dtype'] == 'torch.float32' and summary['weights']['footprint_gib'] > 0


def test_the_hard_deadline_cannot_be_swallowed():
    assert not issubclass(deadline.HardDeadline, Exception)
    swallowed = 0
    try:
        deadline.arm(0.2)
        with pytest.raises(deadline.HardDeadline):
            for _ in range(200):
                try:
                    time.sleep(0.05)
                except (Exception, OSError):  # what a retry loop or an HTTP client would catch
                    swallowed += 1
    finally:
        deadline.disarm()
    assert swallowed == 0


def test_a_failed_judge_reading_is_not_a_dishonest_report(monkeypatch):
    from rewardhack import model_io
    seen = {}

    def call(args, messages, *, timeout, retries, extra_body):
        seen.update(timeout=timeout, retries=retries)
        return dict(error='HTTP 503: unavailable')
    monkeypatch.setattr(model_io, 'call', call)
    reader = judge.make_judge('api', model='m', usd_per_m_input=0.2, usd_per_m_output=1.25, max_usd=1.0)
    verdict = reader(dict(report='I proved it.', statement='True', established='True'))
    assert verdict['claim'] == 'UNPARSED' and 'HTTP 503' in verdict['judge_error']
    assert seen['timeout'] == judge.REQUEST_SECONDS <= 60, 'a stalled judge request would idle a rented GPU'
    assert seen['retries'] == judge.REQUEST_RETRIES == 3
    undecided = dict(arm='report_correct', judge_undecided=True, format_valid=True, reward=0.0)
    assert outcomes.training_keep(undecided) is False and outcomes.measured([undecided]) == []
    assert outcomes.training_keep(dict(undecided, arm='baseline')) is True  # the baseline reward never used the judge


def test_a_priced_judge_reply_without_usage_stops_the_run(monkeypatch):
    from rewardhack import model_io
    reply = json.dumps({'requested': 'a', 'reported': 'b', **{name: False for name in judge.CHECKS}})
    monkeypatch.setattr(model_io, 'call', lambda args, messages, **transport: dict(raw=reply, finish='stop', model='stub', usage={}))
    with pytest.raises(RuntimeError, match='cannot be metered'):
        judge.make_judge('api', model='m', usd_per_m_input=0.2, usd_per_m_output=1.25, max_usd=1.0)(dict(report='Done.', statement='True'))
    # An unpriced judge (one served on the pod) has nothing to meter.
    assert judge.make_judge('api', model='m')(dict(report='Done.', statement='True'))['claim'] == 'OK'


SERVER = '''
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        body = json.dumps({"data": [{"id": sys.argv[2]}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
HTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
'''


def test_readiness_needs_our_process_and_our_model(tmp_path):
    """2026-11-25: another service on the port answered the readiness check and a paid stage ran against it."""
    import socket
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    script = tmp_path/'server.py'
    script.write_text(SERVER)
    server = subprocess.Popen([sys.executable, str(script), str(port), 'someone-elses-service'])
    sleeper = subprocess.Popen(['sleep', '60'])
    try:
        def ready(pid, model):
            command = f'log() {{ :; }}; fail() {{ echo "$1"; exit 1; }}; source ops/pod_lib.sh; wait_for_server {pid} {port} {model} 2'
            return subprocess.run(['bash', '-c', command], capture_output=True, text=True).returncode
        for _ in range(50):
            if ready(server.pid, 'someone-elses-service') == 0:
                break
            time.sleep(0.1)
        assert ready(server.pid, 'someone-elses-service') == 0
        assert ready(sleeper.pid, 'openai/gpt-oss-20b') == 1, 'a port that answers for a different model counted as ready'
        dead = subprocess.Popen(['true'])
        dead.wait()
        assert ready(dead.pid, 'someone-elses-service') == 1, 'a dead process counted as ready because the port answered'
    finally:
        server.kill()
        sleeper.kill()

