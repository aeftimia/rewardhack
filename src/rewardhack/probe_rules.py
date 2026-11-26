"""What the pod script reads from a probe's summary. Run: python -m rewardhack.probe_rules <command> <summary.json>

`rules`: the owner's two rules on a finished trainer probe. Prints `<token limit to train with> <1 if the verifier needs a
look, else 0> <a line for the log>`. A rule the probe could not evaluate is None in the summary; that is not a "no", and
this exits with an error instead of printing.

`stage`: one line for the log on a vLLM probe stage that failed: whether it was stopped at its own --max-seconds (its
finished chunks are saved) or ended otherwise, and how many rows it wrote (review round 30).
"""
import json
import sys
from pathlib import Path

from .deadline import HardDeadline


def decide(probe):
    if not isinstance(probe['raise_token_limit'], bool) or not isinstance(probe['check_verifier'], bool):
        raise ValueError('a probe rule is undetermined: the probe has no rows to judge it by')  # noqa: TRY004  a missing (None) rule is a value problem, and callers catch ValueError
    line = f"tasks {probe['tasks_completed']} of {probe['tasks_planned']} cut-off {probe['truncated_rate']} undecided {probe['unverified_rate']}"
    return 4096 if probe['raise_token_limit'] else 2048, int(probe['check_verifier']), line


def rules(path):
    return ' '.join(map(str, decide(json.loads(Path(path).read_text())['probe'])))


def stage(path):
    """A failed vLLM probe stage, from the summary vllm_probe writes at its start and again at its end."""
    if not Path(path).is_file():
        return 'no summary.json: it ended before writing one'
    summary = json.loads(Path(path).read_text())
    if 'rows_written' not in summary:  # written only by the final summary
        return f"status={summary['status']}: killed before its final summary, so rows written is unknown (count rows.jsonl)"
    said = f"status={summary['status']} rows_written={summary['rows_written']}"
    if 'error' not in summary or summary['error'] is None:
        return f'{said}; no error recorded'
    stopped = summary['error'].startswith(f'{HardDeadline.__name__}:')
    return f"{said} {'stopped at its --max-seconds' if stopped else 'ended by an error'}: {summary['error']}"


COMMANDS = {'rules': rules, 'stage': stage}

if __name__ == '__main__':
    command, path = sys.argv[1:]
    print(COMMANDS[command](path))
