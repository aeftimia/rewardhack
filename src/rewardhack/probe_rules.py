"""The owner's two rules on a finished probe, read from its summary for the pod script. Run: python -m rewardhack.probe_rules <summary.json>

Prints `<token limit to train with> <1 if the verifier needs a look, else 0> <a line for the log>`. A rule the probe
could not evaluate is None in the summary; that is not a "no", and this exits with an error instead of printing.
"""
import json
import sys


def decide(probe):
    if not isinstance(probe['raise_token_limit'], bool) or not isinstance(probe['check_verifier'], bool):
        raise ValueError('a probe rule is undetermined: the probe has no rows to judge it by')  # noqa: TRY004  a missing (None) rule is a value problem, and callers catch ValueError
    line = f"tasks {probe['tasks_completed']} of {probe['tasks_planned']} cut-off {probe['truncated_rate']} undecided {probe['unverified_rate']}"
    return 4096 if probe['raise_token_limit'] else 2048, int(probe['check_verifier']), line


if __name__ == '__main__':
    print(*decide(json.load(open(sys.argv[1]))['probe']))
