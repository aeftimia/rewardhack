"""Every time limit in the pod scripts states its basis at the multiples the launch record uses (gates.MULTIPLES).
This was a rule of the global no-silent-fallbacks hook; the owner moved it here on 26 Nov, so the sizing rule lives once."""
from pathlib import Path

import pytest

from rewardhack import gates

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script', sorted(path.name for path in (ROOT/'ops').glob('*.sh')))
def test_every_time_limit_in_the_pod_scripts_states_its_basis(script):
    assert gates.unsized_limits((ROOT/'ops'/script).read_text()) == []


@pytest.mark.parametrize('text, said', [
    ('timeout 1500 python bench.py\n', 'no stated basis'),
    ('timeout 1500 python bench.py  # cap: 1.5x measured 750 s\n', 'at least 2x'),
    ('timeout 1500 python bench.py  # cap: 1.4x estimated 750 s\n', 'at least 1.5x'),
    ('DEADLINE=19  # cap: 2x measured 17 min\n', 'is under 2 x 17 min'),
    ('run --wait "${READY_WAIT:-180}"  # cap: 1.5x estimated 390 s\n', 'is under'),
])
def test_a_limit_without_a_sufficient_basis_is_found(text, said):
    [(line, why)] = gates.unsized_limits(text)
    assert line == 1 and said in why


def test_a_basis_on_the_line_above_or_at_the_multiples_passes():
    assert gates.unsized_limits('# cap: 2x measured 750 s\ntimeout 1500 python bench.py\n'
                                'READY_WAIT=600  # cap: 1.5x estimated 390 s\nMAX_TRIES=3\n') == []


def test_the_multiples_are_stated_only_in_gates():
    """One source for the cap multiples (owner, 26 Nov): a copy in the instructions went stale once and a reviewer held a
    correct cap to the old rule. CLAUDE.md (and AGENTS.md, which defers to it) point to gates.MULTIPLES instead."""
    import re
    for name in ('CLAUDE.md', 'AGENTS.md'):
        text = (ROOT/name).read_text()
        assert not re.search(r'\d+(?:\.\d+)?x\s+(?:the\s+)?(?:an?\s+)?(?:its\s+)?(?:longest\s+)?(measured|estimate)', text), name
    assert 'gates.MULTIPLES' in (ROOT/'CLAUDE.md').read_text()


def test_every_time_limit_in_the_pod_pins_states_its_basis():
    """The gpt-oss-120b probe's limits live in ops/pod_pins.env, read by the smoke script and the driver, under names the
    script scan does not match (review round 25): each `*_SECONDS=` there states its basis on the line above."""
    import re
    lines = (ROOT/'ops/pod_pins.env').read_text().splitlines()
    # Setup allowances are in minutes (MODEL_SETUP_MINUTES, notes section 80), read in seconds here.
    named = [number for number, line in enumerate(lines) if re.match(r'[A-Z0-9_]+_(?:SECONDS|MINUTES)=', line)]
    per_unit = {'SECONDS': 1, 'MINUTES': 60}
    limits = [(number, int(match.group(2)) * per_unit[match.group(1)]) for number in named
              if (match := re.match(r'[A-Z0-9_]+_(SECONDS|MINUTES)=(\d+)\b', lines[number]))]
    assert limits and len(limits) == len(named), 'every *_SECONDS or *_MINUTES in ops/pod_pins.env is a plain whole number'
    for number, seconds in limits:
        # Its own comment line, not the trailing comment of an assignment above (which would lend it another's basis).
        cap = gates.CAP_COMMENT.search(lines[number-1]) if number > 0 and lines[number-1].startswith('#') else None
        assert cap, lines[number]
        multiple, kind, amount, unit = float(cap.group(1)), cap.group(2), float(cap.group(3)), cap.group(4)
        assert multiple >= gates.MULTIPLES[kind] and seconds >= multiple * amount * gates.UNIT_MINUTES[unit] * 60, lines[number]


def test_the_pod_pins_have_no_unchecked_limit_and_stage_times_name_checked_ones():
    """Round 28: a number typed into PROBE_LARGE_STAGES, or a limit named *_TIMEOUT or *_WAIT in ops/pod_pins.env, escaped
    both this file's checks. The script scan runs over the pins too, and every stage's time is one of the checked limits."""
    import re
    import subprocess
    text = (ROOT/'ops/pod_pins.env').read_text()
    assert gates.unsized_limits(text) == []
    stages = subprocess.run(['bash', '-c', 'source ops/pod_pins.env; printf "%s\\n" "${PROBE_LARGE_STAGES[@]}"'], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout.split('\n')[:-1]
    checked = {int(value) for value in re.findall(r'^[A-Z_]+_SECONDS=(\d+)\b', text, re.M)}
    assert stages and all(int(stage.split()[1]) in checked for stage in stages), stages
    raw = re.search(r'^PROBE_LARGE_STAGES=\((.*)\)$', text, re.M).group(1)
    assert not re.search(r'"\w+ \d', raw), 'a stage time is written as a number; name a *_SECONDS limit instead'
