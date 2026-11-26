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
    limits = [(number, int(match.group(1))) for number, line in enumerate(lines) if (match := re.match(r'[A-Z_]+_SECONDS=(\d+)$', line))]
    assert limits
    for number, seconds in limits:
        cap = gates.CAP_COMMENT.search(lines[number-1])
        assert cap, lines[number]
        multiple, kind, amount, unit = float(cap.group(1)), cap.group(2), float(cap.group(3)), cap.group(4)
        assert multiple >= gates.MULTIPLES[kind] and seconds >= multiple * amount * gates.UNIT_MINUTES[unit] * 60, lines[number]
