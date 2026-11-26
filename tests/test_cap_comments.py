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
