"""Independent review regressions at public pilot trust boundaries."""
import pytest

from rewardhack import pilot
from rewardhack.pilot_simulation import example_spec, synthetic_rows


@pytest.fixture
def spec(tmp_path):
    return pilot.load_spec(example_spec(tmp_path/'independent-demo'))


@pytest.mark.parametrize('change', [
    lambda row: row.update(spec_hash='0'*64),
    lambda row: row.update(baseline_compute=0),
    lambda row: row['quality'].update(valid='false'),
    lambda row: row['quality'].update(invalid_rate=float('nan')),
    lambda row: row['effects'].update(undeclared=dict(mean=0., se=.1)),
])
def test_direct_decision_api_cannot_bypass_evidence_validation(spec, change):
    rows = synthetic_rows(spec, 1, 'null')
    change(rows[0])
    with pytest.raises(ValueError):
        pilot.decide(spec, rows)


def test_direct_decision_cannot_reuse_sources_under_another_seed(spec):
    rows = synthetic_rows(spec, 1, 'null')
    rows[1]['sources'] = rows[0]['sources']
    with pytest.raises(ValueError, match='reused'):
        pilot.decide(spec, rows)


def test_numerically_duplicate_checkpoints_cannot_pass_distinct_json_hashes(spec):
    spec['checkpoints'][1] = 1.0
    spec['stages'][1]['checkpoint'] = 1.0
    with pytest.raises(ValueError):
        pilot.validate_spec(spec)


def test_endpoint_cannot_overwrite_forecast_checkpoint_metadata(spec):
    spec['endpoints'][0]['name'] = 'checkpoint'
    with pytest.raises(ValueError):
        pilot.validate_spec(spec)
