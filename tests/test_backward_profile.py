from types import SimpleNamespace

import pytest
import torch

from rewardhack import gpu, jsonl
from rewardhack.backward_profile import instrument_backward


@pytest.mark.parametrize('fail', [False, True])
def test_backward_memory_includes_failed_calls_and_preserves_earlier_peak(tmp_path, monkeypatch, fail):
    peak = [90]
    cuda = SimpleNamespace(device_count=lambda:1, synchronize=lambda device:None,
                           max_memory_allocated=lambda device:peak[0], max_memory_reserved=lambda device:85,
                           memory_allocated=lambda device:10, memory_reserved=lambda device:20,
                           reset_peak_memory_stats=lambda device:peak.__setitem__(0,10))
    monkeypatch.setattr(torch, 'cuda', cuda)
    def backward(loss, **kwargs):
        assert peak[0] == 10 and kwargs == {'scale':1}
        peak[0] = 75
        if fail:
            raise RuntimeError('out of memory')
    trainer = SimpleNamespace(accelerator=SimpleNamespace(backward=backward),
                              state=SimpleNamespace(global_step=2), backward_shape={'completion_width':128})
    summary = {}
    instrument_backward(trainer, tmp_path, summary)
    loss = SimpleNamespace(device=SimpleNamespace(index=0))
    if fail:
        with pytest.raises(RuntimeError, match='out of memory'):
            trainer.accelerator.backward(loss, scale=1)
    else:
        trainer.accelerator.backward(loss, scale=1)
    row, = jsonl.read(tmp_path/'backward-memory.jsonl')
    assert row['step'] == 3 and row['completed'] is (not fail)
    assert row['peak_allocated_bytes'] == 75 and row['completion_width'] == 128
    assert summary['allocator_peaks_before_reset_bytes'] == [90]


def test_profile_records_each_gpu_and_keeps_aggregate_peak(tmp_path, monkeypatch):
    sampler = gpu.GpuSampler(tmp_path/'gpu-samples.jsonl')
    def smi(command, **kwargs):
        assert command[1] == '--query-gpu=index,memory.used,utilization.gpu'
        return '0, 65000, 70\n1, 82000, 92\n'
    monkeypatch.setattr(gpu.subprocess, 'check_output', smi)
    monkeypatch.setattr(gpu.time, 'sleep', lambda seconds: setattr(sampler, 'running', False))
    sampler.run()
    observation, = jsonl.read(sampler.samples)
    assert observation['cards'] == [dict(device=0, used_mib=65000, utilization_pct=70),
                                    dict(device=1, used_mib=82000, utilization_pct=92)]
    assert sampler.summary() == dict(gpu_peak_mib=147000, gpu_max_util=92, gpu_sampler_error=None)
