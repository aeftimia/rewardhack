"""Time the accelerator's backward call and retain per-microbatch CUDA memory peaks."""
from functools import wraps
import time

from . import jsonl


def instrument_backward(trainer, out, summary):
    import torch
    cuda = torch.cuda
    backward = trainer.accelerator.backward
    peaks = [cuda.max_memory_allocated(i) for i in range(cuda.device_count())]
    summary['allocator_peaks_before_reset_bytes'] = peaks
    path = out / 'backward-memory.jsonl'

    @wraps(backward)
    def measured(loss, **kwargs):
        device = loss.device.index
        cuda.synchronize(device)
        peaks[device] = max(peaks[device], cuda.max_memory_allocated(device))
        cuda.reset_peak_memory_stats(device)
        row = dict(step=trainer.state.global_step + 1, device=device, **trainer.backward_shape,
                   allocated_before_bytes=cuda.memory_allocated(device),
                   reserved_before_bytes=cuda.memory_reserved(device))
        start = time.perf_counter()
        completed = False
        try:
            result = backward(loss, **kwargs)
            cuda.synchronize(device)
            completed = True
            return result
        finally:
            row.update(completed=completed, seconds=time.perf_counter()-start,
                       peak_allocated_bytes=cuda.max_memory_allocated(device),
                       peak_reserved_bytes=cuda.max_memory_reserved(device))
            with path.open('a') as stream:
                jsonl.write(stream, row)

    trainer.accelerator.backward = measured
