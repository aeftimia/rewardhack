"""GPU memory and utilisation sampling for run summaries."""
import subprocess
import threading
import time


class GpuSampler(threading.Thread):
    """Peak memory across every GPU, including what torch's allocator does not report."""

    def __init__(self):
        super().__init__(daemon=True)
        self.peak_mib, self.max_util, self.running, self.error = 0, 0, True, None

    def run(self):
        while self.running:
            try:
                out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True, timeout=10)
                rows = [[float(x) for x in line.split(',')] for line in out.strip().splitlines()]
                self.peak_mib = max(self.peak_mib, sum(r[0] for r in rows))
                self.max_util = max(self.max_util, max(r[1] for r in rows))
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.error = f'{type(error).__name__}: {error}'[:200]  # reported, never replaced by zeros
                return
            time.sleep(2)

    def summary(self):
        self.running = False
        return dict(gpu_peak_mib=self.peak_mib, gpu_max_util=self.max_util, gpu_sampler_error=self.error)
