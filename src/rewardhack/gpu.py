"""GPU memory and utilisation sampling for run summaries."""
import subprocess
import threading
import time
from . import jsonl


class GpuSampler(threading.Thread):
    """Peak memory across every GPU, including what torch's allocator does not report."""

    def __init__(self, samples=None):
        super().__init__(daemon=True)
        self.samples = samples
        self.peak_mib, self.max_util, self.running, self.error = 0, 0, True, None

    def run(self):
        while self.running:
            try:
                out = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True, timeout=10)
                cards = [dict(device=int(index), used_mib=float(memory), utilization_pct=float(util))
                         for index, memory, util in (line.split(',') for line in out.strip().splitlines())]
                self.peak_mib = max(self.peak_mib, sum(card['used_mib'] for card in cards))
                self.max_util = max(self.max_util, max(card['utilization_pct'] for card in cards))
                if self.samples is not None:
                    with self.samples.open('a') as sink:
                        jsonl.write(sink, dict(unix_seconds=time.time(), cards=cards))
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                self.error = f'{type(error).__name__}: {error}'[:200]  # reported, never replaced by zeros
                return
            time.sleep(2)

    def summary(self):
        self.running = False
        return dict(gpu_peak_mib=self.peak_mib, gpu_max_util=self.max_util, gpu_sampler_error=self.error)
