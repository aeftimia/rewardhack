"""Project a stage's finish time from observed completed work."""
import math
from statistics import fmean

from scipy.stats import t

MIN_SAMPLES = 7


class ProgressProjection:
    """Update a locally linear timing fit and return its upper finish estimate."""

    def __init__(self, target, confidence=0.95):
        if not 0 < confidence < 1 or target <= 0:
            raise ValueError('confidence must be between zero and one, and target must be positive')
        self.target = target
        self.confidence = confidence
        self.samples = []

    def update(self, work, elapsed):
        """Return seconds since start, or None until seven readings permit a stable fit."""
        if work < 0 or elapsed < 0 or (self.samples and (work < self.samples[-1][0] or elapsed < self.samples[-1][1])):
            raise ValueError('progress and elapsed time must be nonnegative and nondecreasing')
        self.samples.append((work, elapsed))
        if len(self.samples) < MIN_SAMPLES:
            return None
        work, elapsed = zip(*self.samples, strict=True)
        xbar, ybar = fmean(work), fmean(elapsed)
        spread = sum((x-xbar)**2 for x in work)
        if spread == 0:
            return None
        slope = sum((x-xbar)*(y-ybar) for x, y in self.samples) / spread
        if slope <= 0:
            return None
        intercept = ybar - slope*xbar
        residual = sum((y-intercept-slope*x)**2 for x, y in self.samples)
        error = math.sqrt(residual / (len(self.samples)-2))
        predictive = error * math.sqrt(1 + 1/len(self.samples) + (self.target-xbar)**2/spread)
        return intercept + slope*self.target + t.ppf(self.confidence, len(self.samples)-2)*predictive
