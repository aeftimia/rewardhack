"""A hard wall-clock limit for a paid process that nothing downstream can swallow."""
import signal


class HardDeadline(BaseException):
    """Not an `Exception` and not an `OSError`: HTTP clients and retry loops catch those, and one of them catching
    the deadline would leave a rented machine running with no limit at all."""


def arm(seconds):
    """Raise `HardDeadline` in the main thread after `seconds`, and again every 30 s in case one delivery was lost
    inside code that catches everything. `disarm()` belongs in the caller's `finally`."""
    def hard_stop(*unused):
        raise HardDeadline('hard deadline')
    signal.signal(signal.SIGALRM, hard_stop)
    signal.setitimer(signal.ITIMER_REAL, seconds, 30)


def disarm():
    signal.setitimer(signal.ITIMER_REAL, 0)
