"""Shared CLI configuration for bounded verifier retries."""
import argparse

from .budget import nonnegative_finite
from .outcomes import RETRY_SECONDS, RETRY_TIMEOUT


def retry_seconds(value):
    """Reject invalid allowances during argument parsing, before loading a model."""
    try:
        return nonnegative_finite(value, 'verifier retry seconds')
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


def add_arguments(parser):
    parser.add_argument('--verifier-retry-seconds', type=retry_seconds, default=RETRY_SECONDS,
                        help=f'allowance per phase (probe, each evaluation/monitor, training) for second, '
                             f'{RETRY_TIMEOUT} s verifier checks after a timeout; 0 disables retries')
