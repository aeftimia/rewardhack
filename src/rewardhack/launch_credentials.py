"""Authenticate standard experiment credentials before any paid pod is deployed.

Receipts contain no key values. RunPod injects its pod-scoped access only when
the container exists; the driver's first SSH action must still prove its timer.
"""
import hashlib
import os
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

from .archive import BUCKET, REGION, REQUEST_TIMEOUT
from .judge import API_JUDGE_MODEL, OPENAI_URL

RECEIPT_MAX_AGE_SECONDS = 1800


class CredentialError(RuntimeError):
    """A failed credential check; its message must not contain credential values."""


def judge_fingerprint(settings, environment):
    if 'FORWARD_JUDGE_KEY' not in settings or settings['FORWARD_JUDGE_KEY'] != '1':
        return None
    if 'OPENAI_API_KEY' not in environment or not environment['OPENAI_API_KEY']:
        raise CredentialError('OPENAI_API_KEY is missing; load the intended credential before ready, not after deployment')
    return hashlib.sha256(environment['OPENAI_API_KEY'].encode()).hexdigest()


def check_judge(environment):
    """A non-generating authenticated model lookup; neither headers nor response bodies are logged."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=False) as client:
            response = client.get(f'{OPENAI_URL}/models/{API_JUDGE_MODEL}',
                                  headers={'Authorization': 'Bearer '+environment['OPENAI_API_KEY']})
    except httpx.RequestError as error:
        raise CredentialError(f'OpenAI judge authentication could not connect ({type(error).__name__})') from None
    if response.status_code != 200:
        raise CredentialError(f'OpenAI judge authentication failed: HTTP {response.status_code}; '
                              'refresh the intended shell credential and rerun ready before deploying')
    if response.json()['id'] != API_JUDGE_MODEL:
        raise CredentialError('OpenAI returned a different judge model during credential preflight')


def aws(*arguments):
    try:
        result = subprocess.run(['aws', 's3api', *arguments, '--region', REGION,
                                 '--cli-connect-timeout', '15', '--cli-read-timeout', str(REQUEST_TIMEOUT)],
                                timeout=REQUEST_TIMEOUT, capture_output=True, text=True, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CredentialError(f'S3 {arguments[0]} check could not run ({type(error).__name__})') from None
    if result.returncode:
        # AWS diagnostics can contain credential/provider details. Report the operation and
        # exit status, never its stderr or the credential-bearing environment.
        raise CredentialError(f'S3 {arguments[0]} check failed (exit {result.returncode}); '
                              'verify the configured AWS login and archive permissions before deploying')


def check_archive():
    """Prove write/read/delete access with a unique, tiny, non-sensitive object."""
    key = f'credential-preflight/{uuid.uuid4().hex}'
    payload = b'rewardhack archive credential preflight\n'
    with tempfile.TemporaryDirectory(prefix='rewardhack-credentials-') as folder:
        source, fetched = Path(folder)/'source', Path(folder)/'fetched'
        source.write_bytes(payload)
        # Also clean up after an ambiguous put timeout: the object may already exist.
        failures = []
        try:
            aws('put-object', '--bucket', BUCKET, '--key', key, '--body', str(source))
            aws('get-object', '--bucket', BUCKET, '--key', key, str(fetched))
            if fetched.read_bytes() != payload:
                failures.append('S3 preflight readback differs from the uploaded payload')
        except (CredentialError, OSError) as error:
            failures.append(str(error))
        finally:
            try:
                aws('delete-object', '--bucket', BUCKET, '--key', key)
            except CredentialError as error:
                failures.append(f'preflight cleanup failed: {error}')
        if failures:
            raise CredentialError('; '.join(failures))


def preflight(settings):
    fingerprint = judge_fingerprint(settings, os.environ)
    if fingerprint is not None:
        check_judge(os.environ)
    check_archive()
    return dict(checked_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                judge_sha256=fingerprint, archive_bucket=BUCKET, archive_region=REGION,
                archive_access='write-read-delete', runpod_access='provider-injected; timer proof required on first SSH')


def receipt_problems(receipt, settings, environment):
    if receipt['judge_sha256'] != judge_fingerprint(settings, environment):
        return ['the judge key differs from the authenticated launch credential; rerun ready in the intended environment']
    if (receipt['archive_bucket'], receipt['archive_region']) != (BUCKET, REGION):
        return ['the archive destination differs from the credential preflight']
    # dry-ok: src/rewardhack/launch_credentials.py::preflight | test_receipt_expires_and_binds_the_judge_key constructs an older ISO timestamp independently and expects rejection. preflight can only capture when authentication succeeded. Checking age requires a second clock reading at launch; capturing it during authentication would always yield age zero.
    age = (datetime.now(timezone.utc)-datetime.fromisoformat(receipt['checked_at'])).total_seconds()
    if not 0 <= age <= RECEIPT_MAX_AGE_SECONDS:
        return ['credential preflight is stale or future-dated; rerun ready before deploying']
    return []
