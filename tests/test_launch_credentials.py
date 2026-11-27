"""Credential failures must be found before deployment, without leaking values."""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from rewardhack import launch_credentials as credentials


@pytest.mark.parametrize('status', [401, 403, 429, 500, 302])
def test_judge_refusal_is_sanitized_and_does_not_redirect(monkeypatch, status):
    class Client:
        def __init__(self, **kwargs):
            assert kwargs['timeout'] > 0 and kwargs['follow_redirects'] is False
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def get(self, url, headers):
            assert headers == {'Authorization': 'Bearer test-only-secret'}
            return httpx.Response(status, text='test-only-secret')
    monkeypatch.setattr(credentials.httpx, 'Client', Client)
    with pytest.raises(credentials.CredentialError, match=f'HTTP {status}') as error:
        credentials.check_judge({'OPENAI_API_KEY': 'test-only-secret'})
    assert 'test-only-secret' not in str(error.value)


def test_missing_judge_key_stops_before_archive_mutation(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr(credentials, 'check_archive', lambda: pytest.fail('archive should not run'))
    with pytest.raises(credentials.CredentialError, match='missing'):
        credentials.preflight({'FORWARD_JUDGE_KEY': '1'})


@pytest.mark.parametrize('failure', [None, 'put-object', 'get-object', 'readback', 'delete-object'])
def test_archive_checks_roundtrip_and_cleans_up_even_after_failure(monkeypatch, failure):
    calls, objects = [], {}
    def aws(*args):
        operation = args[0]
        calls.append(operation)
        key = args[args.index('--key')+1]
        if operation == failure:
            raise credentials.CredentialError(f'{operation} denied')
        if operation == 'put-object':
            objects[key] = Path(args[args.index('--body')+1]).read_bytes()
        elif operation == 'get-object':
            Path(args[-1]).write_bytes(b'wrong' if failure == 'readback' else objects[key])
    monkeypatch.setattr(credentials, 'aws', aws)
    if failure is None:
        credentials.check_archive()
    else:
        with pytest.raises(credentials.CredentialError):
            credentials.check_archive()
    assert calls[-1] == 'delete-object'
    assert calls[0] == 'put-object'


def test_archive_reports_both_primary_and_cleanup_failure(monkeypatch):
    def aws(*args):
        raise credentials.CredentialError(f'{args[0]} denied')
    monkeypatch.setattr(credentials, 'aws', aws)
    with pytest.raises(credentials.CredentialError, match='put-object denied.*cleanup failed.*delete-object denied'):
        credentials.check_archive()


def test_aws_cli_errors_do_not_echo_credential_details(monkeypatch):
    monkeypatch.setattr(credentials.subprocess, 'run', lambda *a, **kw:
                        SimpleNamespace(returncode=1, stdout='', stderr='secret-provider-details'))
    with pytest.raises(credentials.CredentialError) as error:
        credentials.aws('get-object')
    assert 'secret-provider-details' not in str(error.value)


def test_receipt_expires_and_binds_the_judge_key():
    settings, environment = {'FORWARD_JUDGE_KEY': '1'}, {'OPENAI_API_KEY': 'test-only-secret'}
    receipt = dict(judge_sha256=credentials.judge_fingerprint(settings, environment),
                   archive_bucket=credentials.BUCKET, archive_region=credentials.REGION,
                   checked_at=datetime.now(timezone.utc).isoformat())
    assert credentials.receipt_problems(receipt, settings, environment) == []
    assert 'differs' in credentials.receipt_problems(receipt, settings, {'OPENAI_API_KEY': 'changed'})[0]
    receipt['checked_at'] = (datetime.now(timezone.utc)-timedelta(seconds=credentials.RECEIPT_MAX_AGE_SECONDS+1)).isoformat()
    assert 'stale' in credentials.receipt_problems(receipt, settings, environment)[0]
