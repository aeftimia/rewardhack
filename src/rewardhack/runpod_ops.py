"""RunPod REST v2 through the official SDK, with project launch checks."""
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from http import HTTPStatus
from pathlib import Path
from urllib.parse import quote

from . import gates
from .state_files import atomic_json, file_lock

API_SECONDS = gates.RUNPOD_API_SECONDS
POLL_SECONDS = 5
REMOVE_ATTEMPTS = 6
MAX_PAGES = 100
CREATE_FIELDS = {'name', 'image', 'gpu', 'disk', 'ports', 'cloud', 'dataCenterIds', 'mounts'}
# Conservative hourly rounding of $0.10/GB/month (docs.runpod.io/pods/pricing, checked 2026-11-27).
DISK_GB_HOURLY_USD = Decimal('0.00014')


def pod_path(pod_id):
    """One encoded REST resource path for pod reads, actions and removal."""
    return 'pods/' + quote(pod_id, safe='')


class RunPod:
    """Use one explicit local key; never print pod environments or credentials."""

    def __init__(self):
        from runpod.api.rest import run_rest_request
        self.transport = run_rest_request
        self.key = os.environ['RUNPOD_API_KEY']
        if not self.key:
            raise ValueError('RUNPOD_API_KEY is empty; source ~/.zshrc')
        if os.environ.get('RUNPOD_API_BASE_URL'):
            raise ValueError('RUNPOD_API_BASE_URL must be unset: this key is for RunPod only')

    def request(self, method, path, **kwargs):
        return self.transport(method, '/v2/' + path, api_key=self.key, timeout=API_SECONDS,
                              max_retries=0, max_rate_limit_wait=0, **kwargs)

    def collection(self, path, field, **params):
        rows, seen = [], set()
        for _ in range(MAX_PAGES):
            response = self.request('GET', path, params=params)
            rows.extend(response[field])
            pagination = response['pagination']
            if not pagination['hasNextPage']:
                return rows
            cursor = pagination['nextCursor']
            if not cursor or cursor in seen:
                raise ValueError(f'Invalid pagination cursor from {path}')
            seen.add(cursor)
            params = {**params, 'cursor': cursor}
        raise ValueError(f'{path} exceeds {MAX_PAGES} pages; refusing an incomplete result')

    def pod(self, pod_id):
        from runpod.error import QueryError
        try:
            return self.request('GET', pod_path(pod_id))
        except QueryError as error:
            if error.status_code == HTTPStatus.NOT_FOUND:
                return None
            raise

    def terminate(self, pod_id):
        """Retry removal and verify absence; accepted DELETE alone is not success."""
        from requests import RequestException
        from runpod.error import QueryError
        path = pod_path(pod_id)
        for attempt in range(REMOVE_ATTEMPTS):
            try:
                try:
                    self.request('DELETE', path)
                except QueryError as error:
                    if error.status_code != HTTPStatus.NOT_FOUND:
                        raise
                if self.pod(pod_id) is None:
                    return {'id': pod_id, 'removed': True}
            except (QueryError, RequestException) as error:
                print(f'Removal attempt {attempt + 1} for {pod_id} failed: {type(error).__name__}', flush=True)
            time.sleep(POLL_SECONDS)
        raise RuntimeError(f'Pod {pod_id} removal is unconfirmed after {REMOVE_ATTEMPTS} attempts; check billing now')


def pod_summary(pod):
    """Select operational fields; environment variables can contain secrets."""
    compute, = (name for name in ('gpu', 'cpu') if name in pod)
    fields = ('id', 'name', 'status', 'cost', compute, 'ssh', 'dataCenterId', 'disk', 'mounts')
    return {**{name: pod[name] for name in fields}, 'estimated_pod_hourly_usd': float(hourly(pod))}


def hourly(pod):
    """Compute plus local disk estimate; independent network-volume charges are excluded."""
    # dry-ok: src/rewardhack/runpod_ops.py::hourly | active is defined once and consumed by gb and compute. The consumer does not independently decide activity: changing its predicate changes both charges. Container disk, persistent storage and compute are distinct charge components.
    active = pod['status'] != 'EXITED'
    mounts = pod['mounts']
    persistent = mounts['persistent']['size'] if 'persistent' in mounts else 0
    gb = (pod['disk'] if active else 0) + persistent * (1 if active else 2)
    compute = Decimal(str(pod['cost'])) if active else Decimal(0)
    return compute + Decimal(str(gb)) * DISK_GB_HOURLY_USD


def status(client):
    with ThreadPoolExecutor(max_workers=2) as pool:
        pods = pool.submit(client.collection, 'pods', 'pods')
        balance = pool.submit(client.request, 'GET', 'billing/balance')
        rows = [pod_summary(pod) for pod in pods.result()]
        return {'pods': rows, 'balance': balance.result(),
                'estimated_pod_hourly_usd': sum(pod['estimated_pod_hourly_usd'] for pod in rows),
                'network_volume_charges_included': False}


def endpoint(pod):
    direct = pod['ssh']['direct']
    if direct is None:
        return None
    if direct['username'] != 'root' or not re.fullmatch(r'[A-Za-z0-9.:-]+', direct['host']):
        raise ValueError('The project driver requires a direct root SSH endpoint')
    port = int(direct['port'])
    if not 1 <= port <= 65535:
        raise ValueError('Invalid SSH port')
    return direct['host'], str(port)


def approved_driver_record(root):
    """Return the approved script and deadline for deployment or relaunch."""
    record, problems = gates.checked_launch_record(root, settings=gates.driver_settings(os.environ))
    if problems:
        raise ValueError('\n'.join(problems))
    return record


def drive(root, record, address, results):
    subprocess.run(['bash', 'ops/drive_runpod.sh', *address, record['remote'], str(results),
                    str(record['deadline_minutes'])], cwd=root, check=True,
                   env={**os.environ, gates.EXPECTED_RECORD_ID: record['record_id']})


def write_pod_receipt(path, pod_id, record):
    """Persist the created resource identity before driving or cleaning up."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(path, {'pod_id': pod_id, 'record_id': record['record_id']}, exclusive=True)


def launch(client, root, config_path, results):
    """Create and drive one pod, removing it on startup failure.

    This API takes no script or deadline: drive passes record['remote'] and
    str(record['deadline_minutes']) to ops/drive_runpod.sh. The saved approval
    is the sole source of those arguments; its record_id binds the child gate.
    """
    root = Path(root)
    config = json.loads(Path(config_path).read_text())
    if set(config) - CREATE_FIELDS:
        raise ValueError(f'Unsupported deployment fields: {sorted(set(config) - CREATE_FIELDS)}')
    if not all(name in config for name in ('name', 'image', 'gpu', 'disk', 'ports', 'cloud')):
        raise ValueError('Deployment needs name, image, gpu, disk, ports, cloud')
    if '22/tcp' not in config['ports'] or ':' not in config['image']:
        raise ValueError('Deployment needs port 22/tcp and an explicitly tagged image')
    destination = root / results
    receipt = destination.with_suffix('.pod.json')
    if destination.exists() or destination.is_symlink() or receipt.exists() or receipt.is_symlink() or any(c.isspace() for c in str(results)):
        raise ValueError('The results directory must be new and its driver argument must contain no spaces')
    (root / '.tools').mkdir(exist_ok=True)
    with file_lock(root / '.tools/runpod-launch.lock', wait=0):
        record = approved_driver_record(root)
        if float(client.request('GET', 'billing/balance')['balanceAmount']) < record['worst_case_usd']:
            raise ValueError('Live balance is below this launch record\'s worst-case cost')
        if any(pod['name'] == config['name'] for pod in client.collection('pods', 'pods')):
            raise ValueError('Use a unique pod name so an ambiguous create can be reconciled')
        # No create retries: a lost response may already have deployed a billable pod.
        from requests import RequestException
        from runpod.error import QueryError
        try:
            pod = client.request('POST', 'pods', json={**config, 'startSsh': True})
        except (RequestException, QueryError):
            try:
                gates.mark_launched(root, record, cleanup=True)
            finally:
                for found in client.collection('pods', 'pods'):
                    if found['name'] == config['name']:
                        try:
                            write_pod_receipt(receipt, found['id'], record)
                        finally:
                            client.terminate(found['id'])
            raise
        pod_id = pod['id']
        print(f'Created pod {pod_id}; driver pending; cleanup: runpod_ops.py terminate {pod_id}', flush=True)
        success = False
        try:
            write_pod_receipt(receipt, pod_id, record)
            expires = time.monotonic() + gates.pod_pin(root, 'SSH_UP_SECONDS')
            while True:
                if hourly(pod) > Decimal(str(record['hourly_usd'])):
                    raise ValueError(f'Pod price estimate ${hourly(pod)}/h exceeds checked ${record["hourly_usd"]}/h')
                address = endpoint(pod)
                if address is not None:
                    break
                if pod['status'] in ('ERROR', 'EXITED') or time.monotonic() >= expires:
                    raise RuntimeError(f'Pod {pod_id} has no direct SSH endpoint in time (status {pod["status"]})')
                time.sleep(POLL_SECONDS)
                # dry-ok: entrypoints/runpod_ops.py::connection | Both sites reuse RunPod.pod for the one GET/404 operation; neither implements transport, decoding or absence policy. This loop polls the newly created pod while waiting for boot; connection performs one lookup of a user-selected existing pod. A change to GET/404 behavior belongs only in RunPod.pod; the separate lookup times and identities are necessary to these different operations.
                pod = client.pod(pod_id)
                if pod is None:
                    raise RuntimeError(f'Pod {pod_id} disappeared before driver startup')
            drive(root, record, address, results)
            success = True
        finally:
            if not success:
                try:
                    gates.mark_launched(root, record, cleanup=True)
                finally:
                    client.terminate(pod_id)
        return {'id': pod_id, 'driver_completed': True,
                'kept_for_relaunch': 'KEEP_POD' in record['settings'] and record['settings']['KEEP_POD'] == '1'}
