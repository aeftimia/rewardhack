#!/usr/bin/env python3
"""Browser-free RunPod operations; launch hands off to the project's gated driver."""
import argparse
import json
import shlex
from pathlib import Path
from urllib.parse import quote

from rewardhack.runpod_ops import RunPod, approved_driver_record, drive, endpoint, launch, pod_summary, status

ROOT = Path(__file__).resolve().parents[1]
READS = {'gpus': ('catalog/gpus', None), 'templates': ('templates', 'templates'),
         'volumes': ('network-volumes', 'networkVolumes'),
         'balance': ('billing/balance', None), 'billing': ('billing/pods', None)}


def read(args, client):
    path, field = READS[args.command]
    params = {'include': 'AVAILABILITY', 'product': 'POD', 'count': args.count} if args.command == 'gpus' else {}
    if args.command == 'billing' and args.pod_id is not None:
        params['podId'] = args.pod_id
    result = ({field: client.collection(path, field, **params)} if field is not None
              else client.request('GET', path, params=params))
    if args.command == 'templates':
        result['templates'] = [{key: row[key] for key in ('id', 'name', 'image', 'disk', 'ports') if key in row}
                               for row in result['templates']]
    return result


def connection(args, client):
    pod = client.pod(args.pod_id)
    if pod is None:
        raise ValueError(f'Pod {args.pod_id} does not exist')
    address = endpoint(pod)
    if address is None:
        raise ValueError('Direct SSH is not available yet')
    return address


def ssh(args, client):
    address = connection(args, client)
    return {'command': shlex.join(['ssh', '-p', address[1], 'root@' + address[0]])}


def existing(args, client):
    completed = False
    try:
        record = approved_driver_record(ROOT)
        drive(ROOT, record, connection(args, client), args.results)
        completed = True
    finally:
        if not completed:
            client.terminate(args.pod_id)
    return {'id': args.pod_id, 'driver_completed': True}


def stop(args, client):
    client.request('POST', 'pods/' + quote(args.pod_id, safe='') + '/action', json={'action': 'stop'})
    return {'id': args.pod_id, 'stop_requested': True, 'storage_may_still_bill': True}


COMMANDS = {
    **{name: (read, ()) for name in READS},
    'status': (lambda args, client: status(client), ()),
    'show': (lambda args, client: pod_summary(client.pod(args.pod_id)), ('pod_id',)),
    'ssh': (ssh, ('pod_id',)),
    'drive': (existing, ('pod_id', '--results')),
    'terminate': (lambda args, client: client.terminate(args.pod_id), ('pod_id',)),
    'stop': (stop, ('pod_id',)),
    'launch': (lambda args, client: launch(client, ROOT, args.config, args.results), ('--config', '--results')),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name, (handler, arguments) in COMMANDS.items():
        command = sub.add_parser(name)
        command.set_defaults(handler=handler)
        if name == 'gpus':
            command.add_argument('--count', type=int, default=1)
        if name == 'billing':
            command.add_argument('--pod-id')
        for argument in arguments:
            command.add_argument(argument, **({'required': True} if argument.startswith('--') else {}))
    args = parser.parse_args()
    print(json.dumps(args.handler(args, RunPod()), indent=2))


if __name__ == '__main__':
    main()
