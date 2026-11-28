"""Describe immutable experiment artifacts from the launch-owned working directory."""
import json
from pathlib import Path
import sys

import pod_snapshot_upload as storage


def snapshots(arm, config):
    historical = config['historical_snapshot_files']
    if historical is not None:
        return {folder.name: historical for folder in arm.glob('adapter-step*')
                if all((folder / n).is_file() for n in historical)}
    return {marker.parent.name: list(json.loads(marker.read_text()))
            for marker in arm.glob('adapter-step*/snapshot-files.json')}


def bulk(arm, config):
    if not Path('EXIT_CODE').is_file():
        raise ValueError('pod output is still being written')
    return [storage.describe(str(p), allow_empty=True) for p in sorted(Path('.').rglob('*'))
            if p.is_file() and not set(p.parts) & {'adapter-rollout', 'adapter-sync-check', '.rsync-partial'}]


def main():
    storage.ROOT = Path.cwd()
    mode = sys.argv[1]
    config = json.loads((Path.home() / 'pilot-repo/.tools/experiment.json').read_text())
    handlers = {'snapshots': snapshots, 'bulk': bulk,
                'describe': lambda arm, config: [storage.describe(name) for name in json.load(sys.stdin)],
                'upload': lambda arm, config: storage.upload(json.load(sys.stdin), allow_empty=True)}
    print(json.dumps(handlers[mode](Path(config['output_arm']), config)))


if __name__ == '__main__':
    main()
