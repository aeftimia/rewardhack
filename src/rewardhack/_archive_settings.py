"""Archive settings shared with the AWS CLI coordinator's standalone Python."""
from pathlib import Path

ENV = Path(__file__).with_name('archive.env')
lines = (line.strip() for line in ENV.read_text().splitlines())
SETTINGS = {name.strip(): value.strip() for name, value in
            (line.split('=', 1) for line in lines if line and not line.startswith('#'))}
