"""Copy the keys in super.index, but make the values empty.
"""

import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument("source_file", type=str)
parser.add_argument("target_file", type=str)
args = parser.parse_args()

with open(args.source_file, "r") as f_s, \
    open(args.target_file, "w") as f_t:
    super_index = json.load(f_s)
    super_index = {k: {"db": v["db"], "files": []} for k, v in super_index.items()}
    json.dump(super_index, f_t)


