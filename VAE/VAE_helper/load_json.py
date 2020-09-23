"""
Created on 11/09/2020
@author bvs
"""
import json
from pathlib import Path

def load_json(file):
    full_path = Path(file)
    with open(full_path) as f:
        return json.load(f)