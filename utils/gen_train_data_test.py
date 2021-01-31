from utils.build_data import *
import tensorflow_datasets as tfds
import json

with open('./dataset/raw.json', 'r') as f:
    d = json.load(f)

section_name = set()
for ex in d:
    for sec in ex['sections']:
        if sec['heading']:
            section_name.add(sec['heading'])
