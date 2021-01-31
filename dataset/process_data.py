import json

with open('raw.json', 'r') as f:
    d = json.load(f)

for ex in d:
    flag = True
    for sec in ex['sections']:
        if not sec['heading']: continue
        if 'conclusion' in sec['heading'].lower():
            flag = False
    if flag:
        print(ex)