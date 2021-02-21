from utils.format import drop_sent
import json


with open('../output/test/pred.json', 'r') as f:
    d = json.load(f)

with open("../output/test/test.json",'w') as f:
    for it in d:
        summ = drop_sent(it['pred'])
        s = json.dumps({it['id']: summ})
        f.write(s+'\n')
