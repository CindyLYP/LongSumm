from utils.format import *
import json


def trans_file():
    with open('../output/test/pred.json', 'r') as f:
        d = json.load(f)

    res = {}
    with open("../output/test/test_with_abs.json", 'w') as f:
        for it in d:
            summ = drop_sent(it['pred'])
            res[it['id']] = summ
        f.write(json.dumps(res))


def summ_test():
    pred_in = '/home/gitlib/longsumm/dataset/json_data/test.json'
    with open(pred_in, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    res = {}
    with open("../output/test/textrank.json", 'w') as f:
        for it in dataset:
            doc = it['document']
            s = add_summ(doc)
            # print(s), print("--" * 64), print(add_keywords(doc)), print("**" * 64)
            res[it['id']] = s
        f.write(json.dumps(res))


trans_file()

