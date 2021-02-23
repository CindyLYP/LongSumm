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


def merge_files():
    with open("../output/test/test_with_abs.json", 'r', encoding='utf-8') as f:
        d1 = json.load(f)
    with open("../output/test/result_1000.json", 'r', encoding='utf-8') as f:
        d2 = json.load(f)
    d = {}
    for k in d1.keys():
        s = d1[k] + " " + d2[k]
        d[k] = add_summ(s)
    with open("../output/test/t3.json", 'w') as f:
        json.dump(d, f)


trans_file()