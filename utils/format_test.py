from utils.format import *
import json
from tqdm import tqdm


def trans_file():
    with open('../output/test/pred.json', 'r') as f:
        d = json.load(f)

    res = {}
    with open("../output/test/cur_test.json", 'w') as f:
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


def test_merge():
    out_dir = "../output/test/"
    file = 'cur_test.json'
    with open(out_dir+file, 'r', encoding='utf-8') as f:
        d = json.load(f)
    clip_d = {}
    for k in tqdm(d.keys()):
        clip_d[k] = self_clip(d[k], r=0.8)
    with open(out_dir+"clip_"+file, 'w') as f:
        json.dump(clip_d, f)


def merge_files():
    f1 = "../output/test/.json"
    f2 = "../output/test/cur_test.json"


trans_file()
test_merge()
