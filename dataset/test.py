import os
import json


def gen_info(path='../abstractive_summaries/by_clusters'):
    info_list = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            filepath = subdir + os.sep + file

            if filepath.endswith(".json"):
                with open(filepath, 'r') as in_f:
                    summ_info = json.load(in_f)
                    info_list.append(summ_info)

    return info_list


d = gen_info()

for ex in d:
    if ex['id'] == 39353178:
        print("ex")