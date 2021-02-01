import json


def gen_train_data(path='train.json'):
    with open(path, 'r') as f:
        raw_data = json.load(f)
    train_data = []
    train_v1 = []
    for d in raw_data:
        article = " ".join(d['section_content'])
        summary = " ".join(d['summary'])
        article_words = len(article.split())
        summary_words = len(summary.split())
        train_v1.append({'article': article, 'summary': summary,
                         'article_words': article_words, 'summary_words': summary_words})
        print("id: %d, article words: %d, summary words: %d" % (d['id'], article_words, summary_words))
    with open('train_v1.json', 'w') as f:
        json.dump(train_v1, f)
    return


gen_train_data()