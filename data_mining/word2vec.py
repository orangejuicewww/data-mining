import os
import sys

import pandas as pd
from gensim.models import word2vec

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)
sys.path.append(root_dir)

from data_mining.config import Config


def train():
    # if not os.path.exists(Config.w2v_train_data_path):
    data_df = pd.read_csv(Config.poem_data_path)
    data_df.dropna(subset=['内容'], inplace=True)
    # text = "".join(data_df["内容"].values)
    with open(Config.w2v_train_data_path, "w", encoding="utf-8") as f:
        text = "\n".join([" ".join(poem) for poem in data_df["内容"].values])
        f.write(text)
    # 加载语料
    sentences = word2vec.Text8Corpus(Config.w2v_train_data_path)
    # 训练模型
    print(Config.w2v_train_data_path)
    model = word2vec.Word2Vec(sentences)
    # 保存模型
    model.save(Config.w2v_model_path)
    # 选出最相似的10个词
    for e in model.most_similar(positive=['春'], topn=10):
        print(e[0], e[1])
    return model


if __name__ == "__main__":
    train()
    exit(1)
    # 加载模型
    model = word2vec.Word2Vec.load('poem.model')
    for word in ["春", '思乡', "梅", "冬", "中秋"]:
        res = model.most_similar(positive=[word], topn=10)
        print(word)
        print(res)
        print("-" * 10)
