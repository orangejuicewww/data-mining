import os

cur_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(cur_dir)
data_dir = os.path.join(root_dir, "data")


class Config:
    data_dir = data_dir
    poem_data_path = os.path.join(data_dir, "poems.csv")
    model_image = os.path.join(data_dir, "model.png")
    w2v_model_path = os.path.join(data_dir, "w2v.model")
    sentiment_model_path = os.path.join(data_dir, "sentiment.model")
    train_data_path = os.path.join(data_dir, "train.csv")  # 情感分类
    w2v_train_data_path = os.path.join(data_dir, "train.txt")  # 情感分类
