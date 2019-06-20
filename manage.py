import argparse
import os
from functools import partial

from data_mining.utils.logger import logging_config

logging_config = partial(logging_config, relative_path="..", stream_log=True)


def test():
    logging_config("./test.log")
    import unittest
    tests = unittest.TestLoader().discover("tests", pattern="test_*")
    unittest.TextTestRunner().run(tests)


def sentiment_train():
    logging_config("./train.log")
    from data_mining.sentiment_classify import train
    train()


def w2v():
    logging_config("./w2v.log")
    from data_mining.word2vec import train
    train()

def predict():
    from data_mining.sentiment_classify import Prediction
    prediction=Prediction()
    prediction.predict()

def main():
    ''' Parse command line arguments and execute the code'''
    parser = argparse.ArgumentParser()
    # 测试
    parser.add_argument('--gpu', default="1", type=str, help="指定GPU编号，0 or 0,1,2...7  | nvidia-smi 查看GPU占用情况")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--w2v', action="store_true")
    # parse args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # 指定GPU 0 or 0,1,2 ...
    # logger = partial(logging_config, relative_path=args.relative_path, stream_log=args.stream_log)
    if args.train:
        sentiment_train()
    elif args.w2v:
        w2v()


if __name__ == '__main__':
    main()
