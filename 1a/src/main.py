from baselines.baseline_1 import Baseline_1
from baselines.baseline_2 import Baseline_2
from ml.xgboost import Xgboost
from ml.mlp import MultiLayerPerceptron

if __name__ == "__main__":

    Baseline_1(
        dataset_dir_path="/home/x7a/Documents/Github/INFOMAIR/1a/data/dialog_acts.dat"
    ).run()

    Baseline_2(
        dataset_dir_path="/home/x7a/Documents/Github/INFOMAIR/1a/data/dialog_acts.dat"
    ).run()

    Xgboost(
        dataset_dir_path="/home/x7a/Documents/Github/INFOMAIR/1a/data/dialog_acts.dat",
        doc2vec_data_dir_path="/home/x7a/Documents/Github/INFOMAIR/1a/data/gensim/doc2vec.model"
    ).run()

    MultiLayerPerceptron(
        dataset_dir_path="/home/x7a/Documents/Github/INFOMAIR/1a/data/dialog_acts.dat",
        doc2vec_data_dir_path="/home/x7a/Documents/Github/INFOMAIR/1a/data/gensim/doc2vec.model"
    ).run()
