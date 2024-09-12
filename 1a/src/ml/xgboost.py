from typing import Tuple, List

from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
from gensim.models.doc2vec import Doc2Vec
import numpy as np

from base import Base
from helpers.evaluation import Evaluate

class Xgboost(Base):
    def __init__(
        self,
        dataset_dir_path: str,
        doc2vec_data_dir_path: str
    ) -> None:
        
        self._dataset_dir_path = dataset_dir_path
        self._doc2vec_data_dir_path = doc2vec_data_dir_path

    def _train(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame
    ) -> None:

        params = {
            'device': 'cuda',
            'objective': 'binary:logistic',
            'verbosity': 0,
            'num_round': 10
        }
        
        # one hot encoding labels
        # turning them into a matrix (H X W)
        one_hot_encoded_train = pd.get_dummies(train['y_true'])
        ndarray_labels_train = np.array(one_hot_encoded_train)

        one_hot_encoded_val = pd.get_dummies(validation['y_true'])
        ndarray_labels_val = np.array(one_hot_encoded_val)

        # creating input matrix (H X W)
        ndarray_embeddings_train = np.vstack(train['embeddings'].values)
        ndarray_embeddings_val = np.vstack(validation['embeddings'].values)

        dtrain = xgb.DMatrix(ndarray_embeddings_train, label=ndarray_labels_train)
        dtest = xgb.DMatrix(ndarray_embeddings_val, label=ndarray_labels_val)

        # evaluation validation
        evallist = [(dtrain, 'train'), (dtest, 'eval')]

        bst = xgb.train(params, dtrain, evals=evallist)

        return bst
    
    def _evaluate_test(
        self,
        model,
        test,
        labels
    ) -> None:
        
        ndarray_embeddings_test = np.vstack(test['embeddings'].values)

        dtest = xgb.DMatrix(ndarray_embeddings_test)

        y_pred = model.predict(dtest)
        index_array = np.argmax(y_pred, axis=-1)

        test["y_true"] = test['act'].apply(lambda x: labels.index(x))
        test["y_pred"] = index_array

        return test

    def _load_doc2vec(
        self
    ) -> None:
        
        return Doc2Vec.load(self._doc2vec_data_dir_path)
    
    def _generate_embeddings(
        self,
        data: list[pd.DataFrame, ...],
        model
    ) -> None:
        
        for datasets in data:
            
            datasets['embeddings'] = datasets['utterance'].apply(
                lambda x: np.array(model.infer_vector(x.split()))
            )

        return data

    def _split_train(
        self,
        df: pd.DataFrame,
        labels: List[str] 
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        train, validation = train_test_split(df, train_size=0.80, random_state=42)

        train['y_true'] = df['act'].apply(lambda x: labels.index(x))
        validation['y_true'] = df['act'].apply(lambda x: labels.index(x))

        return train, validation

    def run(
        self
    ) -> None:
        
        df = self._load_data()
        df = self.set_columns(df=df)
        df = self._preprocess(df=df)

        labels = self._get_labels(df=df)

        train, model_test = self._split_train_test(df=df)

        model_train, model_validation = self._split_train(df=train, labels=labels)

        doc2vec = self._load_doc2vec()

        model_train, model_validation, model_test = self._generate_embeddings(
            data=[model_train, model_validation, model_test],
            model=doc2vec
        )

        xgb_classfier = self._train(
            train=model_train,
            validation=model_validation
        )

        results = self._evaluate_test(
            model=xgb_classfier,
            test=model_test,
            labels=labels
        )
        
        Evaluate(
            experiment="xgboost",
            dataframe=results,
            labels=labels
        ).run()






