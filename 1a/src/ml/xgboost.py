import os
from typing import Tuple, List

from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import typer
from typing_extensions import Annotated
from loguru import logger

from helpers.base import Base
from helpers.evaluation import Evaluate

xgboost_app = typer.Typer()

class Xgboost(Base):
    def __init__(
        self,
        dataset_dir_path: str,
        doc2vec_data_dir_path: str,
        checkpoint_dir_path: str,
        experiment_name: str,
        deduplication: bool
    ) -> None:
        
        self._dataset_dir_path = dataset_dir_path
        self._doc2vec_data_dir_path = doc2vec_data_dir_path
        self._checkpoint_dir_path = checkpoint_dir_path
        self._experiment_name = experiment_name

        self.df = self._load_data()
        self.df = self.set_columns(df=self.df)
        self.df = self._preprocess(df=self.df, deduplication=deduplication)

        self.labels = self._get_labels(df=self.df)
        self.train, self.model_test = self._split_train_test(df=self.df)
        self.model_train, self.model_validation = self._split_train(df=self.train, labels=self.labels)
        self.doc2vec = self._load_doc2vec()
        self.model_train, self.model_validation, self.model_test = self._generate_embeddings(
            data=[self.model_train, self.model_validation, self.model_test],
            model=self.doc2vec
        )

    @logger.catch
    def _train_model(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        verbosity: int,
        num_round: int,
    ) -> None:

        params = {
            'device': 'cuda',
            'objective': 'multi:softprob',
            'verbosity': verbosity,
            'num_round': num_round,
            'num_class': len(train['y_true'].unique()),
        }

        labels_train = train['y_true'].values
        labels_val = validation['y_true'].values

        # creating input matrix (H X W)
        ndarray_embeddings_train = np.vstack(train['embeddings'].values)
        ndarray_embeddings_val = np.vstack(validation['embeddings'].values)

        dtrain = xgb.DMatrix(ndarray_embeddings_train, label=labels_train)
        dtest = xgb.DMatrix(ndarray_embeddings_val, label=labels_val)

        # evaluation validation
        evallist = [(dtrain, 'train'), (dtest, 'eval')]

        bst = xgb.train(params, dtrain, evals=evallist)

        return bst
    
    @logger.catch
    def _save_model(
        self,
        model
    ) -> None:

        os.makedirs(f"{self._checkpoint_dir_path}/{self._experiment_name}", exist_ok=True)
        
        model.save_model(f'{self._checkpoint_dir_path}/{self._experiment_name}/xgboost.json')

    @logger.catch
    def _load_model(
        self
    ) -> None:
        
        xgb_classifier = xgb.Booster()
        xgb_classifier.load_model(f'{self._checkpoint_dir_path}/{self._experiment_name}/xgboost.json')

        return xgb_classifier

    @logger.catch 
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

    @logger.catch
    def _load_doc2vec(
        self
    ) -> None:
        
        return Doc2Vec.load(self._doc2vec_data_dir_path)
    
    @logger.catch
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

    @logger.catch
    def _split_train(
        self,
        df: pd.DataFrame,
        labels: List[str] 
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

        train, validation = train_test_split(df, train_size=0.80, random_state=42)

        train['y_true'] = df['act'].apply(lambda x: labels.index(x))
        validation['y_true'] = df['act'].apply(lambda x: labels.index(x))

        return train, validation

    @logger.catch
    def inference(
        self,
        utterance: str
    ) -> str:

        xgb_classfier = self._load_model()

        # preprocess
        utterance = utterance.lower().lstrip()

        embeddings = np.array(self.doc2vec.infer_vector(utterance.split())).reshape(1, -1)

        dtest = xgb.DMatrix(embeddings)

        y_preds_proba = xgb_classfier.predict(dtest)
        index_array = np.argmax(y_preds_proba, axis=-1)

        categorical_pred = self.labels[index_array[0]]

        print(f"""act: {categorical_pred}, probability: {y_preds_proba[0][index_array]}\n""")

        return categorical_pred
    
    @logger.catch
    def evaluate(
        self
    ) -> None:

        xgb_classfier = self._load_model()
        
        results = self._evaluate_test(
            model=xgb_classfier,
            test=self.model_test,
            labels=self.labels
        )
        
        Evaluate(
            experiment=self._experiment_name,
            dataframe=results,
            labels=self.labels
        ).run()

    @logger.catch
    def run(
        self,
        verbosity: int,
        num_round: int
    ) -> None:

        xgb_classfier = self._train_model(
            train=self.model_train,
            validation=self.model_validation,
            verbosity=verbosity,
            num_round=num_round
        )

        self._save_model(model=xgb_classfier)

@xgboost_app.command()
def evaluate(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    doc2vec_data_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    experiment_name: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    deduplication: Annotated[bool, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
) -> None:
    
    Xgboost(
        dataset_dir_path=dataset_dir_path,
        doc2vec_data_dir_path=doc2vec_data_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    ).evaluate()

@xgboost_app.command()
def inference(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    doc2vec_data_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    experiment_name: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    deduplication: Annotated[bool, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
) -> None:
    
    xgboost = Xgboost(
        dataset_dir_path=dataset_dir_path,
        doc2vec_data_dir_path=doc2vec_data_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    )

    while True:

        utterance = input("Enter your utterance: ")

        xgboost.inference(utterance=utterance)

@xgboost_app.command()
def train(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    doc2vec_data_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    experiment_name: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    verbosity: Annotated[int, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    num_rounds: Annotated[int, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    deduplication: Annotated[bool, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
) -> None:
    
    Xgboost(
        dataset_dir_path=dataset_dir_path,
        doc2vec_data_dir_path=doc2vec_data_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        deduplication=deduplication
    ).run(
        verbosity=verbosity,
        num_round=num_rounds 
    )

