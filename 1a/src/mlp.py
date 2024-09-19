import os
from typing import Tuple, List

import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from loguru import logger
import gensim
import typer
from typing_extensions import Annotated

from helpers.base import Base
from helpers.evaluation import Evaluate

mlp_app = typer.Typer()

class MLP(nn.Module):
    def __init__(
        self,
        feature_shape: int,
        num_classes: int
    ) -> None:
        super().__init__()

        self.linear1 = nn.Linear(in_features=feature_shape, out_features=64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=64, out_features=num_classes)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)

    def forward(
        self, 
        x
    ) -> torch.Tensor:
        
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.flatten(x)
        x = self.softmax(x)
        return x

class MLPDataset(Dataset):
    def __init__(
        self, 
        df: pd.DataFrame
    ) -> None:

        self._df = df.reset_index(drop=True)

    def __len__(
        self
    ) -> int:

        return self._df.shape[0]

    def __getitem__(
        self, 
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        embeddings, labels = self._df.iloc[idx, 3], self._df.iloc[idx, 4:]

        return {"inputs": torch.tensor(embeddings), "labels": torch.tensor(labels)}


class MultiLayerPerceptron(Base):
    """
    This class is meant to house in the functionalities needed to train a mlp model.

    Attributes:
        dataset_dir_path (str): The dataset directory path defining where the dialog_acts data is stored.
        doc2vec_data_dir_path (str): The doc2vec data directory path which is needed to load in the doc2vec model.
        checkpoint_dir_path (str): The checkpoint directory path needed to save the mlp model.
        device (str): A string defining the device
    """
    def __init__(
        self,
        dataset_dir_path: str,
        doc2vec_data_dir_path: str,
        checkpoint_dir_path: str,
        experiment_name: str,
        device: str,
        deduplication: bool
    ) -> None:
        
        self._dataset_dir_path = dataset_dir_path
        self._doc2vec_data_dir_path = doc2vec_data_dir_path
        self._checkpoint_dir_path = checkpoint_dir_path
        self._experiment_name = experiment_name
        self._device = device
        self._deduplication = deduplication

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
    def _train(
        self,
        train_set: pd.DataFrame,
        val_set: pd.DataFrame,
        eta: float,
        batch_size: int,
        epochs: int
    ) -> MLP:
        """
        This function is needed to train the MLP model.

        Args:
            train_set (pd.DataFrame): A pandas dataframe consisting of the train set.
            val_set (pd.DataFrame): A pandas dataframe consisting of the validation set.

        Returns:
            MLP: The mlp model.
        """

        train_set = pd.concat([train_set, pd.get_dummies(train_set['y_true'])], axis=1)
        val_set = pd.concat([val_set, pd.get_dummies(val_set['y_true'])], axis=1)

        train_dataset = MLPDataset(df=train_set)
        val_dataset = MLPDataset(df=val_set)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        device = self._device

        model = MLP(feature_shape=self.doc2vec.vector_size, num_classes=len(self.labels)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=eta)
        loss_fn = torch.nn.BCELoss()

        for epoch in range(epochs):

            model.train()

            running_train_correct, running_train_loss, running_val_loss, running_val_correct = 0, 0, 0, 0

            for batch in tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}/{epochs}", leave=False):
                
                inputs, labels = batch['inputs'].to(device), batch['labels'].to(device)

                labels = labels.float()

                output = model(inputs)
                
                loss = loss_fn(output, labels)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * inputs.size(0)
                running_train_correct += torch.sum(torch.argmax(output, dim=1) == torch.argmax(labels, dim=1)).item()

            mean_train_loss = running_train_loss / len(train_dataloader)
            mean_train_accuracy = running_train_correct / len(train_dataloader)

            model.eval()

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Val Epoch {epoch + 1}/{epochs}", leave=False):
                
                    inputs, labels = batch['inputs'].to(device), batch['labels'].to(device)

                    labels = labels.float()

                    output = model(inputs)

                    loss = loss_fn(output, labels)

                    running_val_loss += loss.item() * inputs.size(0)
                    running_val_correct += torch.sum(torch.argmax(output, dim=1) == torch.argmax(labels, dim=1)).item()

                mean_val_loss = running_val_loss / len(val_dataloader)
                mean_val_accuracy = running_val_correct / len(val_dataloader)

            print(f"""
            Epoch {epoch + 1}/{epochs},
            Train Loss: {mean_train_loss:.4f}, 
            Train Acc: {mean_train_accuracy:.4f}, 
            Val Loss: {mean_val_loss:.4f}, 
            Val Acc: {mean_val_accuracy:.4f}
            """)

        return model

    @logger.catch
    def _evaluate_model(
        self,
        model: MLP,
        test_set: pd.DataFrame
    ) -> pd.DataFrame:
        """
        This function is needed to evaluate the model / predict based on the given test set.

        Args:
            model (MLP): A mlp model construct that will predict the test set.
            test_set (pd.DataFrame): A pandas dataframe consisting of the test set.
        
        Returns:
            pd.DataFrame: A pandas dataframe consisting of the test set with the corresponding predicted values.
        """
        
        test_dataset = MLPDataset(df=test_set)

        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        device = self._device

        model.eval()

        y_preds = []

        with torch.no_grad():

            for batch in test_dataloader:
                
                inputs, labels = batch['inputs'].to(device), batch['labels'].to(device)

                output = model(inputs)

                preds = torch.argmax(output, dim=1)
                y_preds.extend(preds.cpu().tolist())

        test_set['y_pred'] = y_preds

        return test_set

    @logger.catch
    def _save_model(
        self,
        model: MLP
    ) -> None:
        """
        This function is needed to save the MLP model.

        Args:
            model (MLP): The MLP model constructor which needs to be saved.

        Returns:
            None
        """
        
        os.makedirs(self._checkpoint_dir_path + "/" + self._experiment_name, exist_ok=True)
        torch.save(model.state_dict(), self._checkpoint_dir_path + "/" + self._experiment_name + "/" + "model.pth")

    @logger.catch      
    def _load_doc2vec(
        self
    ) -> gensim.models.doc2vec.Doc2Vec:
        """
        This function is needed to load in the doc2vec model.

        Args:
            None

        Returns:
            None
        """
        
        return Doc2Vec.load(self._doc2vec_data_dir_path)

    @logger.catch
    def _load_mlp(
        self
    ) -> None:
        """
        This function is needed to load in the trained MLP model.

        Args:
            None

        Returns:
            None
        """
        
        model = MLP(feature_shape=self.doc2vec.vector_size, num_classes=len(self.labels)).to(self._device)
        model.load_state_dict(torch.load(self._checkpoint_dir_path + "/" + self._experiment_name + "/" + "model.pth", weights_only=True))
        model.eval()
        
        return model
    
    @logger.catch
    def _generate_embeddings(
        self,
        data: list[pd.DataFrame, ...],
        model: gensim.models.doc2vec.Doc2Vec
    ) -> list[pd.DataFrame, ...]:
        """
        This function is needed to generate embeddings based on the utterance.

        Args:
            data (list[pd.DataFrame, ...]): A dataframe consisting the individial datasets.
            model (gensim.models.doc2vec.Doc2Vec): The model required to generate embeddings.

        Returns:
            list[pd.DataFrame, ...]: A list containing the individual datasets.
        """
        
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
        """
        This function is needed to split the initial train dataset into a train and validation set for the MLP.

        Args:
            df (pd.DataFrame): A pandas dataframe consisting of the train dataset.
            labels (List[str]): A list of strings defining the labels for each prediction.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple consisting of the train / valiidation dataset.
        """

        train, validation = train_test_split(df, train_size=0.60, random_state=30)

        train['y_true'] = df['act'].apply(lambda x: labels.index(x))
        validation['y_true'] = df['act'].apply(lambda x: labels.index(x))

        return train, validation

    @logger.catch
    def inference(
        self,
        text: str
    ) -> None:
        """
        This function is needed to do inference.

        Args:
            text (str): A text defining the utterance

        Returns:
            None
        """

        device = self._device
        model = self._load_mlp()

        embeddings = self.doc2vec.infer_vector(text.split())

        embeddings = torch.tensor(embeddings.reshape(1, -1))

        with torch.no_grad():

            output = model(embeddings)

            print(f"""act: {self.labels[torch.argmax(output, dim=1).item()]}\n""")

    @logger.catch
    def evaluate(
        self
    ) -> None:
        """
        This function is being used as the main evaluation function for the MLP model.

        Args:
            None
        
        Returns:
            None
        """

        model = self._load_mlp()

        self.model_test["y_true"] = self.model_test['act'].apply(
            lambda x: self.labels.index(x)
        )

        data = self._generate_embeddings(
            data=[self.model_test],
            model=self.doc2vec
        )  

        model_test = pd.concat([data[0], pd.get_dummies(data[0]['y_true'])], axis=1)

        results = self._evaluate_model(
            model=model,
            test_set=model_test
        )

        Evaluate(
            experiment="multi layer perceptron",
            dataframe=results,
            labels=self.labels
        ).run()

    @logger.catch
    def run(
        self,
        eta: float,
        batch_size: int,
        epochs: int
    ) -> None:
        """
        This function is being used as the main functionality for training the MLP model.

        Args:
            None
        
        Returns:
            None
        """

        model = self._train(
            train_set=self.model_train,
            val_set=self.model_validation,
            eta=eta,
            batch_size=batch_size,
            epochs=epochs
        )

        self._save_model(
            model=model
        )

@mlp_app.command()
def inference(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    doc2vec_data_dir_path: Annotated[str, typer.Option(help="The doc2vec model dataset directory path")] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="Checkpoint directory path for the mlp")] = None,
    experiment_name: Annotated[str, typer.Option(help="The experiment name you want to use")] = None,
    device: Annotated[str, typer.Option(help="The device you want to use.")] = None,
    deduplication: Annotated[bool, typer.Option(help="Whether the xgboost model should be trained on deduplicated data from dialog acts dataset.")] = None,
) -> None:

    mlp = MultiLayerPerceptron(
        dataset_dir_path = dataset_dir_path,
        doc2vec_data_dir_path = doc2vec_data_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        device=device,
        deduplication=deduplication
    )

    while True:

        text = input("Enter your utterance: ")

        mlp.inference(text=text.lower())

@mlp_app.command()
def evaluate(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    doc2vec_data_dir_path: Annotated[str, typer.Option(help="The doc2vec model dataset directory path")] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="Checkpoint directory path for the mlp")] = None,
    experiment_name: Annotated[str, typer.Option(help="The experiment name you want to use")] = None,
    device: Annotated[str, typer.Option(help="The device you want to use.")] = None,
    deduplication: Annotated[bool, typer.Option(help="Whether the xgboost model should be trained on deduplicated data from dialog acts dataset.")] = None,
) -> None:
    
    MultiLayerPerceptron(
        dataset_dir_path = dataset_dir_path,
        doc2vec_data_dir_path = doc2vec_data_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        device=device,
        deduplication=deduplication
    ).evaluate()

@mlp_app.command()
def train(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    doc2vec_data_dir_path: Annotated[str, typer.Option(help="The doc2vec model dataset directory path")] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="Checkpoint directory path for the mlp")] = None,
    experiment_name: Annotated[str, typer.Option(help="The experiment name you want to use")] = None,
    device: Annotated[str, typer.Option(help="The device you want to use.")] = None,
    eta: Annotated[float, typer.Option(help="The learning rate you want to use.")] = None,
    batch_size: Annotated[int, typer.Option(help="The amount of batch size you want to set the dataloaders for")] = None,
    epochs: Annotated[int, typer.Option(help="The amount of epochs you want to run the mlp for.")] = None,
    deduplication: Annotated[bool, typer.Option(help="Whether the xgboost model should be trained on deduplicated data from dialog acts dataset.")] = None,
) -> None:
    
    MultiLayerPerceptron(
        dataset_dir_path = dataset_dir_path,
        doc2vec_data_dir_path = doc2vec_data_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        device=device,
        deduplication=deduplication
    ).run(
        eta=eta,
        batch_size=batch_size,
        epochs=epochs
    )
        

