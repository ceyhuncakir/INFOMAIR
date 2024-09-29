import os
from typing import Tuple, List, Union

import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
from loguru import logger
import gensim
import typer
import pickle
from typing_extensions import Annotated
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt

from helpers.callbacks import *
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

        self.model = nn.Sequential(
            nn.Linear(feature_shape, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(
        self, 
        x
    ) -> torch.Tensor:
        
        x = self.model(x)
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

        sparse_matrix, label = self._df.iloc[idx, 3], self._df.loc[idx, 'y_true']

        return {"inputs": torch.tensor(sparse_matrix, dtype=torch.float32), "labels": torch.tensor(label, dtype=torch.long)}


class MultiLayerPerceptron(Base):
    """
    This class is meant to house in the functionalities needed to train a mlp model.

    Attributes:
        dataset_dir_path (str): The dataset directory path defining where the dialog_acts data is stored.
        checkpoint_dir_path (str): The checkpoint directory path needed to save the mlp model.
        device (str): A string defining the device
    """
    def __init__(
        self,
        dataset_dir_path: str,
        vectorizer_dir_path: str,
        checkpoint_dir_path: str,
        experiment_name: str,
        device: str,
        deduplication: bool
    ) -> None:
        
        self._dataset_dir_path = dataset_dir_path
        self._vectorizer_dir_path = vectorizer_dir_path
        self._checkpoint_dir_path = checkpoint_dir_path
        self._experiment_name = experiment_name
        self._device = device
        self._deduplication = deduplication

        self._train, self._test, self._labels, self._majority = self.process(
            deduplication=deduplication
        )
        
        self._vectorizer = self._load_vectorizer()

        self._train_sparse_matrix, self._test_sparse_matrix = self._vectorize(
            train=self._train,
            test=self._test
        )

        self._train['dense_matrix'] = self._train_sparse_matrix.toarray().tolist()
        self._test['dense_matrix'] = self._test_sparse_matrix.toarray().tolist()

        self._model_train, self._model_validation = self._split_train(df=self._train)

    @logger.catch
    def _train_model(
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

        train_dataset = MLPDataset(df=train_set)
        val_dataset = MLPDataset(df=val_set)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        device = self._device

        model = MLP(feature_shape=len(self._vectorizer.vocabulary_), num_classes=len(self._labels)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=eta)
        loss_fn = torch.nn.CrossEntropyLoss()

        train_accuracy_list = []
        val_accuracy_list = []

        train_loss_list = []
        val_loss_list = []

        for epoch in range(epochs):

            model.train()

            running_train_correct, running_train_loss, running_val_loss, running_val_correct = 0, 0, 0, 0

            for batch in tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}/{epochs}", leave=False):
                
                inputs, labels = batch['inputs'].to(device), batch['labels'].to(device)

                # feed forward
                output = model(inputs)
                
                loss = loss_fn(output, labels)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                running_train_loss += loss.item() * inputs.size(0)
                
                preds = torch.argmax(output, dim=1)
                running_train_correct += (preds == labels).sum().item()

            mean_train_loss = running_train_loss / len(train_dataset)
            mean_train_accuracy = running_train_correct / len(train_dataset)

            train_accuracy_list.append(mean_train_accuracy)
            train_loss_list.append(mean_train_loss)

            model.eval()

            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Val Epoch {epoch + 1}/{epochs}", leave=False):
                
                    inputs, labels = batch['inputs'].to(device), batch['labels'].to(device)

                    output = model(inputs)

                    loss = loss_fn(output, labels)

                    running_val_loss += loss.item() * inputs.size(0)
                    preds = torch.argmax(output, dim=1)
                    running_val_correct += (preds == labels).sum().item()

                mean_val_loss = running_val_loss / len(val_dataset)
                mean_val_accuracy = running_val_correct / len(val_dataset)

                val_accuracy_list.append(mean_val_accuracy)
                val_loss_list.append(mean_val_loss)

            print(f"""
            Epoch {epoch + 1}/{epochs},
            Train Loss: {mean_train_loss:.4f}, 
            Train Acc: {mean_train_accuracy:.4f}, 
            Val Loss: {mean_val_loss:.4f}, 
            Val Acc: {mean_val_accuracy:.4f}
            """)

        self.plot_training_validation(
            train_acc=train_accuracy_list,
            val_acc=val_accuracy_list,
            train_loss=train_loss_list,
            val_loss=val_loss_list
        )

        return model

    @logger.catch
    def plot_training_validation(
        self,
        train_acc: List[float], 
        val_acc: List[float], 
        train_loss: List[float], 
        val_loss: List[float], 
    ) -> None:
        """
        This function is needed to plot the training and validation accuracy and loss.

        Args:
            train_acc (List[float]): A list of floats defining the training accuracy.
            val_acc (List[float]): A list of floats defining the validation accuracy.
            train_loss (List[float]): A list of floats defining the training loss.
            val_loss (List[float]): A list of floats defining the validation loss.

        Returns:
            None
        """
        
        # Check that all lists have the same length
        if not (len(train_acc) == len(val_acc) == len(train_loss) == len(val_loss)):
            raise ValueError("All input lists must have the same length.")
        
        # Set Seaborn style
        sns.set(style="whitegrid")

        # Get the number of epochs from the length of the lists
        epochs = np.arange(1, len(train_acc) + 1)

        # Create the plot with Seaborn styling
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Plot Training and Validation Loss on the left y-axis
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:red', fontsize=12)
        sns.lineplot(x=epochs, y=train_loss, ax=ax1, marker='o', color='tab:red', label='Train Loss')
        sns.lineplot(x=epochs, y=val_loss, ax=ax1, marker='s', color='tab:orange', label='Validation Loss', linestyle='--')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        # Instantiate a second axes that shares the same x-axis
        ax2 = ax1.twinx()

        # Plot Training and Validation Accuracy on the right y-axis
        ax2.set_ylabel('Accuracy', color='tab:blue', fontsize=12)
        sns.lineplot(x=epochs, y=train_acc, ax=ax2, marker='^', color='tab:blue', label='Train Accuracy')
        sns.lineplot(x=epochs, y=val_acc, ax=ax2, marker='v', color='tab:green', label='Validation Accuracy', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        # Combine all lines for a single legend
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                fancybox=True, shadow=True, ncol=2)

        # Add a title
        plt.title(f'Training {"without duplicate entries" if self._deduplication else "with duplicate entries"}', fontsize=14, fontweight='bold')

        # Improve layout to make room for the legend
        fig.tight_layout()

        # Save the plot as a file
        os.makedirs(self._checkpoint_dir_path + "/" + self._experiment_name, exist_ok=True)
        plt.savefig(f"{self._checkpoint_dir_path}/{self._experiment_name}/training.png", dpi=300, bbox_inches='tight')

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
        y_true = []

        with torch.no_grad():

            for batch in test_dataloader:
                
                inputs, labels = batch['inputs'].to(device), batch['labels'].to(device)

                output = model(inputs)

                pred_proba = F.softmax(output, dim=1)

                preds = torch.argmax(pred_proba, dim=1)

                y_preds.extend(preds.cpu().tolist())
                y_true.extend(labels.cpu().tolist())

        df = pd.concat([pd.Series(y_true, name='y_true'), pd.Series(y_preds, name='y_pred')], axis=1)
        df['y_true'] = df['y_true'].apply(lambda x: self._labels[x])
        df['y_pred'] = df['y_pred'].apply(lambda x: self._labels[x])

        return df

    @logger.catch()
    def _load_vectorizer(
        self
    ) -> Union[CountVectorizer, TfidfVectorizer]:
        """
        This function is needed to load the trained vectorizer model.

        Args:
            None
        
        Returns:
            Union[CountVectorizer, TfidfVectorizer]: The trained vectorizer model
        """
        
        with open(self._vectorizer_dir_path, 'rb') as f:
            vectorizer = pickle.load(f)

        return vectorizer

    @logger.catch
    def _vectorize(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function is needed to vectorize the given dataframes.

        Args:
            train (pd.DataFrame): The training dataframe that needs to be vectorized.
            test (pd.DataFrame): The test dataframe that needs to be vectorized.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: The vectorized training and test dataframes.
        """

        train_corpus = train['utterance'].values
        test_corpus = test['utterance'].values

        train_sparse_matrix = self._vectorizer.transform(train_corpus)
        test_sparse_matrix = self._vectorizer.transform(test_corpus)

        return train_sparse_matrix, test_sparse_matrix

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
    def _load_model(
        self
    ) -> None:
        """
        This function is needed to load in the trained MLP model.

        Args:
            None

        Returns:
            None
        """
        
        model = MLP(feature_shape=len(self._vectorizer.vocabulary_), num_classes=len(self._labels)).to(self._device)
        model.load_state_dict(torch.load(self._checkpoint_dir_path + "/" + self._experiment_name + "/" + "model.pth", weights_only=True))
        model.eval()
        
        return model

    @logger.catch
    def _split_train(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        This function is needed to split the initial train dataset into a train and validation set for the MLP.

        Args:
            df (pd.DataFrame): A pandas dataframe consisting of the train dataset.
            labels (List[str]): A list of strings defining the labels for each prediction.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: A tuple consisting of the train / valiidation dataset.
        """

        train, validation = train_test_split(df, train_size=0.90, random_state=42)

        return train, validation

    @logger.catch
    def inference(
        self,
        text: str
    ) -> Tuple[str, np.ndarray]:
        """
        This function is needed to do inference.

        Args:
            text (str): A text defining the utterance

        Returns:
            None
        """

        device = self._device
        model = self._load_model()

        sparse_matrix = self._vectorizer.transform([text])

        sparse_matrix_tensor = torch.tensor(sparse_matrix.toarray().tolist(), dtype=torch.float32).to(device)

        with torch.no_grad():

            output = model(sparse_matrix_tensor)

            pred_proba = F.softmax(output, dim=1)

            index_array = np.argmax(pred_proba.cpu().numpy())    

            return self._labels[torch.argmax(pred_proba, dim=1).item()], pred_proba.cpu().numpy()[0][index_array]

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

        model = self._load_model()

        results = self._evaluate_model(
            model=model,
            test_set=self._test
        )

        Evaluate(
            experiment="multi layer perceptron",
            dataframe=results,
            labels=self._labels
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

        model = self._train_model(
            train_set=self._model_train,
            val_set=self._model_validation,
            eta=eta,
            batch_size=batch_size,
            epochs=epochs
        )

        self._save_model(
            model=model
        )

@mlp_app.command()
def inference(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.", callback=path_valid)] = None,
    vectorizer_dir_path: Annotated[str, typer.Option(help="The vectorizer directory path where the trained vectorizer model resides in.", callback=path_valid)] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="Checkpoint directory path for the mlp", callback=path_valid)] = None,
    experiment_name: Annotated[str, typer.Option(help="The experiment name you want to use", callback=experiment_value)] = None,
    device: Annotated[str, typer.Option(help="The device you want to use.")] = "cpu",
    deduplication: Annotated[bool, typer.Option(help="Whether the mlp model should be trained on deduplicated data from dialog acts dataset.")] = False,
) -> None:

    mlp = MultiLayerPerceptron(
        dataset_dir_path = dataset_dir_path,
        vectorizer_dir_path=vectorizer_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        device=device,
        deduplication=deduplication
    )

    while True:

        text = input("Enter your utterance: ")

        categorical_pred, probability = mlp.inference(text=text.lower())

        print(f"""act: {categorical_pred}, probability: {probability}\n""")

@mlp_app.command()
def evaluate(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.", callback=path_valid)] = None,
    vectorizer_dir_path: Annotated[str, typer.Option(help="The vectorizer directory path where the trained vectorizer model resides in.", callback=path_valid)] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="Checkpoint directory path for the mlp", callback=path_valid)] = None,
    experiment_name: Annotated[str, typer.Option(help="The experiment name you want to use", callback=experiment_value)] = None,
    device: Annotated[str, typer.Option(help="The device you want to use.")] = "cpu",
    deduplication: Annotated[bool, typer.Option(help="Whether the mlp model should be trained on deduplicated data from dialog acts dataset.")] = False,
) -> None:
    
    MultiLayerPerceptron(
        dataset_dir_path = dataset_dir_path,
        vectorizer_dir_path=vectorizer_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        device=device,
        deduplication=deduplication
    ).evaluate()

@mlp_app.command()
def train(
    dataset_dir_path: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.", callback=path_valid)] = None,
    vectorizer_dir_path: Annotated[str, typer.Option(help="The vectorizer directory path where the trained vectorizer model resides in.", callback=path_valid)] = None,
    checkpoint_dir_path: Annotated[str, typer.Option(help="Checkpoint directory path for the mlp", callback=path_valid)] = None,
    experiment_name: Annotated[str, typer.Option(help="The experiment name you want to use", callback=experiment_value)] = None,
    device: Annotated[str, typer.Option(help="The device you want to use.")] = "cpu",
    eta: Annotated[float, typer.Option(help="The learning rate you want to use.")] = 1e-2,
    batch_size: Annotated[int, typer.Option(help="The amount of batch size you want to set the dataloaders for")] = 64,
    epochs: Annotated[int, typer.Option(help="The amount of epochs you want to run the mlp for.")] = 10,
    deduplication: Annotated[bool, typer.Option(help="Whether the mlp model should be trained on deduplicated data from dialog acts dataset.")] = False,
) -> None:
    
    MultiLayerPerceptron(
        dataset_dir_path = dataset_dir_path,
        vectorizer_dir_path=vectorizer_dir_path,
        checkpoint_dir_path=checkpoint_dir_path,
        experiment_name=experiment_name,
        device=device,
        deduplication=deduplication
    ).run(
        eta=eta,
        batch_size=batch_size,
        epochs=epochs
    )
        

