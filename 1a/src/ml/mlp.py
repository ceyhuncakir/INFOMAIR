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
import gensim
import typer

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

        self.linear1 = nn.Linear(in_features=feature_shape, out_features=128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(in_features=128, out_features=num_classes)
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
    def __init__(
        self,
        dataset_dir_path: str,
        doc2vec_data_dir_path: str
    ) -> None:
        
        self._dataset_dir_path = dataset_dir_path
        self._doc2vec_data_dir_path = doc2vec_data_dir_path

    def _train(
        self,
        train_set: pd.DataFrame,
        val_set: pd.DataFrame
    ) -> MLP:

        train_dataset = MLPDataset(df=train_set)
        val_dataset = MLPDataset(df=val_set)

        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

        model = MLP(feature_shape=300, num_classes=15).to("cuda")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = torch.nn.BCELoss()

        epochs = 5

        for epoch in range(epochs):

            model.train()

            running_train_correct, running_train_loss, running_val_loss, running_val_correct = 0, 0, 0, 0

            for batch in tqdm(train_dataloader, desc=f"Train Epoch {epoch + 1}/{epochs}", leave=False):
                
                inputs, labels = batch['inputs'].to("cuda"), batch['labels'].to("cuda")

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
                
                    inputs, labels = batch['inputs'].to("cuda"), batch['labels'].to("cuda")

                    labels = labels.float()

                    output = model(inputs)

                    loss = loss_fn(output, labels)

                    running_val_loss += loss.item() * inputs.size(0)
                    running_val_correct += torch.sum(torch.argmax(output, dim=1) == torch.argmax(labels, dim=1)).item()

                mean_val_loss = running_val_loss / len(val_dataloader)
                mean_val_accuracy = running_val_correct / len(val_dataloader)

            print(f"""
            Train Loss: {mean_train_loss:.4f}, 
            Train Acc: {mean_train_accuracy:.4f}, 
            Val Loss: {mean_val_loss:.4f}, 
            Val Acc: {mean_val_accuracy:.4f}
            """)

        return model

    def _evaluate_model(
        self,
        model: MLP,
        test_set: pd.DataFrame
    ) -> pd.DataFrame:
        
        test_dataset = MLPDataset(df=test_set)

        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        model.eval()

        y_preds = []

        with torch.no_grad():

            for batch in test_dataloader:
                
                inputs, labels = batch['inputs'].to("cuda"), batch['labels'].to("cuda")

                output = model(inputs)

                preds = torch.argmax(output, dim=1)
                y_preds.extend(preds.cpu().tolist())

        test_set['y_pred'] = y_preds

        return test_set
            
    def _load_doc2vec(
        self
    ) -> gensim.models.doc2vec.Doc2Vec:
        
        return Doc2Vec.load(self._doc2vec_data_dir_path)
    
    def _generate_embeddings(
        self,
        data: list[pd.DataFrame, ...],
        model: gensim.models.doc2vec.Doc2Vec
    ) -> list[pd.DataFrame, ...]:
        
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

        model_test["y_true"] = model_test['act'].apply(
            lambda x: labels.index(x)
        )

        model_train, model_validation = self._split_train(
            df=train, 
            labels=labels
        )

        doc2vec = self._load_doc2vec()

        model_train, model_validation, model_test = self._generate_embeddings(
            data=[model_train, model_validation, model_test],
            model=doc2vec
        )  

        model_train = pd.concat([model_train, pd.get_dummies(model_train['y_true'])], axis=1)
        model_validation = pd.concat([model_validation, pd.get_dummies(model_validation['y_true'])], axis=1)
        model_test = pd.concat([model_test, pd.get_dummies(model_test['y_true'])], axis=1)
       
        model = self._train(
            train_set=model_train,
            val_set=model_validation
        )

        results = self._evaluate_model(
            model=model,
            test_set=model_test
        )

        Evaluate(
            experiment="multi layer perceptron",
            dataframe=results,
            labels=labels
        ).run()

@mlp_app.command()
def inference(

) -> None:
    pass

@mlp_app.command()
def evaluate(

) -> None:
    pass

        

