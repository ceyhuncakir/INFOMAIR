import os
from typing import List

import gensim
from gensim.test.utils import datapath
from gensim.corpora import WikiCorpus
import typer
import uuid
from typing_extensions import Annotated

doc2vec_app = typer.Typer()

class Doc2vec:
    """
    Attributes:
        data_dump_data_dir (str):
        checkpoint_data_dir (str):
        vector_size (int):
        min_count (int):
        epochs (int):
        preprocess_lower (bool):
        preprocess_token_max_len (int):
        preprocess_processes (int):
    """

    def __init__(
        self,
        data_dump_data_dir: str,
        checkpoint_data_dir: str,
        experiment_name: str,
        vector_size: int,
        min_count: int,
        epochs: int,
        preprocess_lower: bool,
        preprocess_token_max_len: int,
        preprocess_processes: int,
    ) -> None:
        
        self._dataset_dir_path = data_dump_data_dir
        self._checkpoint_data_dir = checkpoint_data_dir
        self._experiment_name = experiment_name
        self._vector_size = vector_size
        self._min_count = min_count
        self._epochs = epochs
        self._preprocess_lower = preprocess_lower
        self._preprocess_token_max_len = preprocess_token_max_len
        self._preprocess_processes = preprocess_lower

    def _load_wiki_dumps(
        self
    ) -> List[gensim.corpora.wikicorpus.WikiCorpus]:

        dump_collection = []
        
        for files in os.listdir(self._dataset_dir_path):
            
            wiki = WikiCorpus(
                self._dataset_dir_path + "/" + files, 
                lower=True, 
                token_max_len=20, 
                processes=6
            )

            dump_collection.append(wiki)

        return dump_collection

    def _prep_wiki_dumps(
        self,
        dumps: List[gensim.corpora.wikicorpus.WikiCorpus]
    ) -> None:
        """
        This function is needed to prep the wiki dumps that have been collected.

        Args:
            dumps (list): A list of dumps

        Returns:
            None
        """
        
        for dump in dumps:
            
            for idx, tokens in enumerate(dump.get_texts()):

                yield gensim.models.doc2vec.TaggedDocument(tokens, [idx])

    def split_train_test(
        self,
        data: list
    ) -> None:
        pass

    def _train_model(
        self,
        data: List[gensim.models.doc2vec.TaggedDocument]
    ) -> gensim.models.doc2vec.Doc2Vec:
        """
        This function is needed train the doc2vec model.

        Args:
            data (list): A list of data that is needed to train the doc2vec model.

        Returns:

        """

        model = gensim.models.doc2vec.Doc2Vec(
            vector_size=self._vector_size, 
            min_count=self._min_count, 
            epochs=self._epochs
        )

        model.build_vocab(data)

        model.train(data, total_examples=model.corpus_count, epochs=model.epochs)

        return model

    def evaluate_model(
        self
    ) -> None:
        pass

    def _save_model(
        self,
        model: gensim.models.doc2vec.Doc2Vec
    ) -> None:
        """
        This function is needed to save the model to the designated path.

        Args:
            model: A model constructor class so we save the model.

        Returns;
            None
        """

        save_path = f"{self._checkpoint_data_dir}/{self._experiment_name}/"

        os.makedirs(save_path, exist_ok=True)

        model.save(save_path + "doc2vec.model")

    def run(
        self
    ) -> None:
        """
        This function is needed to run the main process of training a doc2vec model.

        Args:
            None

        Returns:
            None
        """
        
        dumps = self._load_wiki_dumps()   
        data = list(self._prep_wiki_dumps(dumps=dumps)) 
        model = self._train_model(data=data)
        self._save_model(model=model)

@doc2vec_app.command()
def train(
    data_dump_data_dir: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    checkpoint_data_dir: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    experiment_name: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    vector_size: Annotated[int, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    min_count: Annotated[int, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    epochs: Annotated[int, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    preprocess_lower: Annotated[bool, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    preprocess_token_max_len: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
    preprocess_processes: Annotated[str, typer.Option(help="The dataset dir path we want to specify for the dataset.")] = None,
) -> None:

    doc2vec = Doc2vec(
        data_dump_data_dir=data_dump_data_dir,
        checkpoint_data_dir=checkpoint_data_dir,
        experiment_name=experiment_name,
        vector_size=vector_size,
        min_count=min_count,
        epochs=epochs,
        preprocess_lower=preprocess_lower,
        preprocess_token_max_len=preprocess_token_max_len,
        preprocess_processes=preprocess_processes
    ).run()