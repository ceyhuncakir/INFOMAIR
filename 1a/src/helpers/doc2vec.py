import os

import gensim
from gensim.test.utils import datapath
from gensim.corpora import WikiCorpus

class Doc2vec:
    def __init__(
        self,
        data_dump_data_dir: str,
        checkpoint_data_dir: str,
    ) -> None:
        
        self._dataset_dir_path = data_dump_data_dir
        self._checkpoint_data_dir = checkpoint_data_dir
    
    def _load_wiki_dumps(
        self
    ) -> list:

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
        dumps: list
    ) -> None:
        
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
        data: list
    ) -> None:

        print("training model")

        model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=25)
        model.build_vocab(data)
        model.train(data, total_examples=model.corpus_count, epochs=model.epochs)
    
        print("model trained")

        return model

    def evaluate_model(
        self
    ) -> None:
        pass

    def _save_model(
        self,
        model
    ) -> None:

        model.save(self._checkpoint_data_dir)

        print("model saved")

    def run(
        self
    ) -> None:
        
        dumps = self._load_wiki_dumps()   
        data = list(self._prep_wiki_dumps(dumps=dumps)) 
        print(len(data))
        print("processed")

        model = self._train_model(data=data)
        self._save_model(model=model)


if __name__ == "__main__":
    doc2vec = Doc2vec(
        data_dump_data_dir="/Users/ceyhuncakir/Documents/github/INFOMAIR/1a/data/wikipedia",
        checkpoint_data_dir="/Users/ceyhuncakir/Documents/github/INFOMAIR/1a/gensim/doc2vec.model"
    ).run()