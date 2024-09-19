import typer

from baselines.baseline_1 import baseline_1_app
from baselines.baseline_2 import baseline_2_app
from ml.xgboost import xgboost_app
from ml.decisiontree import decisiontree_app
from ml.mlp import mlp_app
from helpers.doc2vec import doc2vec_app

app = typer.Typer()
app.add_typer(baseline_1_app, name="baseline_1", help="Manages the baseline 1 implementation")
app.add_typer(baseline_2_app, name="baseline_2", help="Manages the baseline 2 implementation")
app.add_typer(xgboost_app, name="xgboost", help="Manages the xgboost implementation")
app.add_typer(decisiontree_app, name="decisiontree", help="Manages the decision tree implementation")
app.add_typer(mlp_app, name="mlp", help="Manages the mlp implementation")
app.add_typer(doc2vec_app, name="doc2vec", help="Manages the doc2vec implementation")

if __name__ == "__main__":
    app()

