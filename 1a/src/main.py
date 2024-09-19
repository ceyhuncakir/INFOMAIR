import typer

from baseline_1 import baseline_1_app
from baseline_2 import baseline_2_app
from xgboost import xgboost_app
from decisiontree import decisiontree_app
from mlp import mlp_app

app = typer.Typer()
app.add_typer(baseline_1_app, name="baseline_1", help="Manages the baseline 1 implementation")
app.add_typer(baseline_2_app, name="baseline_2", help="Manages the baseline 2 implementation")
app.add_typer(xgboost_app, name="xgboost", help="Manages the xgboost implementation")
app.add_typer(decisiontree_app, name="decisiontree", help="Manages the decision tree implementation")
app.add_typer(mlp_app, name="mlp", help="Manages the mlp implementation")

if __name__ == "__main__":
    app()

