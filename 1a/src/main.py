import typer

from statistics import statistics_app
from baseline_1 import baseline_1_app
from baseline_2 import baseline_2_app
from vectorizer import vectorizer_app
from logistic_regression import logisticreg_app
from decisiontree import decisiontree_app
from mlp import mlp_app

app = typer.Typer()
app.add_typer(statistics_app, name="statistics", help="Manages the statistics implementation")
app.add_typer(baseline_1_app, name="baseline_1", help="Manages the baseline 1 implementation")
app.add_typer(baseline_2_app, name="baseline_2", help="Manages the baseline 2 implementation")
app.add_typer(vectorizer_app, name="vectorizer", help="Manages the vectorizer implementation")
app.add_typer(logisticreg_app, name="logistic_regression", help="Manages the logistic regression implementation")
app.add_typer(decisiontree_app, name="decisiontree", help="Manages the decision tree implementation")
app.add_typer(mlp_app, name="mlp", help="Manages the mlp implementation")

if __name__ == "__main__":
    app()

