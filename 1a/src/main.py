import typer

from baselines.baseline_1 import Baseline_1, baseline_1_app
from baselines.baseline_2 import Baseline_2, baseline_2_app
from ml.xgboost import Xgboost, xgboost_app
from ml.mlp import MultiLayerPerceptron, mlp_app

app = typer.Typer()
app.add_typer(baseline_1_app, name="baseline_1", help="Manages the baseline 1 implementation")
app.add_typer(baseline_2_app, name="baseline_2", help="Manages the baseline 2 implementation")
app.add_typer(xgboost_app, name="xgboost", help="Manages the xgboost implementation")
app.add_typer(mlp_app, name="mlp", help="Manages the mlp implementation")

if __name__ == "__main__":
    app()

