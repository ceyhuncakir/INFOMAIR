import os

import typer

def path_valid(value: str) -> str:
    if not os.path.exists(value):
        raise typer.BadParameter("The path does not exist. check the path and try again.")
    return value

def vectorizer_value(value: str) -> str:

    values = ["tfidf", "count"]

    if value not in values:
        raise typer.BadParameter(f"{value} does not meet the criteria: {values}")
    return value

def deduplication_value(value: str) -> bool:
    
    values = ["True", "False"]

    if value not in values:
        raise typer.BadParameter(f"{value} does not meet the criteria: {values}")
    return value

def experiment_value(value: str) -> str:

    if value == None:
        raise typer.BadParameter(f"You need to specify the experiment name.")
    return value