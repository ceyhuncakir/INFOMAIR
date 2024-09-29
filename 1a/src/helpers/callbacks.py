import os

import typer

def path_valid(value: str) -> str:
    if not os.path.exists(value):
        raise typer.BadParameter("The path does not exist.")
    return value

def vectorizer_value(value: str) -> str:

    values = ["tfidf", "count"]

    if value not in values:
        raise typer.BadParameter(f"{value} does not meet the criteria: {values}")
    return value
