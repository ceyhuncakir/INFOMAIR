import typer

from dialog_manager import dialog_manager_app

app = typer.Typer()

app.add_typer(dialog_manager_app, name="dialog_manager", help="Manages the dialog manager implementation")

if __name__ == "__main__":
    app()