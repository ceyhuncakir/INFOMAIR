import typer

from dialog import dialog_manager_app

app = typer.Typer()

app.add_typer(dialog_manager_app, name="dialog_manager", help="Manages the statistics implementation")

if __name__ == "__main__":
    app()
