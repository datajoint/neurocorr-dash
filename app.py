"""Entry point: ``gunicorn app:server`` or ``python app.py``."""

from neurocorr_dash.app import app, server  # noqa: F401

if __name__ == "__main__":
    app.run(debug=True, port=8052)
