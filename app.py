"""Entrypoint. `python app.py` for local dev, `gunicorn app:app` in production."""
from __future__ import annotations

import os

from cissou import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"CISSOU is listening on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
