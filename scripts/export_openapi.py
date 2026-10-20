"""Export the API contract without opening storage or touching user configuration."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from stockregression.api import create_app


def main() -> None:
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "frontend/openapi.json")
    app = create_app("__openapi__", start_jobs=False, openapi_only=True)
    document = app.openapi()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
