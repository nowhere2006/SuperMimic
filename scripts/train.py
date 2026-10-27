"""Local training entrypoint; shares the bounded runner with train_smoke.py."""
from train_smoke import main


if __name__ == "__main__":
    raise SystemExit(main())
