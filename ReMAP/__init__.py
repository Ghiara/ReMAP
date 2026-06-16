from pathlib import Path

# Allow imports such as `ReMAP.utils...` to resolve to the repository root
# after `pip install -e .`, without depending on the clone parent directory.
__path__ = [str(Path(__file__).resolve().parent.parent)]
