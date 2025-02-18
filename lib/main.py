import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[0] / "spline"))

from spline.engine import train_spline_model

if __name__ == "__main__":
    """Example usage:
    >>> python main.py

    For tensorboard:
    >>> tensorboard --logdir=./logs/ --host 131.230.192.122 --port 8888
    """
    model, datamodule = train_spline_model()
