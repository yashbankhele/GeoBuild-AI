from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = ROOT / "data" / "demo"


def load_sites():
    return pd.read_csv(DEMO_DIR / "sites.csv")


def load_observations():
    return pd.read_csv(DEMO_DIR / "observations.csv", parse_dates=["date"])
