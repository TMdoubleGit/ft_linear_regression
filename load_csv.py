import pandas as pd


def load(path: str) -> pd.core.frame.DataFrame:
    """
    Load a csv file, prints its shape and returns it.
    """
    try:
        dataset = pd.read_csv(path)
        print(f"Loading dataset of dimensions {dataset.shape}")
        return (dataset)

    except Exception as e:
        print(f"Error: {e}")
