import importlib
from datetime import datetime
from pathlib import Path
from hashlib import md5

import pandas as pd


RESULT_DIR = "./results/"

def get_result_dir(
    dataset_version: str,
):
    base_dir = Path(RESULT_DIR) / f"{dataset_version}"
    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)

    return base_dir


def main(
    dataset_version: str,
    trainer_version: str
):
    result_dir = get_result_dir(dataset_version)

    dataset_module_name = f"src.dataset.{dataset_version}"
    dataset_module_path = Path(dataset_module_name.replace(".", "/") + ".py")
    if not dataset_module_path.exists():
        raise FileNotFoundError(f"Dataset module not found: {dataset_module_path}")
    data_hash = md5(dataset_module_path.read_bytes()).hexdigest()
    hash_path = result_dir / "hash.txt"
    if hash_path.exists() and hash_path.read_text() == data_hash and (result_dir / "data.pkl").exists():
        dataset = pd.read_pickle(result_dir / "data.pkl")
        print("** Load train/test data from cache")
    else:
        print("** Hash not found or dataset file has been changed. Load train/test data from source")
        dataset_module = importlib.import_module(dataset_module_name)
        dataset = dataset_module.get_train_test_df()
        pd.to_pickle(dataset, result_dir / "data.pkl")
        hash_path.write_text(data_hash)

    total_df: pd.DataFrame = dataset


    train_dir = result_dir / "train" / trainer_version / datetime.now().strftime("%Y%m%d_%H%M%S")
    train_dir.mkdir(parents=True, exist_ok=True)

    trainer_module_name = f"src.train.{trainer_version}"
    trainer_module_path = Path(trainer_module_name.replace(".", "/") + ".py")
    if not trainer_module_path.exists():
        raise FileNotFoundError(f"Trainer module not found: {trainer_module_path}")
    trainer_module = importlib.import_module(trainer_module_name)

    result = trainer_module.train(total_df, result_dir=train_dir)
    result.to_csv(train_dir / "result.csv", index=False)


if __name__ == "__main__":
    dataset_version = "v1"
    trainer_version = "v1"

    main(dataset_version, trainer_version)
