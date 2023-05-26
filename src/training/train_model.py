import os

import yaml

from src.training.training_utils import training_info


@training_info
def train():
    ROOT_DIR = os.path.abspath(os.curdir)
    print(ROOT_DIR)

    # Read yaml config.
    with open(os.path.join(ROOT_DIR, "src/model_params/small.yaml"), "rb") as f:
        model_params = yaml.safe_load(f.read())

    batch_size = model_params["batch_size"]

    print(batch_size)

    return


if __name__ == "__main__":
    train()
