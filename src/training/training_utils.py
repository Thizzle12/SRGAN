from functools import wraps
import torch


def training_info(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            print(f'{"-" * 40}')
            print(f"Started Training")
            print(f'{"-" * 40}')
            print(f"Torch version {torch.__version__}")
            print(f"CUDA is available.")
            print(f"Running CUDA version {torch.version.cuda}.")
            print(f'{"-" * 40}')
            return func(*args, **kwargs)

    return wrapper
