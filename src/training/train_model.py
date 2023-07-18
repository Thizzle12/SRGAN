import os

import yaml
from tqdm import tqdm

import torch
from torch.optim import AdamW
from torch.nn import MSELoss
from torch.utils.data.dataloader import DataLoader

from src.model.data_handler import DatasetHandler
from src.model.srgan import SRGenerator, SRDiscriminator
from src.training.training_utils import training_info


@training_info
def train():
    # ROOT_DIR = os.path.abspath(os.curdir)
    # print(ROOT_DIR)

    # # Read yaml config.
    # with open(os.path.join(ROOT_DIR, "src/model_params/small.yaml"), "rb") as f:
    #     model_params = yaml.safe_load(f.read())

    # batch_size = model_params["batch_size"]

    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    epochs = 10

    # Build train dataset.
    data = build_dataset(device=device)

    # Build models.
    G_model = SRGenerator(n_channels=1).to(device)
    D_model = SRDiscriminator(in_channels=1).to(device)

    # Optimizers and loss functions for both discriminator and generator.
    # Loss function should contain content loss which is the euclidian distance between feature maps from a pre-trained model like ResNet
    # for both the grund truth image and the generated image. MSE is not enough.
    criterion = MSELoss()

    G_optim = AdamW(
        G_model.parameters(),
        lr=1e-4,
    )

    D_optim = AdamW(
        D_model.parameters(),
        lr=1e-4,
    )

    # Training
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    real_label = 1
    fake_label = 0

    # Training loop.
    for epoch in range(epochs):
        epoch_mean_loss = []

        # Run through batches.
        for batch_idx, (data, target) in enumerate(tqdm(data)):
            low_res = data.to(device)
            high_res = target.to(device)

            b_size = high_res.shape[0]

            # Train Discriminator. Max log(D(real)) + log(1 - D(G(z)))
            ## Train with all-real batch
            fake = G_model(low_res)

            # Log(D(real))
            disc_real = D_model(high_res).view(-1)
            lossD_real = criterion(disc_real, torch.ones_like(disc_real))

            # Log(1 - D(G(z)))
            disc_fake = D_model(fake.detach()).view(-1)
            lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

            # Backporpagation discriminator.
            lossD = (lossD_real + lossD_fake) / 2
            D_model.zero_grad()
            lossD.backward()
            D_optim.step()

            # Train Generator. Minimize log(1- D(G(z))), same as max log(D(G(z)))
            output = D_model(fake).view(-1)
            lossG = criterion(output, torch.ones_like(output))
            G_model.zero_grad()
            lossG.backward()
            G_optim.step()

            # Output training stats
            if batch_idx % 50 == 0:
                print(f"Discriminator loss: {lossD}")
                print(f"Generator loss: {lossG}")

            # Save Losses for plotting later
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())


def build_dataset(
    batch_size: int = 8,
    split: str = "train",
    device: str = "cuda",
):
    # dataset = DatasetHandler(
    #     files_path="D:\location_data",
    # )

    dataset = DatasetHandler(
        files_path="D:\img_align_celeba",
    )

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
    )

    return train_loader


if __name__ == "__main__":
    train()
