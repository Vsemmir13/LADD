import argparse
import os
import torch
import lightning as L
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from utils import load_config
from dataset import GeneratedDistillationDataset
from train_utils import LADDDistillationModule
from models import EDMPrecond


def get_teacher():
    teacher = EDMPrecond(
        img_resolution=32,
        img_channels=3,
        model_type="CUNet",
        noise_channels=128,
        base_factor=64,
        emb_channels=128,
        label_dim=11,
    )
    teacher.load_state_dict(torch.load("weights_and_dataset/EDMPrecond_base.pt"))
    return teacher


if __name__ == "__main__":
    config = load_config("./config.yaml")
    L.seed_everything(42)
    teacher = get_teacher()
    dataset = GeneratedDistillationDataset(
        teacher=teacher,
        num_samples=config["dataset_params"]["size"],
        params=config["dataset_params"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["ladd"]["batch_size"],
        num_workers=config["ladd"]["num_workers"],
    )

    model = LADDDistillationModule(teacher=teacher, config=config["ladd"])

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename="ladd-{step:07d}-{train_loss_g:.4f}",
        monitor="train/loss_g",
        mode="min",
        save_top_k=3,
        save_last=True,
        every_n_train_steps=1000,
    )
    trainer = L.Trainer(
        accelerator="gpu",
        devices=torch.cuda.device_count(),
        strategy="ddp" if torch.cuda.device_count() > 1 else "auto",
        max_steps=config["ladd"]["train_steps"],
        logger=TensorBoardLogger(save_dir="logs/", name="LADD"),
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, dataloader)
