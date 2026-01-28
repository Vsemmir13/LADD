import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from typing import Optional, Dict, Any
import os


def normalize(x):
    return x / x.abs().max(dim=0)[0][None, ...]


def velocity_from_denoiser(
    x,
    model,
    sigma,
    class_labels=None,
    error_eps=1e-4,
    stochastic=False,
    cfg=0.0,
    **model_kwargs,
):
    sigma = sigma[:, None, None, None]
    cond_v = (-model(x, sigma, class_labels, **model_kwargs) + x) / (sigma + error_eps)
    if cfg > 0.0:
        dummy_labels = torch.zeros_like(class_labels)
        dummy_labels[:, -1] = 1
        uncond_v = (-model(x, sigma, dummy_labels, **model_kwargs) + x) / (
            sigma + error_eps
        )
        v = cond_v + cfg * (cond_v - uncond_v)
    else:
        v = cond_v
    if stochastic:
        v = v * 2
    return v


def get_timesteps(params):
    num_steps = params["num_steps"]
    sigma_min, sigma_max = params["sigma_min"], params["sigma_max"]
    rho = params["rho"]
    step_indices = torch.arange(num_steps, device=params["device"])
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    return t_steps


def sample_euler(model, noise, params, class_labels=None, **model_kwargs):
    num_steps = params["num_steps"]
    vis_steps = params["vis_steps"]
    t_steps = get_timesteps(params)
    x = noise * params["sigma_max"]
    x_history = [normalize(noise)]
    with torch.no_grad():
        for i in range(len(t_steps) - 1):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            t_net = t_steps[i] * torch.ones(x.shape[0], device=params["device"])
            if i >= params["iter_start"] and i <= params["iter_end"]:
                x = x + velocity_from_denoiser(
                    x,
                    model,
                    t_net,
                    class_labels=class_labels,
                    stochastic=params["stochastic"],
                    cfg=params["cfg"],
                    **model_kwargs,
                ) * (t_next - t_cur)
            else:
                x = x + velocity_from_denoiser(
                    x,
                    model,
                    t_net,
                    class_labels=class_labels,
                    stochastic=params["stochastic"],
                    cfg=0.0,
                    **model_kwargs,
                ) * (t_next - t_cur)
            if params["stochastic"]:
                x = x + torch.randn_like(x) * torch.sqrt(
                    torch.abs(t_next - t_cur) * 2 * t_cur
                )
            x_history.append(normalize(x).view(-1, 3, *x.shape[2:]))
    x_history = (
        [x_history[0]]
        + x_history[:: -(num_steps // (vis_steps - 2))][::-1]
        + [x_history[-1]]
    )
    return x, x_history


class GeneratedDistillationDataset(Dataset):

    def __init__(
        self,
        teacher,
        num_samples: int,
        params: Dict[str, Any],
        img_channels: int = 3,
        img_resolution: int = 32,
        label_dim: int = 11,
        regenerate: bool = False,
    ):
        self.save_path = params["save_path"]
        if not regenerate and os.path.exists(self.save_path):
            print(f"[Dataset] Loading dataset from {self.save_path}")
            data = torch.load(self.save_path, map_location="cpu")
            self.latents = data["latents"]
            self.condition = data["condition"]
            self.noise = data["noise"]
            return
        print(f"[Dataset] Generating dataset → {self.save_path}")
        self.latents = []
        self.condition = []
        self.noise = []
        teacher = teacher.to(params["device"]).eval()
        generated = 0
        with torch.inference_mode():
            while generated < num_samples:
                bs = min(params["batch_size"], num_samples - generated)
                labels = torch.randint(0, label_dim - 1, (bs,), device=params["device"])
                cond = (
                    labels[:, None]
                    == torch.arange(label_dim, device=params["device"])[None, :]
                ).float()
                noise = torch.randn(
                    bs,
                    img_channels,
                    img_resolution,
                    img_resolution,
                    device=params["device"],
                )
                samples, _ = sample_euler(
                    model=teacher,
                    noise=noise,
                    params=params,
                    class_labels=cond,
                )
                self.latents.append(samples.cpu())
                self.condition.append(cond.cpu())
                self.noise.append(noise.cpu())
                generated += bs
        self.latents = torch.cat(self.latents, dim=0)
        self.condition = torch.cat(self.condition, dim=0)
        self.noise = torch.cat(self.noise, dim=0)
        torch.save(
            {"latents": self.latents, "condition": self.condition, "noise": self.noise},
            self.save_path,
        )

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, idx):
        return {
            "latents": self.latents[idx],
            "condition": self.condition[idx],
            "noise": self.noise[idx],
        }


class LatentDataModule(L.LightningDataModule):

    def __init__(self, dataset: Dataset, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
