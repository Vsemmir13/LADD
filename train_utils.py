from dataclasses import dataclass
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import lightning as L
from models import DiscriminatorHead, DiscHead
from dnnlib import util
from edm.torch_utils import misc
import pickle
from dataset import velocity_from_denoiser, get_timesteps
import math
import os


def q_sample(x0: torch.Tensor, t: torch.Tensor, params: DiffusionParams, noise=None):
    """
    Add noise to x0 at timesteps t.
    x_t = sqrt(alpha_cumprod[t]) * x0 + sqrt(1 - alpha_cumprod[t]) * noise
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_acp = params.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
    sqrt_1macp = (1 - params.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
    return sqrt_acp * x0 + sqrt_1macp * noise, noise


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    def copy_to(self, model: nn.Module):
        model.load_state_dict({k: self.shadow[k].clone() for k in self.shadow})


class LADDDistillationModule(L.LightningModule):
    def __init__(self, teacher: nn.Module, config: dict):
        super().__init__()
        self.config = config
        self.automatic_optimization = False  # easier for GAN training

        self.teacher: nn.Module = teacher
        self.student: nn.Module = copy.deepcopy(teacher)
        self.teacher.eval()
        self.teacher.requires_grad_(False)

        self._build_discriminators()

        self.vae = None
        if self.vae is not None:
            self.vae.eval()

        self.save_hyperparameters(ignore=["teacher"])

    def setup(self, *args, **kwargs):
        super().setup(*args, **kwargs)
        self.ema = EMA(self.student)

    def configure_optimizers(self):
        opt_g = torch.optim.AdamW(
            self.student.parameters(), lr=self.config["student_lr"]
        )
        opt_d = torch.optim.AdamW(
            self.disc_heads.parameters(), lr=self.config["discriminator_lr"]
        )
        return [opt_g, opt_d]


    def sample_discriminator_timesteps(self, batch_size, device):
        """
        Sample sigma values for discriminator renoising.
        Uses logit-normal distribution mapped to [sigma_min, sigma_max].
        """
        sigma_min = self.config["sigma_min"]
        sigma_max = self.config["sigma_max"]

        loc = self.config.get("disc_sigma_loc", 0.0)
        scale = self.config.get("disc_sigma_scale", 1.0)

        norm_samples = torch.randn(batch_size, device=device) * scale + loc
        t_uniform = torch.sigmoid(norm_samples)  # Maps to (0, 1)

        log_sigma_min = math.log(sigma_min)
        log_sigma_max = math.log(sigma_max)
        log_sigma_range = log_sigma_max - log_sigma_min

        log_sigma = log_sigma_min + t_uniform * log_sigma_range
        sigma = torch.exp(log_sigma)

        return sigma

    def _build_discriminators(self):
        with torch.no_grad():
            dummy_x = torch.randn(1, 3, 32, 32, device=self.device)
            dummy_t = torch.rand(1, device=self.device)
            dummy_y = torch.zeros(1, self.teacher.label_dim, device=self.device)
            dummy_y[:, -1] = 1.0

            _, feats = self.teacher(dummy_x, dummy_t, dummy_y, return_features=True)

        disc_heads = []
        for f in feats:
            disc_heads.append(DiscHead(f.shape[1]).to(self.device))

        self.disc_heads = nn.ModuleList(disc_heads)

    def add_noise(self, x_start, sigma, noise=None):
        """
        EDM-style noise addition: x_noisy = x_clean + sigma * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        if isinstance(sigma, (int, float)):
            sigma = torch.full((x_start.shape[0],), sigma, device=x_start.device)

        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1, 1)

        return x_start + sigma * noise, noise

    def forward_teacher_discriminator(self, latents, t_renoise, condition):
        _, features = self.teacher(latents, t_renoise, condition, return_features=True)

        selected_features = features[-len(self.disc_heads) :]
        logits = []

        for feat, head in zip(selected_features, self.disc_heads):
            logits.append(head(feat))  # head returns [B, 1]

        return torch.cat(logits, dim=1)  # [B, num_heads]

    def grad_norm(self, parameters, norm_type=2.0):
        params = [p for p in parameters if p.grad is not None]
        if len(params) == 0:
            return torch.tensor(0.0, device=self.device)

        device = params[0].grad.device
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).to(device) for p in params]
            ),
            norm_type,
        )
        return total_norm

    def multistep_sample(self, model, noise, condition):
        """
        Sample function for EDM
        """
        # t_steps = get_timesteps(params)
        num_steps = self.config["num_student_steps"]
        sigma_min, sigma_max = self.config["sigma_min"], self.config["sigma_max"]
        rho = self.config["rho"]
        step_indices = torch.arange(num_steps, device="cuda")
        t_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])

        x = noise * self.config["sigma_max"]
        for i in range(len(t_steps) - 1):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            t_net = torch.full((x.shape[0],), t_cur, device=self.device)
            x = x + velocity_from_denoiser(x, model, t_net, class_labels=condition) * (
                t_next - t_cur
            )

        return x

    def get_alpha_sigma(self, t):
        t_expand = t.view(-1, 1, 1, 1)

        if self.config["prediction_type"] == "flow_matching":
            return 1.0 - t_expand, t_expand
        else:
            T = self.diffusion_params.alphas_cumprod.shape[0]
            t_idx = (t * (T - 1)).long().clamp(0, T - 1)
            alpha_bar = self.diffusion_params.alphas_cumprod[t_idx].view(-1, 1, 1, 1)
            return torch.sqrt(alpha_bar), torch.sqrt(1.0 - alpha_bar)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        clean_latents = batch["latents"]
        condition = batch["condition"]
        batch_size = clean_latents.shape[0]

        noise = batch["noise"]  # this is noise from teacher generation!

        fake_latents = self.multistep_sample(
            model=self.student, noise=noise, condition=condition
        )

        t_renoise_real = self.sample_discriminator_timesteps(batch_size, self.device)
        t_renoise_fake = self.sample_discriminator_timesteps(batch_size, self.device)
        noise_real = torch.randn_like(clean_latents)
        noise_fake = torch.randn_like(clean_latents)
        real_renoised, _ = self.add_noise(clean_latents, t_renoise_real, noise_real)
        fake_renoised, _ = self.add_noise(
            fake_latents.detach(), t_renoise_fake, noise_fake
        )

        # DISCRIMINATOR UPDATE
        self.toggle_optimizer(opt_d)

        real_logits = self.forward_teacher_discriminator(
            real_renoised, t_renoise_real, condition
        )
        fake_logits = self.forward_teacher_discriminator(
            fake_renoised, t_renoise_fake, condition
        )

        # Hinge loss
        loss_d_real = F.softplus(-real_logits).mean()
        loss_d_fake = F.softplus(fake_logits).mean()
        loss_d = loss_d_real + loss_d_fake

        self.manual_backward(loss_d)

        torch.nn.utils.clip_grad_norm_(self.disc_heads.parameters(), 1.0)

        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # STUDENT UPDATE
        self.toggle_optimizer(opt_g)

        # Re-compute fake renoised with gradients
        fake_renoised_g, _ = self.add_noise(fake_latents, t_renoise_fake)
        fake_logits_g = self.forward_teacher_discriminator(
            fake_renoised_g, t_renoise_fake, condition
        )

        loss_adv = F.softplus(-fake_logits_g).mean()

        loss_distill = F.mse_loss(fake_latents, clean_latents)

        loss_g = self.get_adv_weight() * loss_adv + loss_distill

        self.manual_backward(loss_g)

        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)

        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        self.log_dict(
            {
                "train/loss_d": loss_d,
                "train/loss_d_real": loss_d_real,
                "train/loss_d_fake": loss_d_fake,
                "train/loss_g": loss_g,
                "train/loss_adv": loss_adv,
                "train/loss_distill": loss_distill,
                "train/real_logits": real_logits.mean(),
                "train/fake_logits": fake_logits.mean(),
            },
            on_step=True,
            sync_dist=True,
        )

        if self.global_step % 500 == 0:
            save_dir = self.config["latents_path"]
            os.makedirs(save_dir, exist_ok=True)
            true_latents = self.multistep_sample(
                model=self.teacher, noise=noise, condition=condition
            )
            torch.save(
                {
                    "step": self.global_step,
                    "clean_latents": clean_latents.detach().cpu(),
                    "true_latents": true_latents.detach().cpu(),
                    "fake_latents": fake_latents.detach().cpu(),
                    "condition": condition.detach().cpu(),
                },
                f"{save_dir}/latents_step_{self.global_step}.pt",
            )

    def get_adv_weight(self):
        warmup_steps = self.config.get("warmup_steps", 500)
        max_adv_weight = self.config.get("adv_weight", 0.1)

        if self.global_step < warmup_steps:
            return 0.0
        else:
            return max_adv_weight

    def on_train_batch_end(self, *args, **kwargs):
        if hasattr(self, "ema"):
            self.ema.update(self.student)
