import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def _sobel_kernels(device):
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device) / 8.0
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=device) / 8.0
    kx = kx.view(1, 1, 3, 3)
    ky = ky.view(1, 1, 3, 3)
    return kx, ky


def _laplace_kernel(device):
    k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device)
    return k.view(1, 1, 3, 3)


def physics_loss(pred_u, pred_v, pressure=None, nu=1e-3):
    device = pred_u.device
    kx, ky = _sobel_kernels(device)
    lap = _laplace_kernel(device)

    def grad(field, kernel):
        return F.conv2d(field, kernel, padding=1)

    du_dx = grad(pred_u, kx)
    du_dy = grad(pred_u, ky)
    dv_dx = grad(pred_v, kx)
    dv_dy = grad(pred_v, ky)

    div = du_dx + dv_dy
    loss_div = (div ** 2).mean()

    loss_mom = 0.0
    if pressure is not None:
        dp_dx = grad(pressure, kx)
        dp_dy = grad(pressure, ky)
        lap_u = F.conv2d(pred_u, lap, padding=1)
        lap_v = F.conv2d(pred_v, lap, padding=1)
        mom_u = pred_u * du_dx + pred_v * du_dy + dp_dx - nu * lap_u
        mom_v = pred_u * dv_dx + pred_v * dv_dy + dp_dy - nu * lap_v
        loss_mom = (mom_u ** 2).mean() + (mom_v ** 2).mean()

    return loss_div + loss_mom
