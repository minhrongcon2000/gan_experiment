import torch


class WGANLoss(torch.nn.Module):
    def forward(self, 
                fake_pred, 
                real_pred=None, 
                interpolate=None, 
                l: float=10.):
        if real_pred is None:
            return -torch.mean(fake_pred)
        
        gp = 0
        if interpolate is not None:
            interpolate_img, interpolate_pred = interpolate
            gradient = torch.autograd.grad(
                inputs=interpolate_img,
                outputs=interpolate_pred,
                grad_outputs=torch.ones_like(interpolate_pred),
                create_graph=True
            )[0]
            gradient = gradient.view(gradient.shape[0], -1)
            gp = torch.mean((gradient.norm(2, dim=-1) - 1) ** 2)
        
        return torch.mean(real_pred - fake_pred) - l * gp
