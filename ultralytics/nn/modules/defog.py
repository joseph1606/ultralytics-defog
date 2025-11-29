import torch
import torch.nn as nn
import torch.nn.functional as F


class DCP(nn.Module):
    """
    Dark Channel Prior defogging module.
    """
    
    def __init__(self):
        super().__init__()
        self.patch_size = 15
        self.omega = 0.95
        self.t0 = 0.1
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W) in range [0, 255]
        Returns:
            Defogged tensor (B, C, H, W) in range [0, 255]
        """
        x_norm = x / 255.0
        
        dark = self._get_dark_channel(x_norm)
        
        atmosphere = self._get_atmosphere(x_norm, dark)
        
        transmission = self._get_transmission(x_norm, atmosphere)
        
        dehazed = self._recover(x_norm, transmission, atmosphere)
        
        return dehazed * 255.0
    
    def _get_dark_channel(self, img):
        """Compute dark channel"""
        dc, _ = torch.min(img, dim=1, keepdim=True)
        
        kernel_size = self.patch_size
        padding = kernel_size // 2
        dc = -F.max_pool2d(-dc, kernel_size, stride=1, padding=padding)
        
        return dc
    
    def _get_atmosphere(self, img, dark):
        b, c, h, w = img.shape
        
        num_pixels = int(h * w * 0.001)
        
        dark_flat = dark.view(b, -1)
        img_flat = img.view(b, c, -1)
        
        _, indices = torch.topk(dark_flat, num_pixels, dim=1)
        
        atmos = torch.zeros(b, c, 1, 1, device=img.device)
        for i in range(b):
            pixels = img_flat[i, :, indices[i]]
            atmos[i] = pixels.max(dim=1, keepdim=True)[0].view(c, 1, 1)
        
        return atmos
    
    def _get_transmission(self, img, atmosphere):
        """Estimate transmission map"""
        norm_img = img / (atmosphere + 1e-8)
        transmission = 1 - self.omega * self._get_dark_channel(norm_img)
        return transmission
    
    def _recover(self, img, transmission, atmosphere):
        """Recover clean image"""
        transmission = torch.clamp(transmission, min=self.t0)
        
        recovered = (img - atmosphere) / transmission + atmosphere
        
        return torch.clamp(recovered, 0, 1)
