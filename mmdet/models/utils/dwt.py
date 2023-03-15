import torch
from torch import nn
import pywt
import numpy as np
import torch.nn.functional as F


class DWTForward(nn.Module):
    def __init__(self, wave_name="haar"):
        super(DWTForward, self).__init__()
        wavelet = pywt.Wavelet(wave_name)
        ll = np.outer(wavelet.dec_lo, wavelet.dec_lo)
        lh = np.outer(wavelet.dec_hi, wavelet.dec_lo)
        hl = np.outer(wavelet.dec_lo, wavelet.dec_hi)
        hh = np.outer(wavelet.dec_hi, wavelet.dec_hi)
        filters = np.stack([ll[None, :, :], lh[None, :, :],
                            hl[None, :, :], hh[None, :, :]],
                           axis=0)
        self.weight = nn.Parameter(
            # torch.tensor(filters).to(torch.get_default_dtype()),
            torch.tensor(filters, dtype=torch.float32),
            requires_grad=False)

    def forward(self, x):
        channel = x.shape[1]
        filters = torch.cat([self.weight, ] * channel, dim=0)
        # in tf2 self.strides = [1, 1, 2, 2, 1]
        # x = tf.nn.conv3d(x, self.filter, padding='VALID', strides=self.strides)
        y = F.conv2d(x, filters, groups=channel, stride=2)
        return y


class DWTPool2d(nn.Module):
    def __init__(self, in_channels, wave_name="haar"):
        super(DWTPool2d, self).__init__()
        self.dwt2d = DWTForward(wave_name)
        self.squeeze = nn.Conv2d(in_channels * 4, in_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x = self.dwt2d(x)
        x = self.squeeze(x)
        x = self.bn(x)
        return x
