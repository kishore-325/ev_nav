import torch
import torch.nn as nn
import torch.nn.functional as F
from ConvLSTM_pytorch.convlstm import ConvLSTM

class OrigUNet(nn.Module):
    """
    5-layer UNet for log-difference → depth estimation.
    Adapted from evfly/learner/learner_models.py (depth prediction only).

    Input:  (N, 1, 260, 346)  single-channel log-diff
    Output: (N, 1, 260, 346)  depth map in [0, 1]  (×100 → metric metres)
    """

    def __init__(self, input_mode=2, evs_min_cutoff=0):
        super().__init__()
        self.input_mode = input_mode        # 1 = 2-ch polarity, 2 = 1-ch binary mask
        self.evs_min_cutoff = evs_min_cutoff
        self.input_h = 260
        self.input_w = 346

        in_ch = 2 if input_mode == 1 else 1

        #----------Encoder-----------
        self.e11 = nn.Conv2d(in_ch, 32, kernel_size=3, padding=0)  # (N,32,258,344)
        self.e12 = nn.Conv2d(32, 32, kernel_size=3, padding=0)   # (N,32,256,342)
        self.pool1 = nn.MaxPool2d(2,2)                           # (N,32,128,171)

        self.e21 = nn.Conv2d(32, 64, kernel_size=3, padding=0)
        self.e22 = nn.Conv2d(64, 64, kernel_size=3, padding=0)   # (N,64,124,167)
        self.pool2 = nn.MaxPool2d(2, 2)                          # (N,64,62,83)

        self.e31 = nn.Conv2d(64, 128, kernel_size=3, padding=0)  
        self.e32 = nn.Conv2d(128, 128, kernel_size=3, padding=0) # (N,128,58,79)
        self.pool3 = nn.MaxPool2d(2, 2)                          # (N,128,29,39)

        self.e41 = nn.Conv2d(128, 256, kernel_size=3, padding=0)
        self.e42 = nn.Conv2d(256, 256, kernel_size=3, padding=0)  # (N,256,25,35)
        self.pool4 = nn.MaxPool2d(2, 2)                           # (N,256,12,17)

        self.e51 = nn.Conv2d(256, 512, kernel_size=3, padding=0)
        self.e52 = nn.Conv2d(512, 512, kernel_size=3, padding=0)  # (N,512,8,13) bottleneck

        #---------Decoder (skip = bilinear interp)---------
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # (N,256,16,26)
        self.d11 = nn.Conv2d(512, 256, kernel_size=3, padding=0)
        self.d12 = nn.Conv2d(256, 256, kernel_size=3, padding=0)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # (N,128,24,44)
        self.d21 = nn.Conv2d(256, 128, kernel_size=3, padding=0)
        self.d22 = nn.Conv2d(128, 128, kernel_size=3, padding=0)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # (N,64,40,80) → after crop+conv → (N,64,36,76)
        self.d31 = nn.Conv2d(128, 64, kernel_size=3, padding=0)
        self.d32 = nn.Conv2d(64, 64, kernel_size=3, padding=0)

        self.upconv4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)   # (N,32,72,152)
        self.d41 = nn.Conv2d(64, 32, kernel_size=3, padding=0)
        self.d42 = nn.Conv2d(32, 32, kernel_size=3, padding=0)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)     # (N, 1, 68, 148)

        self.nonlin = nn.ReLU()

        # ConvLSTM at bottleneck (1 layer, 1x1 kernel)
        self.lstm = ConvLSTM(input_dim=512, hidden_dim=[512], num_layers=1,
                             kernel_size=(1, 1), bias=False, batch_first=True,
                             return_all_layers=False)

    def _form_input(self, x):
        """
        Preprocess single-channel event frame based on input_mode.
        x: (N, 1, H, W)

        mode 1 (polarity): returns (N, 2, H, W) — channel-0 = |neg|, channel-1 = pos
        mode 2 (bev):      returns (N, 1, H, W) — binary mask, 1 where events exist
        """
        if self.input_mode == 1:
            x = x.clone()
            x[x.abs() < self.evs_min_cutoff] = 0.0
            ch_neg = torch.where(x < 0, x.abs(), torch.zeros_like(x))[:, 0]
            ch_pos = torch.where(x > 0, x, torch.zeros_like(x))[:, 0]
            return torch.stack([ch_neg, ch_pos], dim=1)
        else:  # mode 2
            mask = torch.zeros_like(x)
            mask[x != 0.0] = 1.0
            return mask

    @staticmethod
    def _skip(enc, target_h, target_w):
        """Bilinear interpolate encoder feature map to (target_h, target_w)."""
        return F.interpolate(enc, size=(target_h, target_w), mode='bilinear', align_corners=False)
    
    def forward(self, x, h=None):
        """
        x: (N, 1, 260, 346) single-channel event frame
        h: hidden state for ConvLSTM (None on first call)
        returns: (depth, h_new)
            depth: (N, 1, 260, 346) depth in [0, 1]
            h_new: updated ConvLSTM hidden state
        """
        im = self._form_input(x)    # (N, in_ch, 260, 346)

        # Encoder
        e1 = self.nonlin(self.e12(self.nonlin(self.e11(im))))               # (N,32,256,342)
        e2 = self.nonlin(self.e22(self.nonlin(self.e21(self.pool1(e1)))))   # (N,64,124,167)
        e3 = self.nonlin(self.e32(self.nonlin(self.e31(self.pool2(e2)))))   # (N,128,58,79)
        e4 = self.nonlin(self.e42(self.nonlin(self.e41(self.pool3(e3)))))   # (N,256,25,35)
        e5 = self.nonlin(self.e52(self.nonlin(self.e51(self.pool4(e4)))))   # (N,512,8,13)

        # ConvLSTM at bottleneck
        e5_lstm, h_new = self.lstm(e5.unsqueeze(0), h)  # add seq dim
        e5 = e5_lstm[0].squeeze(0)                       # remove seq dim

        # Decoder
        up1 = self.upconv1(e5)                                              # (N,256,16,26)
        d1 = self.nonlin(self.d12(self.nonlin(self.d11(
            torch.cat([self._skip(e4, 16, 26), up1], dim=1)))))             # (N,256,12,22)

        up2 = self.upconv2(d1)                                              # (N,128,24,44)
        d2 = self.nonlin(self.d22(self.nonlin(self.d21(
            torch.cat([self._skip(e3, 24, 44), up2], dim=1)))))             # (N,128,20,40)

        up3 = self.upconv3(d2)                                              # (N,64,40,80)
        d3 = self.nonlin(self.d32(self.nonlin(self.d31(
            torch.cat([self._skip(e2, 40, 80), up3], dim=1)))))             # (N,64,36,76)

        up4 = self.upconv4(d3)                                              # (N,32,72,152)
        d4 = self.nonlin(self.d42(self.nonlin(self.d41(
            torch.cat([self._skip(e1, 72, 152), up4], dim=1)))))            # (N,32,68,148)

        y = self.out_conv(d4)                                               # (N,1,68,148)

        # Upsample to input resolution
        y = F.interpolate(y, size=(self.input_h, self.input_w), mode='bilinear', align_corners=False)
        return y, h_new                                                      # (N,1,260,346)