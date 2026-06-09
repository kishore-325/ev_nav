import torch
import torch.nn as nn
import torch.nn.functional as F
from perception.ConvLSTM_pytorch.convlstm import ConvLSTM


class BasicBlock(nn.Module):
    """ResNet18 residual block."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class DecoderBlock(nn.Module):
    """Bilinear upsample → concat skip → two conv+BN+ReLU."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = F.relu(self.bn1(self.conv1(torch.cat([x, skip], dim=1))))
        return F.relu(self.bn2(self.conv2(x)))


class ResNet18UNet(nn.Module):
    """
    ResNet18 encoder (trained from scratch) + UNet-style decoder.

    Input:  (N, 1, 260, 346)  single-channel log-diff
    Output: (N, 1, 260, 346)  depth map in [0, 1]  (×100 → metres)

    Spatial sizes (H×W) through the encoder for input 260×346:
        stem  (stride 2)   →  64 ch, 130×173
        pool  (stride 2)   →  64 ch,  65×87
        layer1 (stride 1)  →  64 ch,  65×87
        layer2 (stride 2)  → 128 ch,  33×44
        layer3 (stride 2)  → 256 ch,  17×22
        layer4 (stride 2)  → 512 ch,   9×11  ← ConvLSTM here
    """

    def __init__(self, input_mode=1, evs_min_cutoff=0):
        super().__init__()
        self.input_mode     = input_mode
        self.evs_min_cutoff = evs_min_cutoff
        in_ch = 2 if input_mode == 1 else 1

        # Encoder
        self.stem   = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
        )                                                            # (N,  64, 130, 173)
        self.pool   = nn.MaxPool2d(3, stride=2, padding=1)          # (N,  64,  65,  87)
        self.layer1 = self._make_layer( 64,  64, n=2, stride=1)     # (N,  64,  65,  87)
        self.layer2 = self._make_layer( 64, 128, n=2, stride=2)     # (N, 128,  33,  44)
        self.layer3 = self._make_layer(128, 256, n=2, stride=2)     # (N, 256,  17,  22)
        self.layer4 = self._make_layer(256, 512, n=2, stride=2)     # (N, 512,   9,  11)

        # ConvLSTM at bottleneck
        self.lstm = ConvLSTM(input_dim=512, hidden_dim=[512], num_layers=1,
                             kernel_size=(1, 1), bias=False, batch_first=True,
                             return_all_layers=False)

        # Decoder
        self.dec4 = DecoderBlock(512, 256, 256)   # → (N, 256, 17, 22)
        self.dec3 = DecoderBlock(256, 128, 128)   # → (N, 128, 33, 44)
        self.dec2 = DecoderBlock(128,  64,  64)   # → (N,  64, 65, 87)
        self.dec1 = DecoderBlock( 64,  64,  32)   # → (N,  32, 130, 173)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    @staticmethod
    def _make_layer(in_ch, out_ch, n, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _form_input(self, x):
        if self.input_mode == 1:
            x = x.clone()
            x[x.abs() < self.evs_min_cutoff] = 0.0
            ch_neg = torch.where(x < 0, x.abs(), torch.zeros_like(x))[:, 0]
            ch_pos = torch.where(x > 0, x, torch.zeros_like(x))[:, 0]
            return torch.stack([ch_neg, ch_pos], dim=1)
        else:
            mask = torch.zeros_like(x)
            mask[x != 0.0] = 1.0
            return mask

    def forward(self, x, h=None):
        """
        x: (N, 1, 260, 346)  single-channel log-diff event frame
        h: ConvLSTM hidden state (None on first call)
        returns: (depth, h_new)
            depth: (N, 1, 260, 346) in [0, 1]
            h_new: updated ConvLSTM hidden state
        """
        im = self._form_input(x)               # (N,   2, 260, 346)

        # Encoder
        e0 = self.stem(im)                     # (N,  64, 130, 173)
        e1 = self.layer1(self.pool(e0))        # (N,  64,  65,  87)
        e2 = self.layer2(e1)                   # (N, 128,  33,  44)
        e3 = self.layer3(e2)                   # (N, 256,  17,  22)
        e4 = self.layer4(e3)                   # (N, 512,   9,  11)

        # ConvLSTM at bottleneck
        e4_seq, h_new = self.lstm(e4.unsqueeze(0), h)
        e4 = e4_seq[0].squeeze(0)

        # Decoder
        d = self.dec4(e4, e3)                  # (N, 256,  17,  22)
        d = self.dec3(d,  e2)                  # (N, 128,  33,  44)
        d = self.dec2(d,  e1)                  # (N,  64,  65,  87)
        d = self.dec1(d,  e0)                  # (N,  32, 130, 173)

        y = self.out_conv(d)                   # (N,   1, 130, 173)
        y = F.interpolate(y, size=(260, 346), mode='bilinear', align_corners=False)
        return y, h_new                        # (N,   1, 260, 346)