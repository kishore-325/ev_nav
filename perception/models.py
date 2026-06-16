import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
from perception.ConvLSTM_pytorch.convlstm import ConvLSTM

class DecoderBlock(nn.Module):
    #Bilinerar upsample -> concat skip -> 2_conv+BN+ReLU
    
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch+skip_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = F.relu(self.bn1(self.conv1(torch.cat([x, skip], dim=1))))
        return F.relu(self.bn2(self.conv2(x)))

class EfficientNetB2UNet(nn.Module):
    """
    EfficientNet-B2 encoder + UNet-style decoder with ConvLSTM at bottleneck.

    Input:  (N, 1, 260, 346)  single-channel log-diff
    Output: (N, 1, 260, 346)  depth map (raw logits, apply sigmoid externally)

    Encoder feature sizes for input 260×346:
        f0:  16ch, 130×173
        f1:  24ch,  65×87
        f2:  48ch,  33×44
        f3: 120ch,  17×22
        f4: 352ch,   9×11  ← ConvLSTM here
    """

    def __init__(self, input_mode=1, evs_min_cutoff=0,
                 lstm_hidden_dim=512, lstm_kernel_size=1):
        super().__init__()
        self.input_mode = input_mode
        self.evs_min_cutoff = evs_min_cutoff
        in_ch = 2 if input_mode == 1 else 1

        self.encoder = timm.create_model(
            'efficientnet_b2',
            pretrained=False,
            in_chans = in_ch,
            features_only = True,
            out_indices = (0, 1, 2, 3, 4),
        )

        self.lstm = ConvLSTM(
            input_dim=352,
            hidden_dim=[lstm_hidden_dim],
            kernel_size=(lstm_kernel_size, lstm_kernel_size),
            num_layers =1,
            batch_first=True,
            bias=False,
            return_all_layers=False,
        )

        self.dec4 = DecoderBlock(lstm_hidden_dim, 120, 256)
        self.dec3 = DecoderBlock(256, 48, 128)
        self.dec2 = DecoderBlock(128, 24, 64)
        self.dec1 = DecoderBlock(64, 16, 32)

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

        
    def _form_input(self, x):

        if self.input_mode == 1:
            x = x.clone()
            x[x.abs() < self.evs_min_cutoff] = 0.0
            ch_neg = torch.where(x <0, x.abs(), torch.zeros_like(x))[:, 0]
            ch_pos = torch.where(x>0, x, torch.zeros_like(x))[:, 0]
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
            depth: (N, 1, 260, 346) raw logits — apply sigmoid externally
            h_new: updated ConvLSTM hidden state
        """

        im = self._form_input(x)                       # (N,  2, 260, 346)          

        #Encoder
        f0, f1, f2, f3, f4 = self.encoder(im)         # 16, 24, 48, 120, 352 ch

        #ConvLSTM at bottleneck
        f4_seq, h_new = self.lstm(f4.unsqueeze(0), h)
        f4 = f4_seq[0].squeeze(0)

        #Decoder
        d = self.dec4(f4, f3)                         # (N, 256, 17, 22)
        d = self.dec3(d, f2)                          # (N, 128, 33, 44)
        d = self.dec2(d, f1)                          # (N, 64, 65, 87)
        d = self.dec1(d, f0)                          # (N, 32, 130, 173)

        y = self.out_conv(d)                          # (N, 1, 130, 173)
        y = F.interpolate(y, size=(260, 346), mode="bilinear", align_corners=False)
        return y, h_new
          
          
