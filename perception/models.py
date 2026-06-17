import torch
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

class AlexNetUNet(nn.Module):
    """
    AlexNet encoder + UNet-style decoder with ConvLSTM at bottleneck.

    Input:  (N, 1, 260, 346)  single-channel log-diff
    Output: (N, 1, 260, 346)  depth map (raw logits, apply sigmoid externally)

    Encoder feature sizes for input 260×346:
        e0:  96ch,  63×84   (after conv1, before pool1)
        e1: 256ch,  31×41   (after conv2, before pool2)
        e2: 384ch,  15×20   (after conv3)
        e3: 256ch,   7×9    ← ConvLSTM here (after conv5 + pool3)
    """

    def __init__(self, input_mode=1, evs_min_cutoff=0,
                 lstm_hidden_dim=512, lstm_kernel_size=1):
        super().__init__()
        self.input_mode = input_mode
        self.evs_min_cutoff = evs_min_cutoff
        in_ch = 2 if input_mode == 1 else 1

        self.conv1 = nn.Conv2d(in_ch, 96, kernel_size=11, stride=4, padding=0)
        self.pool1 = nn.MaxPool2d(3,2)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(3,2)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(3,2)

        self.lstm = ConvLSTM(
            input_dim=256,
            hidden_dim=[lstm_hidden_dim],
            kernel_size=(lstm_kernel_size, lstm_kernel_size),
            num_layers =1,
            batch_first=True,
            bias=False,
            return_all_layers=False,
        )

        self.dec3 = DecoderBlock(lstm_hidden_dim, 384, 256)
        self.dec2 = DecoderBlock(256, 256, 128)
        self.dec1 = DecoderBlock(128, 96, 64)

        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

        
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
        e0 = F.relu(self.conv1(im))
        e1 = F.relu(self.conv2(self.pool1(e0)))
        e2 = F.relu(self.conv3(self.pool2(e1)))
        e2_out = F.relu(self.conv4(e2))
        e3 = self.pool3(F.relu(self.conv5(e2_out)))

        #ConvLSTM at bottleneck
        e3_seq, h_new = self.lstm(e3.unsqueeze(0), h)
        e3 = e3_seq[0].squeeze(0)

        #Decoder
        d = self.dec3(e3, e2)                          # (N, 256, 15, 20)
        d = self.dec2(d,  e1)                         # (N, 128, 31, 42)
        d = self.dec1(d,  e0)                         # (N,  64, 64, 85)

        y = self.out_conv(d)                          # (N,   1, 64, 85)
        y = F.interpolate(y, size=(260, 346), mode="bilinear", align_corners=False)
        return y, h_new
          
          
