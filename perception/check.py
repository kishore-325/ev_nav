import torch
import timm

encoder = timm.create_model(
    'resnext50_32x4d',
    pretrained=False,
    in_chans = 2,
    features_only = True,
    out_indices = (0,1,2,3,4),
)

x = torch.zeros(1, 2, 260, 346)
feats = encoder(x)

for i, f in enumerate(feats):
    print(f"f{i}: {f.shape}")