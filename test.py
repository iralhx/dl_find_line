from userloss import *
import torch
import torch.nn as nn

input = torch.randn(1, 16, 12, 12)
downsample = nn.Conv2d(16, 16, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
h = downsample(input)
print(h.size())
output = upsample(h, output_size=input.size())
print(output.size())
