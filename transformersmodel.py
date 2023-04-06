import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, c, num_heads,dropout=0.2):
        super().__init__()
        self.q = nn.Sequential(nn.Linear(c, c, bias=False),
                    nn.Dropout(dropout))
        self.k = nn.Sequential(nn.Linear(c, c, bias=False),
                    nn.Dropout(dropout))
        self.v = nn.Sequential(nn.Linear(c, c, bias=False),
                    nn.Dropout(dropout))
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Sequential(nn.Linear(c, c, bias=False),
                    nn.Dropout(dropout))
        self.fc2 = nn.Sequential(nn.Linear(c, c, bias=False),
                    nn.Dropout(dropout))

    def forward(self, x):
        q1=self.q(x)
        k1=self.k(x)  
        v1=self.v(x)    
        y = self.ma(q1, k1, v1)[0]
        x = y + x
        x = self.fc2(self.fc1(x)) + x
        return x

class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers=1,dropout=0.2):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = nn.Conv2d(c1, c2,kernel_size=3,stride=1)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads,dropout) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

