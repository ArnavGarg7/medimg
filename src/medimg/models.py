"""Model zoo: ResNet classifier, compact U-Net, RetinaNet builder."""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

def resnet_classifier(arch="resnet18", pretrained=False, n_out=1):
    w = "DEFAULT" if pretrained else None
    m = getattr(torchvision.models, arch)(weights=w)
    m.fc = nn.Linear(m.fc.in_features, n_out)
    return m

def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

class UNet(nn.Module):
    """U-Net-style encoder/decoder with skip connections. Output: logits (B,1,H,W). H,W divisible by 2**depth."""
    def __init__(self, in_ch=3, base=16, depth=4):
        super().__init__()
        ch = [base * 2 ** k for k in range(depth + 1)]
        self.enc = nn.ModuleList([_block(in_ch if k == 0 else ch[k - 1], ch[k]) for k in range(depth + 1)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[k + 1], ch[k], 2, stride=2) for k in range(depth)])
        self.dec = nn.ModuleList([_block(ch[k] * 2, ch[k]) for k in range(depth)])
        self.head = nn.Conv2d(ch[0], 1, 1)

    def forward(self, x):
        skips = []
        for k, e in enumerate(self.enc):
            x = e(x if k == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        for k in reversed(range(len(self.up))):
            x = self.dec[k](torch.cat([self.up[k](x), skips[k]], 1))
        return self.head(x)

def retinanet(num_classes=2, pretrained_backbone=False):
    """torchvision RetinaNet (ResNet50-FPN). num_classes includes background."""
    from torchvision.models.detection import retinanet_resnet50_fpn
    return retinanet_resnet50_fpn(weights=None, weights_backbone="DEFAULT" if pretrained_backbone else None,
                                  num_classes=num_classes, min_size=128, max_size=128)

def dice_bce_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - (2 * inter + eps) / (den + eps)).mean()
