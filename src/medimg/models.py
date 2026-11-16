"""Model zoo: Classifiers (ResNet18, ResNet50, GoogLeNet), Segmenters (UNet, ResNetUNet),
Detectors (RetinaNet, Faster R-CNN), Generative VAE, and Clinical Transformer Summary Decoder.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

def classifier(arch="resnet18", pretrained=False, n_out=1):
    """Builds ResNet18, ResNet50, or GoogLeNet classifier."""
    w = "DEFAULT" if pretrained else None
    if arch in ("resnet18", "resnet50"):
        m = getattr(torchvision.models, arch)(weights=w)
        m.fc = nn.Linear(m.fc.in_features, n_out)
    elif arch == "googlenet":
        m = torchvision.models.googlenet(weights=w)
        m.aux_logits = False
        m.aux1 = None
        m.aux2 = None
        m.fc = nn.Linear(m.fc.in_features, n_out)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return m

resnet_classifier = classifier

def _block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(True))

class UNet(nn.Module):
    """Compact U-Net encoder/decoder with skip connections."""
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

class ResNetUNet(nn.Module):
    """U-Net with pre-trained ResNet18 encoder for segmentation."""
    def __init__(self, pretrained=True):
        super().__init__()
        w = "DEFAULT" if pretrained else None
        res = torchvision.models.resnet18(weights=w)
        self.init_conv = nn.Sequential(res.conv1, res.bn1, res.relu) # (64, H/2, W/2)
        self.maxpool = res.maxpool
        self.layer1 = res.layer1 # (64, H/4, W/4)
        self.layer2 = res.layer2 # (128, H/8, W/8)
        self.layer3 = res.layer3 # (256, H/16, W/16)
        self.layer4 = res.layer4 # (512, H/32, W/32)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True))

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True))

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True))

        self.up0 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x0 = self.init_conv(x)
        x1 = self.layer1(self.maxpool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.dec4(torch.cat([self.up4(x4), x3], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), x2], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), x1], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), x0], 1))
        out = self.head(self.up0(d1))
        return out

def retinanet(num_classes=2, pretrained_backbone=False):
    from torchvision.models.detection import retinanet_resnet50_fpn
    return retinanet_resnet50_fpn(weights=None, weights_backbone="DEFAULT" if pretrained_backbone else None,
                                  num_classes=num_classes, min_size=128, max_size=128)

def fasterrcnn(num_classes=2, pretrained_backbone=False):
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    return fasterrcnn_resnet50_fpn(weights=None, weights_backbone="DEFAULT" if pretrained_backbone else None,
                                   num_classes=num_classes, min_size=128, max_size=128)

def dice_bce_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3)); den = p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    return F.binary_cross_entropy_with_logits(logits, target) + (1 - (2 * inter + eps) / (den + eps)).mean()

# --- VAE for Minority Class Augmentation ---
class ConvVAE(nn.Module):
    """Convolutional VAE for melanoma generation and latent space visualization."""
    def __init__(self, in_ch=3, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(0.2, True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, True),
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

        self.dec_fc = nn.Linear(latent_dim, 256 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, in_ch, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z).view(-1, 256, 8, 8)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon, x, mu, logvar, kld_weight=1e-3):
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    return recon_loss + kld_weight * kld_loss, recon_loss, kld_loss

# --- Transformer Summary Module ---
class TransformerSummary(nn.Module):
    """Transformer decoder that conditions on structured triage metrics and generates a textual summary."""
    def __init__(self, vocab_size=64, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.feature_proj = nn.Linear(4, d_model) # input features: [score, area_frac, box_count, confidence]
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(128, d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, metrics, tgt_tokens):
        memory = self.feature_proj(metrics).unsqueeze(1) # (B, 1, d_model)
        B, T = tgt_tokens.shape
        pos = torch.arange(0, T, device=tgt_tokens.device).unsqueeze(0)
        x = self.token_emb(tgt_tokens) + self.pos_emb(pos)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(T, device=tgt_tokens.device)
        out = self.transformer(tgt=x, memory=memory, tgt_mask=tgt_mask)
        return self.head(out)
