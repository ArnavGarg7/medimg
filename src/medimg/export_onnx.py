"""Export trained classifier / U-Net to ONNX for the C++ application (ONNX Runtime)."""
import argparse, torch
from .models import UNet, resnet_classifier

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["cls", "seg"], required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--size", type=int, default=128); ap.add_argument("--arch", default="resnet18")
    a = ap.parse_args()
    m = resnet_classifier(a.arch) if a.task == "cls" else UNet(3, 16)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=True)); m.eval()
    torch.onnx.export(m, torch.randn(1, 3, a.size, a.size), a.out, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False)
    print("exported", a.out)

if __name__ == "__main__":
    main()
