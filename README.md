# Medical Imaging Triage & Lesion Analysis Workstation (Python / PyTorch)

Single-language project: **NumPy** for the from-scratch fundamentals (steps 3-6), **PyTorch/torchvision** for the vision models (7-9).

```bash
pip install -e ".[dev]"          # add ,torch for phases 7-9
make baseline                    # tests + all fundamentals demos/ablations + synthetic data + leakage-safe split
python -m medimg fundamentals ablation
# phases 7+ (need torch):
python -m medimg.train_cls --root data/synth --epochs 10 --balance
python -m medimg.train_seg --root data/synth
python -m medimg.train_det --root data/synth
```

## Layout
```
src/medimg/
  fundamentals/   tensor_ops (broadcast, SVD, conv2d/3d, numeric grad) | neurons | mlp (backprop, 4 optimizers) | learning_rules | experiments
  synth_data.py   prepare_splits.py   metrics.py        # data curation + evaluation (NumPy only)
  datasets.py models.py train_cls.py train_seg.py train_det.py export_onnx.py common.py   # PyTorch (needs torch)
tests/            pytest: 34 tests
docs/             DATASET_CARD.md  MODEL_CARD.md (templates)
```

## Status vs the 10 steps
| # | Step | Status |
|---|---|---|
| 1 | Use case + acceptance tests | TODO - `docs/REQUIREMENTS.md` |
| 2 | Curation, dedup, leakage-safe split | **Done on synthetic data** (patient+duplicate grouped split, hard audit); real dataset still to plug in |
| 3 | Tensor sandbox, SVD, grad check, visuals | **Done** (Jacobi SVD matches LAPACK to 1e-13; 2D+3D conv figures) |
| 4 | MP/sigmoid neurons, MLP, backprop | **Done** (grad check ~1e-9; matches torch autograd when torch is present) |
| 5 | GD/SGD/Momentum/Nesterov + ablations | **Done for the NumPy MLP** (init, activation, L2, augmentation, normalisation; 5 seeds). Dropout/batch-norm ablations to be done in PyTorch on real images |
| 6 | Hebb/Oja/competitive/delta/RBM | **Done** |
| 7 | ResNet + U-Net + RetinaNet | **Code written, never executed** (torch could not be installed in my sandbox - disk limit). Run it first and expect small fixes |
| 8 | Benchmark detection/classification families | TODO |
| 9 | Transformer summary, VAE augmentation, latent viz | TODO |
| 10 | Integration, robustness, error analysis, go/no-go | TODO |

## What the ablations actually show (toy rings data, mean +/- std over 5 seeds)
- **Init:** zeros -> no learning (hidden units stay identical); 0.01-scale init sits on a plateau (stuck at 150 epochs in all seeds; only 1 of 3 seeds escaped in 1500); Xavier/He learn immediately.
- **Optimizers:** Momentum/Nesterov reach loss<0.05 in ~9 epochs vs ~63 for SGD and ~142 for batch GD. *Caveat:* batch GD used lr=0.5 vs 0.05 and makes 1 update/epoch vs ~25, so it is not a matched-budget comparison. Momentum and Nesterov are indistinguishable here.
- **Generalisation (40 noisy training points):** baseline held-out 0.765; rotation augmentation 0.830 (best); L2=1e-2 0.770 (no real gain, within noise); L2=5e-2 under-fits (0.516).
- **Normalisation:** with one feature scaled x50, standardising lifts held-out accuracy 0.578 -> 0.750.
- **Step activation:** cannot be trained by backprop (gradient is 0 almost everywhere) - enforced by an error.
These are toy-problem results; re-run the same ablations with PyTorch on real images before drawing conclusions for the report.

## Known limitations
- Synthetic data only exercises the plumbing; it says nothing about clinical performance.
- Near-duplicate detection uses a hash filter + pixel check tuned on synthetic data; re-tune on real images.
