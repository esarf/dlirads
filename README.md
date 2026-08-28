# dLI-RADS: an interpretable deep learning method for early hepatocellular carcinoma diagnosis on multiphase CT

Official implementation of **dLI-RADS**, an interpretable deep learning framework for hepatocellular carcinoma (HCC) diagnosis from multiphase CT.

dLI-RADS combines **four deep-learning predictions** and **four handcrafted imaging features** inspired by LI-RADS, and integrates them using logistic regression to estimate the probability of HCC.

## Model overview

<p align="center">
  <img src="assets/Figure 1.jpeg" width="900">
</p>

<p align="center">
  <em>Overview of the dLI-RADS framework.</em>
</p>

---

## Pretrained models

Pretrained weights for the four deep-learning components of dLI-RADS are provided in the `weights/` directory:

```text id="6k3l6r"
weights/
├── hcc.ckpt
├── aphe.ckpt
├── ec.ckpt
└── npw.ckpt
```

The models respectively predict:

* **HCC-DL** — HCC presence
* **APHE-DL** — arterial phase hyperenhancement
* **EC-DL** — enhancing capsule
* **NPW-DL** — non-peripheral washout

These predictions are used as four of the eight intermediate features of dLI-RADS.

---

## Method

For each liver lesion, dLI-RADS extracts eight intermediate features:

* **Deep learning:** HCC-DL, APHE-DL, EC-DL, NPW-DL
* **Handcrafted:** APHE-HF, EC-HF, NPW-HF, lesion size

The final prediction is obtained using:

$$
P(\mathrm{HCC}) =
\sigma\left(
\beta_0 + \sum_{k=1}^{8}\beta_k x_k
\right)
$$

where \(x_k\) are the eight intermediate features.

## Preprocessing

CT volumes are:

1. clipped between **−100 and 400 HU**;
2. spatially registered across arterial, portal venous and delayed phases;
3. resampled to **2.00 × 0.76 × 0.76 mm³**;
4. cropped into lesion-centered patches of **24 × 96 × 96 voxels**.

The three CT phases and lesion segmentation are stacked as model inputs.

Relevant preprocessing scripts:

```text id="enlog3"
preprocessing.py
transfer_numpy.py
generate_patches.py
```

Local paths in these files and in `config.py` must be adapted to the user's dataset.

## Repository structure

```text id="8am9br"
dlirads/
├── assets/
├── models/                         # neural network architectures
├── weights/                        # pretrained models
│   ├── aphe.ckpt
│   ├── ec.ckpt
│   ├── hcc.ckpt
│   └── npw.ckpt
│
├── main.py                         # training entry point
├── yAwareContrastiveLearning.py    # PyTorch Lightning training module
├── config.py                       # configuration
├── dataset.py
├── dataset3d.py                    # 3-D lesion dataset
├── augmentations.py
├── losses.py
├── sampler.py
├── inference.py
├── preprocessing.py
├── transfer_numpy.py
├── generate_patches.py
├── misalignment.py
└── dino.py
```

## Training

Training is launched through `main.py`. Arguments are defined in `config.py` and can be provided from the command line following:

```bash id="mr2qer"
python main.py \
    --mode finetuning \
    --label_name <target> \
    --lr <learning_rate> \
    --weight_decay <weight_decay> \
    --batch_size <batch_size> \
    --max_epochs <epochs>
```

The four classifiers correspond to the following targets:

```text id="u5cs42"
HCC                    has_hcc
APHE                   aphe
Enhancing capsule      ec
Non-peripheral washout npw
```

For example:

```bash id="lt0b6a"
python main.py \
    --mode finetuning \
    --label_name has_hcc \
    --lr 1e-5 \
    --weight_decay 1e-3 \
    --batch_size 32 \
    --max_epochs 300
```

The experiments reported in the paper used:

```text id="iqvitd"
Optimizer       AdamW
Learning rate   1e-5
Weight decay    1e-3
Batch size      32
Epochs          300
```

The current implementation was developed on the Jean Zay / IDRIS computing environment. Paths and SLURM-specific settings may therefore need to be adapted for local execution.

## Operating points

The logistic regression produces a continuous HCC probability.

To define the **dLR-5** and **dLR-4,5** operating points, thresholds are selected on the training set to match the sensitivity and specificity of the corresponding LI-RADS operating points.

$$
t_c^* =
\arg\min_t
\left[
|SE(t)-SE_c| + |SP(t)-SP_c|
\right]
$$

where \(c\) denotes either the LI-RADS 5 or LI-RADS 4/5 operating point.

The resulting thresholds define **dLR-5** and **dLR-4,5**, respectively.

## Results

### Diagnostic performance

dLI-RADS achieved an AUC of **0.76** on the internal evaluation set and **0.84** on the independent external evaluation set.

<p align="center">
  <img src="assets/Figure%204a.jpg" width="49%">
  <img src="assets/Figure%204b.jpg" width="49%">
</p>

<p align="center">
  <em>ROC curves on the internal (left) and external (right) evaluation sets.</em>
</p>

At the dLR-5 operating point, dLI-RADS achieved an accuracy of **80%** on the internal evaluation set, compared with **74%** for LI-RADS. On the external evaluation set, dLI-RADS reached **93% sensitivity** and **86% accuracy**.

### Interpretable predictions

The intermediate dLI-RADS features provide a lesion-level representation of the imaging characteristics driving each prediction.

<p align="center">
  <img src="assets/Figure%203.jpeg" width="850">
</p>

<p align="center">
  <em>Examples of dLI-RADS predictions and their intermediate imaging features.</em>
</p>

The radar plots allow the contribution of deep-learning and handcrafted LI-RADS-related features to be inspected for individual lesions, and highlight cases where dLI-RADS and radiological LI-RADS assessments agree or differ.

### Complementarity with LI-RADS

dLI-RADS and LI-RADS capture partly complementary information. Combining their risk categories produced a maximal-risk group with **96% specificity at 69% sensitivity**.

<p align="center">
  <img src="assets/Figure%205.jpg" width="600">
</p>

<p align="center">
  <em>Diagnostic operating points obtained by combining dLI-RADS and LI-RADS risk categories.</em>
</p>

---

## Citation

If you use this code, please cite:

```bibtex id="920qp8"
@article{sarfati_dlirads,
  title   = {dLI-RADS: an interpretable deep learning method for early hepatocellular carcinoma diagnosis on multiphase CT},
  author  = {Sarfati, Emma and Bône, Alexandre and Gori, Pietro and Yang, Sisi and Rohé, Marc-Michel and Nicolas, François and Lee, Jeong-Min and Yoon, Jeong Hee and Ronot, Maxime and Bloch, Isabelle and Aubé, Christophe},
  journal = {},
  year    = {2026}
}
```

Final publication information and DOI will be added upon publication.

## License

This repository is distributed under the **CC BY-NC-SA 4.0** license.
