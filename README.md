# VAK-Former: Swin-L based Mask2Former for Maritime Semantic Segmentation

<div align="center">

<img src="resources/mmseg-logo.png" width="500"/>

**VAK-Former** is a research-oriented semantic segmentation framework built on top of **MMSegmentation**, designed for **autonomous inspection of Unmanned Surface Vessels (USVs) in maritime environments**.  
The project implements a **Mask2Former-style transformer architecture with a Swin-Large backbone**, evaluated on the **LaRS (Lakes, Rivers and Seas) Maritime Dataset**.

</div>

> **Note:** This repository contains code directly associated with our manuscript submitted to *The Visual Computer*,  
> **“A Transformer-Based Semantic Segmentation Framework for Autonomous Inspection of Unmanned Surface Vessels.”**  
> If you use this code or results, please cite the corresponding paper (see Citation section).

---

## Code and Data Availability

To enhance transparency and reproducibility, we provide permanent links to the code and related artifacts:

- **GitHub (source code):** https://github.com/VAK-Former/VAK-Former  
- **Archived code (Zenodo):** https://doi.org/10.5281/zenodo.xxxxxxx  <!-- replace with actual DOI -->
- **LaRS dataset:** Publicly available from the original authors; please follow their access and license terms.

This code is **directly related** to the manuscript currently submitted to *The Visual Computer*.  
Readers are encouraged to use these links to **replicate the experiments** and **evaluate the reported results**, and to **cite the manuscript** when using this repository.

After acceptance, the manuscript’s final citation (volume/issue/pages/DOI) will be added here.

---

## Motivation and Application Context

Maritime environments such as lakes, rivers, and coastal seas are visually complex due to waves, reflections, low contrast, small obstacles, and rapidly changing illumination. Reliable perception in these conditions is essential for:

- Safe navigation and collision avoidance by USVs/USCVs.  
- Environmental monitoring and floating-waste removal.  
- Defence and surveillance applications where perception failures can lead to mission risk.

Classical morphology-based segmentation methods rely on shallow features and are sensitive to illumination changes, making them unsuitable for complex maritime scenes. Modern deep-learning-based semantic segmentation provides **pixel-level understanding of water surfaces, vessels, and obstacles**, enabling robust downstream tasks such as tracking, planning, and decision-making.

VAK-Former focuses on **high-quality, robust segmentation** rather than purely maximizing FPS, aiming at realistic deployment on USV platforms where accuracy and boundary precision are critical.

---

## Method: Mask2Former-Style Transformer with Swin-L

VAK-Former follows a **Mask2Former-style encoder–decoder paradigm** that formulates semantic segmentation as a **mask classification (set prediction) problem** instead of dense per-pixel classification.

### Architecture Overview

The framework has three major components:

1. **Hierarchical Vision Transformer Backbone (Swin-Large)**  
   - Extracts multi-scale feature maps from the input LaRS maritime image using **shifted window attention**.  
   - Captures long-range contextual dependencies while remaining computationally efficient.  
   - Preserves both low-level spatial details (edges, boundaries) and high-level semantic context (objects, background structures).

2. **Multi-Scale Pixel Decoder with Deformable Attention**  
   - Aggregates multi-resolution backbone features using a feature-pyramid-like structure.  
   - Employs deformable attention to better align features across scales and regions.  
   - Produces a high-resolution **mask feature map** shared by all queries, improving boundary awareness and semantic consistency across scales and object sizes.

3. **Transformer Decoder with Mask Classification Head**  
   - Operates on a fixed set of learnable queries, each representing a potential semantic region.  
   - Stacked self-attention models interactions between queries, while cross-attention links queries to pixel features from the pixel decoder.  
   - Each query outputs a **class label** and a **segmentation mask**, reframing segmentation as a **set prediction** problem without heuristic post-processing (e.g., NMS).  
   - Final semantic maps are obtained by combining predicted masks weighted by their corresponding class scores.

This design leverages **global context (transformer)**, **multi-scale fusion (pixel decoder)**, and **query-based mask prediction (Mask2Former)** to yield accurate, spatially precise segmentation in challenging maritime conditions.

---

## Dataset: LaRS Maritime Dataset

Experiments are conducted on the **LaRS (Lakes, Rivers and Seas) dataset**, a diverse panoptic maritime obstacle detection benchmark built from real USV recordings.

Key characteristics:

- **USV-mounted viewpoint**, matching deployment scenarios.  
- Thousands of per-pixel labeled images with panoptic annotations.  
- Varied locations, scene types, obstacle classes, and acquisition conditions.  
- Challenging factors: strong reflections, waves, low contrast, haze, small/distant obstacles.

We follow the dataset splits and evaluation protocol described by the LaRS authors.  
Please refer to the LaRS project page for download instructions and licensing.

---

## Experimental Setup

- **Framework:** MMSegmentation (OpenMMLab, v1.x)  
- **Model:** Mask2Former-style segmentation model  
- **Backbone:** Swin-Large (hierarchical vision transformer)  
- **Task:** Semantic segmentation for autonomous USV inspection  
- **Dataset:** LaRS (training/validation split as in the paper)  
- **Metrics:** F1-score, Frames Per Second (FPS); mean Accuracy (mAcc) and mean Intersection over Union (mIoU) are monitored during training.

### Environment and Dependencies

We recommend the following environment (as used in the manuscript):

- Python 3.8  
- PyTorch 2.1.2, TorchVision 0.16.2  
- CUDA-capable GPU with ≥ 16 GB VRAM (for Swin-L)  
- MMSegmentation v1.x  
- MMEngine, MMCV 2.1.0, MMDetection

Install dependencies:

```bash
conda create -n vakformer python=3.8 -y
conda activate vakformer

pip install torch==2.1.2 torchvision==0.16.2 numpy==1.26.4
pip install openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
mim install mmsegmentation
```

All core components (Swin-L backbone, multi-scale pixel decoder with deformable attention, Mask2Former-style transformer decoder) are implemented following MMSegmentation’s modular structure (see `configs/` and `mmseg/models/`).

---

## Dataset Preparation

Prepare LaRS in MMSegmentation format:

```text
data/
 └── LaRS/
     ├── images/
     │   ├── train/
     │   ├── val/
     └── annotations/
         ├── train/
         ├── val/
```

- Place training and validation images in the corresponding `images` subdirectories.  
- Place label masks (semantic segmentation) in the corresponding `annotations` subdirectories.  
- Update dataset paths and class definitions in your configuration file, for example:
  - `configs/mask2former/mask2former_swin-l_lars.py`

---

## Training

Launch training with:

```bash
python tools/train.py configs/mask2former/mask2former_swin-l_lars.py
```

Important configuration aspects:

- **Backbone:** Swin-Large with ImageNet pretraining.  
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum.  
- **LR schedule:** Polynomial decay for stable convergence.  
- **Loss:** Weighted combination of classification and mask losses (Mask2Former formulation).  
- **Augmentations:** Random horizontal flip, scaling, color jittering.

Training is performed until convergence (stabilization of loss and validation metrics).  
In our experiments, training was conducted on a single GPU; inference speed (FPS) was measured on the same hardware.

---

## Evaluation

Evaluate a trained checkpoint on the validation set:

```bash
python tools/test.py \
  configs/mask2former/mask2former_swin-l_lars.py \
  work_dirs/mask2former_swin-l/latest.pth \
  --eval mIoU
```

For full reproducibility of the paper’s results, we recommend:

- Computing F1-score, mAcc, and mIoU from the predictions.  
- Measuring FPS by averaging inference time across multiple consecutive frames under the same hardware conditions.

---

## Quantitative Results on LaRS

The table below summarizes the quantitative comparison reported in the manuscript, using F1-score (segmentation accuracy) and FPS (runtime efficiency) on the LaRS dataset.

| S.No. | Model                | Backbone   | FPS   | F1-score (%) |
|-------|----------------------|-----------:|------:|-------------:|
| 1     | Mask2Former          | Swin-B     | 4.80  | 71.10        |
| 2     | Panoptic FPN         | ResNet-50  | 21.70 | 58.90        |
| 3     | Mask2Former          | Swin-T     | 5.40  | 56.70        |
| 4     | Panoptic FPN         | ResNet-101 | 16.70 | 58.10        |
| 5     | Mask2Former          | ResNet-50  | 10.60 | 54.90        |
| 6     | Mask2Former          | ResNet-101 | 5.70  | 53.20        |
| 7     | Panoptic-DeepLab     | ResNet-50  | 6.00  | 64.60        |
| 8     | MaX-DeepLab          | MaX-S      | 3.70  | 60.20        |
| 9     | **VAK-Former (ours)** | **Swin-L** | **12.33** | **97.71** |

Key observations:

- **Highest F1-score (97.71%)** among evaluated models, indicating very robust pixel-level classification in challenging maritime scenes.  
- **12.33 FPS**, substantially faster than many transformer-based baselines while retaining a powerful Swin-L backbone.  
- A favorable balance between **accuracy** and **runtime**, suitable for near real-time USV inspection.

---

## Training Dynamics and Qualitative Results

During training on LaRS:

- **mAcc** and **mIoU** increase steadily and converge smoothly to high values (≈ 97–98% mAcc and ≈ 94–95% mIoU), suggesting stable optimization and effective learning of multi-scale semantics.  
- The training loss decreases monotonically and stabilizes after sufficient iterations, with no large oscillations.

Qualitative examples (see manuscript figures) show:

- Accurate demarcation of water vs. non-water regions.  
- Precise boundaries around small and partially occluded obstacles.  
- Robustness to reflections, low contrast, haze, and cluttered backgrounds.

These qualitative results are consistent with the quantitative metrics, demonstrating the model’s suitability for **autonomous USV inspection and navigation**.

---

## Project Contributions

- **Transformer-based segmentation for USVs:**  
  A Mask2Former-inspired transformer architecture tailored to autonomous USV inspection in complex marine environments.

- **Swin-L backbone for global context:**  
  A hierarchical vision transformer backbone that captures long-range dependencies while preserving fine spatial detail.

- **Multi-scale pixel decoder with deformable attention:**  
  Enhanced fusion of multi-resolution features, improving boundary precision and semantic consistency.

- **Query-based transformer decoder for mask classification:**  
  Segmentation formulated as a set prediction problem with learnable queries, eliminating heuristic post-processing.

- **LaRS-specific training and evaluation pipeline:**  
  Reproducible configuration, data preprocessing, and evaluation protocol for LaRS, with detailed comparisons to state-of-the-art architectures.

- **Strong accuracy–speed trade-off:**  
  Achieves **97.71% F1** at **12.33 FPS**, demonstrating that transformer-based designs can be both accurate and practical for near real-time USV deployment.

---

## Notes

- Training with a Swin-L backbone is computationally demanding; **high-memory single-GPU or multi-GPU setups** are recommended.  
- This implementation focuses on **robust segmentation quality** and **practical runtime**, rather than strict embedded real-time constraints.  
- For deployment on constrained onboard hardware, further pruning, quantization, or backbone downsizing (e.g., Swin-T/Swin-B) may be required.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{vakformer2025,
  title        = {VAK-Former: Swin-L based Mask2Former for Maritime Semantic Segmentation},
  author       = {Khagendra Saini and Anirudh Phophalia and Vaani Mehta},
  year         = {2025},
  howpublished = {\url{https://github.com/VAK-Former/VAK-Former}}
}
```

(After acceptance in *The Visual Computer*, please update this entry with the journal’s official citation information.)

---

## License

This repository is released for **non-commercial research and academic use**.  
Please refer to the `LICENSE` file for full terms and conditions.
