# VAK-Former: Swin-L based Mask2Former for Maritime Semantic Segmentation

<div align="center">

<img src="resources/mmseg-logo.png" width="500"/>

  
**VAK-Former** is a research-oriented semantic segmentation framework built on top of **MMSegmentation**, designed for **autonomous inspection of Unmanned Surface Vessels (USVs) in maritime environments**.  
The project implements a **Mask2Former-style transformer architecture with a Swin-Large backbone**, evaluated on the **LaRS (Lakes, Rivers and Seas) Maritime Dataset**.


</div>

---

## Motivation and Application Context

Maritime environments such as lakes, rivers, and coastal seas are critical for transportation, trade, environmental monitoring, and defence, but they are visually complex due to waves, reflections, low contrast, small obstacles, and changing illumination.  
Autonomous and semi-autonomous **USVs/USCVs (Unmanned Surface Cleaning Vessels)** depend on robust perception to safely navigate, avoid collisions, and perform tasks like inspection and floating-waste removal.

Traditional vision pipelines struggle with fine-grained obstacle boundaries and challenging imaging conditions, especially when relying on shallow features or classic morphology-based segmentation.  
By contrast, modern deep-learning-based semantic segmentation enables **pixel-level understanding of water surfaces, vessels, and obstacles**, which is essential for real-time decision-making and safe USV operation.

---

## Method: Mask2Former-Style Transformer with Swin-L

VAK-Former adopts a **Mask2Former-style encoder–decoder framework** that reformulates semantic segmentation as a **mask classification (set prediction) problem** instead of dense per-pixel classification.  
This design unifies semantic, instance, and panoptic segmentation and is particularly effective in cluttered maritime scenes dominated by water with multiple small and distant obstacles.

The architecture consists of three main components:

1. **Hierarchical Vision Transformer Backbone (Swin-Large)**  
   - Extracts multi-scale feature maps from the input LaRS image using **shifted window attention**.  
   - Captures long-range contextual dependencies while preserving spatial detail, improving separation of water regions, vessels, and background structures.

2. **Multi-Scale Pixel Decoder with Deformable Attention**  
   - Aggregates multi-resolution features in a feature-pyramid-like fashion.  
   - Uses deformable attention to align and fuse features across scales, enhancing **boundary awareness** and **semantic consistency**.  
   - Produces a high-resolution shared mask feature map that supports accurate segmentation of both large water regions and small obstacles.

3. **Transformer Decoder with Mask Classification Head**  
   - Operates on a fixed set of learnable queries, each representing a potential semantic region in the scene.  
   - Uses stacked self-attention to model interactions between regions and cross-attention to link queries with pixel features from the pixel decoder.  
   - Each query jointly predicts a **semantic class label** and an associated **segmentation mask**, eliminating heuristic post-processing (e.g., NMS) and framing segmentation as a set prediction task.

This combination of **global context modeling (transformer), multi-scale fusion (pixel decoder), and query-based mask prediction (Mask2Former)** yields high-quality, spatially precise segmentation results in challenging maritime scenarios.

---

## Dataset: LaRS Maritime Dataset

The **LaRS (Lakes, Rivers and Seas) Maritime Dataset** is a diverse benchmark for marine obstacle detection and semantic segmentation.  
It contains thousands of per-pixel annotated images captured from real USVs across varied locations, obstacle types, and environmental conditions.

Key characteristics:

- **USV-mounted viewpoint**, matching realistic deployment scenarios.  
- Per-pixel labels for water, obstacles/vessels, and background categories.  
- Challenging conditions: strong reflections, waves, low contrast, fog/haze, small and distant objects.  

VAK-Former uses LaRS for both training and evaluation, enabling a direct comparison with existing CNN- and transformer-based segmentation baselines.

---

## Experimental Setup

- **Framework:** MMSegmentation (OpenMMLab, v1.x)  
- **Model:** Mask2Former-style transformer segmentation  
- **Backbone:** Swin-Large (hierarchical vision transformer)  
- **Task:** Semantic segmentation for autonomous USV inspection  
- **Dataset:** LaRS maritime dataset (train/val split as in the paper)  
- **Metrics:** F1-score, Frames Per Second (FPS); mIoU and mean Accuracy (mAcc) are also monitored during training

### Training Protocol (High-Level)

- Supervised training on LaRS with images resized to a fixed resolution and normalized.  
- Standard augmentations: random horizontal flip, scaling, and color jitter to improve robustness to environmental variability.  
- Swin-L backbone initialized from ImageNet pretraining; pixel decoder and transformer decoder trained from scratch.  
- Optimization with stochastic gradient descent (SGD) with momentum and polynomial learning-rate decay.  
- Combined classification and mask losses following the Mask2Former formulation.  
- Training on a single GPU, with inference speed measured on the same hardware to obtain FPS.

These choices aim to balance **high segmentation accuracy** with **practical runtime performance** for deployment on USV platforms.

---

## Key Results on LaRS

VAK-Former significantly improves **segmentation accuracy and runtime** compared to strong CNN and transformer baselines on the LaRS dataset.

- Achieves an **F1-score of 97.71%**, indicating highly accurate pixel-level classification of water, obstacles, and background.  
- Sustains an inference speed of **12.33 FPS**, providing near real-time performance suitable for onboard USV deployment.  
- Demonstrates smoother and more consistent water–obstacle boundaries, fewer misclassifications around small objects, and improved robustness to reflections and low contrast.

This balance between **accuracy (high F1)** and **efficiency (12.33 FPS)** makes the model suitable for real-time or near real-time autonomous inspection and obstacle-aware navigation.

---

## Quantitative Comparison on LaRS

The table below summarizes the quantitative comparison between VAK-Former and representative state-of-the-art segmentation architectures on the LaRS dataset (F1-score and FPS as reported in the paper).

| S.No. | Model              | Backbone   | FPS   | F1-score (%) |
|-------|--------------------|-----------:|------:|-------------:|
| 1     | Mask2Former        | Swin-B     | 4.80  | 71.10        |
| 2     | Panoptic FPN       | ResNet-50  | 21.70 | 58.90        |
| 3     | Mask2Former        | Swin-T     | 5.40  | 56.70        |
| 4     | Panoptic FPN       | ResNet-101 | 16.70 | 58.10        |
| 5     | Mask2Former        | ResNet-50  | 10.60 | 54.90        |
| 6     | Mask2Former        | ResNet-101 | 5.70  | 53.20        |
| 7     | Panoptic-DeepLab   | ResNet-50  | 6.00  | 64.60        |
| 8     | MaX-DeepLab        | MaX-S      | 3.70  | 60.20        |
| 9     | **VAK-Former (ours)** | **Swin-L** | **12.33** | **97.71** |

Compared to these baselines, VAK-Former offers:

- A substantial **F1-score margin** over both CNN-based and prior transformer-based approaches.  
- A **higher FPS** than many transformer baselines, indicating an efficient design despite the large Swin-L backbone.

---

## Training Dynamics and Qualitative Results

During training on LaRS, both **mean Accuracy (mAcc)** and **mIoU** steadily improve and converge smoothly, indicating stable optimization and effective learning of multi-scale semantic information.  
The absence of large oscillations in the loss curve and the synchronized increase of mAcc and mIoU reflect robust training dynamics.

Qualitative results on representative LaRS samples show:

- Accurate segmentation of water versus non-water regions.  
- Clear delineation of small and partially occluded obstacles.  
- Robustness to strong reflections, hazy conditions, and cluttered backgrounds.  

These observations align with the quantitative metrics and highlight the suitability of VAK-Former for **real-world autonomous USV inspection**.

---

## Installation

This repository is built on **MMSegmentation v1.x**.

```bash
conda create -n vakformer python=3.8 -y
conda activate vakformer

pip install torch==2.1.2
pip install torchvision==0.16.2
pip install numpy==1.26.4
pip install openmim
mim install mmengine
mim install "mmcv==2.1.0"
mim install mmdet
```

For environment details, please also refer to the official **MMSegmentation** documentation.

---

## Dataset Preparation

Prepare the LaRS dataset in MMSegmentation format:

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

Update dataset paths and class definitions in the configuration files (for example, `configs/mask2former/mask2former_swin-l_lars.py`) before training.

---

## Training

Launch training with:

```bash
python tools/train.py configs/mask2former/mask2former_swin-l_lars.py
```

You can adjust learning rate, batch size, and training iterations in the config file to match the settings used in the paper or your hardware constraints.

---

## Evaluation

Evaluate a trained checkpoint (e.g., the best F1-score model) on the validation set:

```bash
python tools/test.py configs/mask2former/mask2former_swin-l_lars.py \
    work_dirs/mask2former_swin-l/latest.pth --eval mIoU
```

For a full analysis, you may additionally compute F1-score, mAcc, and FPS as described in the paper.

---

## Project Contributions

- **Transformer-based segmentation for USVs:**  
  Adaptation of a Mask2Former-style transformer architecture for **autonomous USV inspection** in challenging maritime conditions.

- **Swin-L backbone for global context:**  
  Integration of a Swin-Large hierarchical transformer to capture long-range dependencies while preserving fine spatial detail.

- **Multi-scale pixel decoder with deformable attention:**  
  Improved multi-resolution feature fusion, enhancing boundary precision and semantic consistency across scales.

- **Query-based transformer decoder for mask classification:**  
  Reformulation of segmentation as a set prediction problem using learnable queries for joint class and mask prediction.

- **LaRS-specific training and evaluation pipeline:**  
  Tailored data preprocessing, augmentation, and evaluation for the LaRS dataset, including detailed comparison with state-of-the-art baselines.

- **Strong accuracy–speed trade-off:**  
  Achieves **97.71% F1** at **12.33 FPS**, demonstrating that transformer-based architectures can be both accurate and practically deployable on USVs.

---

## Notes

- Training with a Swin-L backbone is computationally demanding; **high-memory single-GPU or multi-GPU systems** are recommended.  
- The current implementation focuses on **high-quality segmentation and robust real-time performance**, rather than strict embedded real-time constraints.  
- Intended primarily for **research and academic use**; additional optimization may be required for deployment on resource-constrained onboard hardware.

---

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{vakformer2026,
  title        = {VAK-Former: Swin-L based Mask2Former for Maritime Semantic Segmentation},
  author       = {Khagendra Saini and Anirudh Phophalia and Vaani Mehta},
  year         = {2026},
  howpublished = {\url{https://github.com/VAK-Former/VAK-Former}}
}
```

---

## Acknowledgements

This project builds upon:

- **OpenMMLab MMSegmentation**  
- **Mask2Former**  
- **Swin Transformer**  

We acknowledge the authors of the LaRS dataset and the broader open-source community for enabling advanced research in maritime semantic segmentation and autonomous USV perception.

---

## License

This repository is released for non-commercial research and academic use.  
Please check the repository’s LICENSE file for full terms and conditions.
