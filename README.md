# VAK-Former: Swin-L based Mask2Former for Maritime Semantic Segmentation

<div align="center">

<img src="resources/mmseg-logo.png" width="500"/>

**VAK-Former** is a research-oriented semantic segmentation framework built on top of **MMSegmentation**, designed specifically for **Unmanned Surface Vehicle (USV) perception in maritime environments**.  
The project focuses on improving segmentation accuracy on the **LARS Maritime Dataset** using a **Mask2Former architecture with a Swin-L Transformer backbone**.

</div>

---

## Motivation and Defence Relevance

Autonomous and semi-autonomous **Unmanned Surface Vehicles (USVs)** are increasingly important for maritime surveillance, coastal monitoring, port security, and defence applications. Organizations such as **DRDO, BEL, and the Indian Navy** actively explore perception systems that enable reliable navigation in complex marine environments.

Maritime scenes are uniquely challenging due to dynamic water surfaces, strong reflections, low contrast between objects and background, small and distant obstacles, and large illumination variations caused by weather and time of day. For defence-grade USVs, perception failures can directly translate into navigation errors, collision risks, or mission failure.

Accurate **semantic segmentation** enables pixel-level understanding of the environment, allowing USVs to reliably identify navigable water regions, detect obstacles and vessels, and support downstream modules such as collision avoidance, path planning, and autonomous decision-making. This project is motivated by the need for **high-accuracy, robust segmentation models** rather than purely real-time systems.

---

## Why Mask2Former with Swin-L Backbone

Mask2Former formulates semantic segmentation as a **mask classification problem**, unifying semantic, instance, and panoptic segmentation under a single framework. This formulation is particularly effective in cluttered maritime scenes where object boundaries are ambiguous and water regions dominate the image.

The **Swin-L (Large) Transformer backbone** was selected to maximize segmentation accuracy by capturing long-range spatial dependencies and global contextual information. Unlike traditional CNN backbones, Swin-L enables the model to better distinguish between water surfaces, distant vessels, and background structures. Although computationally expensive, Swin-L significantly improves boundary quality and small-object segmentation, which are critical for safety-critical defence applications.

---

## Dataset: LARS Maritime Dataset

The **LARS Maritime Dataset** contains real-world maritime images captured from USV perspectives, with pixel-level annotations for water, obstacles, vessels, and background classes. The dataset reflects realistic operational conditions such as reflections, waves, occlusions, and varying lighting, making it well-suited for evaluating defence-oriented perception systems.

---

## Experimental Setup

- **Framework:** MMSegmentation (OpenMMLab)
- **Model:** Mask2Former
- **Backbone:** Swin-L Transformer
- **Task:** Semantic Segmentation
- **Evaluation Metrics:** Mean Intersection over Union (mIoU), F1 Score, Frames Per Second (FPS)

Training the Swin-L based Mask2Former model is computationally intensive and requires careful dependency management and long training times. However, this cost is justified by the significant performance gains in challenging maritime scenarios.

---

## Key Results

The proposed approach achieves improved segmentation performance compared to baseline CNN-based methods, particularly in terms of mIoU and boundary accuracy. The model demonstrates superior handling of small and distant objects, smoother water–obstacle boundaries, and increased robustness to environmental variations. These results support the suitability of transformer-based segmentation models for real-world USV deployment.

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

Refer to the official MMSegmentation documentation for detailed installation instructions.

---

## Dataset Preparation

Prepare the LARS dataset in MMSegmentation format:

```
data/
 └── LARS/
     ├── images/
     │   ├── train/
     │   ├── val/
     └── annotations/
         ├── train/
         ├── val/
```

Update dataset paths in the configuration files before training.

---

## Training

```bash
python tools/train.py configs/mask2former/mask2former_swin-l_lars.py
```

---

## Evaluation

```bash
python tools/test.py configs/mask2former/mask2former_swin-l_lars.py \
    work_dirs/mask2former_swin-l/latest.pth --eval mIoU
```

---

## Project Contributions

- Adaptation of Mask2Former for maritime semantic segmentation
- Integration of Swin-L backbone for enhanced global context modeling
- Custom training and evaluation pipeline for the LARS dataset
- Extensive experimentation under limited computational resources

---

## Notes

- Training is computationally expensive; multi-GPU systems are recommended
- The focus of this project is segmentation accuracy rather than real-time inference
- Intended for research and academic use

---

## Citation

```bibtex
@misc{vakformer2025,
  title={VAK-Former: Swin-L based Mask2Former for Maritime Semantic Segmentation},
  author={Khagendra Saini, Anirudh Phophalia, Vaani Mehta},
  year={2025},
  howpublished={\url{https://github.com/VAK-Former/VAK-Former}}
}
```

---

## Acknowledgements

This project builds upon **OpenMMLab MMSegmentation**, **Mask2Former**, and **Swin Transformer**. We acknowledge the open-source community for enabling advanced research in semantic segmentation.

---

## License

This project follows the **Apache 2.0 License**, consistent with MMSegmentation.
