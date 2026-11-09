## SDKD on COCO

This repository provides the object detection and instance segmentation implementation of **Structured Difficulty Knowledge Distillation (SDKD)** on MS COCO 2017. SDKD adaptively partitions non-target classes into hard and easy groups and transfers their knowledge separately. It also preserves the relative difficulty order across samples using standardized target class scores. For both tasks, SDKD is applied to the classification logits of the same RoIs sampled by the student's RoI head.

### Usage

The code is built on Detectron2, KD++, ReviewKD, and DKD.

### 0. Environments
The experiments use **PyTorch 1.10.0+cu111**, **torchvision 0.11.0+cu111**, and **Detectron2 0.6+cu111**. Please follow the [Detectron2 installation instructions](https://github.com/facebookresearch/detectron2) to prepare the environment. Run the commands below from the project root.

Place [MS COCO 2017](https://cocodataset.org/#download) under **datasets/coco/**. The image directories should be **datasets/coco/train2017/** and **datasets/coco/val2017/**. Place **instances_train2017.json** and **instances_val2017.json** under **datasets/coco/annotations/**.

Place the combined teacher and student checkpoints from the [ReviewKD releases](https://github.com/dvlab-research/ReviewKD/releases/) under **pretrained/**. The detection configurations use **r18-r101.pth**, **r50-r101.pth**, and **mv2-r50.pth**. The instance segmentation configurations use **r18-r101mask.pth**, **r50-r101mask.pth**, and **mv2-r50mask.pth**. If a checkpoint is stored elsewhere, update **MODEL.WEIGHTS** in the corresponding configuration.

### 1. Train student by SDKD for object detection

```bash
# Teacher: ResNet101, Student: ResNet18 (Faster R-CNN + FPN)
python3 train_net.py \
        --config-file configs/SDKD/SDKD-R18-R101.yaml --num-gpus 4

# Teacher: ResNet101, Student: ResNet50 (Faster R-CNN + FPN)
python3 train_net.py \
        --config-file configs/SDKD/SDKD-R50-R101.yaml --num-gpus 4

# Teacher: ResNet50, Student: MobileNetV2 (Faster R-CNN + FPN)
python3 train_net.py \
        --config-file configs/SDKD/SDKD-MV2-R50.yaml --num-gpus 4
```

### 2. Train student by SDKD for instance segmentation

```bash
# Teacher: ResNet101, Student: ResNet18 (Mask R-CNN + FPN)
python3 train_net.py \
        --config-file configs/SDKD/SDKD-R18-R101-Mask.yaml --num-gpus 4

# Teacher: ResNet101, Student: ResNet50 (Mask R-CNN + FPN)
python3 train_net.py \
        --config-file configs/SDKD/SDKD-R50-R101-Mask.yaml --num-gpus 4

# Teacher: ResNet50, Student: MobileNetV2 (Mask R-CNN + FPN)
python3 train_net.py \
        --config-file configs/SDKD/SDKD-MV2-R50-Mask.yaml --num-gpus 4
```

### 3. Train student by SDKD with ReviewKD

SDKD can be combined with the feature distillation loss of ReviewKD. This combination is named **ReviewSDKD** in the code. The examples below use ResNet101 as the teacher and ResNet50 as the student; configurations for the other two model pairs are also available under **configs/SDKD/**.

```bash
# Object detection (Faster R-CNN + FPN)
python3 train_net.py \
        --config-file configs/SDKD/ReviewSDKD-R50-R101.yaml --num-gpus 4

# Instance segmentation (Mask R-CNN + FPN)
python3 train_net.py \
        --config-file configs/SDKD/ReviewSDKD-R50-R101-Mask.yaml --num-gpus 4
```

# Acknowledgement

We thank the contributors of Detectron2, KD++, ReviewKD, and DKD for sharing their code and pretrained models.
