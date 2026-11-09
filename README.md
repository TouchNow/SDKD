# SDKD: Structured Difficulty Knowledge Distillation

## Abstract

SDKD transfers class difficulty within each sample and relative recognition difficulty across samples. The intra-sample module adaptively partitions non-target classes into hard and easy groups and distills them separately. The inter-sample module uses standardized teacher target-class scores to construct pairwise constraints that preserve the relative difficulty order across samples. SDKD uses only logits and requires no intermediate feature alignment or changes to the student architecture at inference.

## Training on CIFAR-100

Set **_get_data_root()** in **dataset/cifar100.py** to your dataset directory. Provide a pretrained teacher checkpoint through **--path-t**, with its state dictionary stored under the **model** key. The following command trains a ResNet8×4 student with a ResNet32×4 teacher.

```bash
python train_student.py \
    --dataset cifar100 --distill sdkd \
    --path-t ./save/models/resnet32x4/vanilla/resnet32x4_best.pth \
    --model_s resnet8x4 --epochs 240 \
    --batch_size 64 --learning_rate 0.05 \
    --gamma 1 --gap_head 1 --gap_tail 6 --beta 1.0 --kd_T 4 \
    --have_mlp 0 --gpu_id 0 --seed 0 --save_model \
    --experiments_dir sdkd --experiments_name resnet32x4_resnet8x4
```

## Acknowledgements

This implementation is based on [CTKD](https://github.com/zhengli97/CTKD), the code for **Curriculum Temperature for Knowledge Distillation**. We thank the CTKD authors for making their code publicly available.
