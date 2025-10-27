# LMU-Net
  This is our research code of "LMU-Net: An Efficient and Generalizable Architecture for Medical Image Segmentation"
  
  If you need any help for the code and data, do not hesitate to leave issues in this repository.

## Method
### Training
```

!python train.py \
--dataset-name 'TSRS_RSNA-Epiphysis' \
--backbone 'mobilenet_v2' \
--epochs 550 \
--batch-size 16 \
--lr 1e-3 \
--weight-decay 0.0001 \
--patience 40 \
--fine-tune-epochs 60 \
--deep-supervision-weights 0.2 0.4 0.6 0.8 1.0 \
--focal-loss-weight 0.8 \
--dice-loss-weight 1.2 \
--scheduler-type 'CosineAnnealingWarmRestarts' \
--scheduler-T0 15 \
--num-workers 4

```

### Testing

```

!python train_light.py \
--dataset-name 'TSRS_RSNA-Epiphysis' \
--backbone 'mobilenet_v2' \
--test-only

!python benchmark_fps.py \
--backbone 'mobilenet_v2' \
--input-h 224 \
--input-w 224

```
