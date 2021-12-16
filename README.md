# NV
Neural Vocoder HW for DLA HSE course

This `overfit branch` contains version, which successfully overfits on batch to show correctness of Generator.

---
To start overfitting in Colab run
```
!git clone -b overfit https://github.com/ArturGoldman/NV
!chmod u+x ./NV/prep.sh
!./NV/prep.sh
!python ./NV/train.py -c ./NV/hw_nv/config.json
```
