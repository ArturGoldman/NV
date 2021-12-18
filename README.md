# NV
Neural Vocoder HW for DLA HSE course

Implementing [HiFiGAN](https://arxiv.org/pdf/2010.05646.pdf)

## Running guide

All commands are written as if they are executed in Google Colab

To set up environment run
```
!git clone https://github.com/ArturGoldman/NV
!chmod u+x ./NV/prep.sh
!./NV/prep.sh
```

To start testing run
```
!chmod u+x ./NV/test.sh
! ./NV/test.sh
```

By default testing audios are specified in `hw_nv/audio_to_test`. If you want to pass you own audios,
place it in the same directory and provide path to it in `config_test.json` under 'file_dir' field. Then, execute commands above.

If you want to start training process from the start run
```
!python3 ./NV/train.py -c ./NV/hw_tts/config.json
```
Note that after training you will have to pass trained model to test on your own. See `test.sh`. Training was performed using `config.json`.

## Results
Model works great!

Details of training can be found here: https://wandb.ai/artgoldman/NV-HSE-DLA

Audios sound just as in ground truth.

## Credits
Structure of this repository is based on [template repository of first ASR homework](https://github.com/WrathOfGrapes/asr_project_template),
which itself is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.
