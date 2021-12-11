#!/bin/bash

wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18tdoPVvvCOYvQsUGV0SzHL_yVS3yRCfp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18tdoPVvvCOYvQsUGV0SzHL_yVS3yRCfp" -O ./TTS/my_model.pth && rm -rf /tmp/cookies.txt

python3 ./TTS/test.py -c ./TTS/hw_tts/config_test.json -r ./TTS/my_model.pth