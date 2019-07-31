#!/usr/bin/env bash

rm global_min_acer_model.pth
scp loki@ljiljana:/home/loki/Projects/rn/CVPR19-Face-Anti-spoofing/models/model_A_color_48/checkpoint/global_min_acer_model.pth \
~/Projects/poso/02-CVPR19-Face-Anti-spoofing/models/model_A_color_48/checkpoint/
