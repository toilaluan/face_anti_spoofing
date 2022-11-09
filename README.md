# Face Anti Spoofing - Zalo Ai Challenge
1. Get data from Zalo:
- Train data: https://dl-challenge.zalo.ai/liveness-detection/train.zip

- Public test: https://dl-challenge.zalo.ai/liveness-detection/public_test.zip

- Last update: 11:35 AM, 04/11/2022
then extract data follow this folder structure:\
```
.
|-- dataset
|  |-- train
|     |-- videos
|     |-- label.csv
|  |-- public
|     |-- videos
```
2. Extract all frames from videos:
** extract_frame.sh ** only extract frames from train/videos. Please change folder path in this file to extract frames from public/videos (public test).\
 `bash extract_frame.sh`

