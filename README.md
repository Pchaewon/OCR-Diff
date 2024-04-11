# OCR-Diff
A Two-Stage Deep Learning Framework for Optical Character Recognition Using Diffusion Model in Industrial Internet-of-Things <br/>

This code repository has been temporarily made private due to an ongoing review of intellectual property matters. Our team is working diligently to conclude this process swiftly, and we plan to make the code publicly available again once this review is completed. We appreciate your understanding and patience in this matter.


## Dataset Preparation

The dataset should be organized inside the `dataset` folder of the project. The dataset used in this paper can be downloaded from the following link:

- TextZoom Dataset: [https://github.com/WenjiaWang0312/TextZoom](https://github.com/WenjiaWang0312/TextZoom)

After downloading, unzip and arrange the dataset within the `dataset` folder as required.

## Pretrained Models

Pretrained models for testing can be downloaded from the following links:

- ASTER: [https://github.com/ayumiymk/aster.pytorch](https://github.com/ayumiymk/aster.pytorch)
- MORAN: [https://github.com/Canjie-Luo/MORAN_v2](https://github.com/Canjie-Luo/MORAN_v2)
- CRNN: [https://github.com/meijieru/crnn.pytorch](https://github.com/meijieru/crnn.pytorch)
- CDistNet: [https://github.com/simplify23/CDistNet](https://github.com/simplify23/CDistNet)

After downloading, save each model in the specified folder and configure the paths according to the project settings.

## Installation

[Add instructions on how to install and set up your project here. This might include steps to install necessary libraries or packages.]

## Training Instructions

To train the project, use the following command:

```bash
CUDA_VISIBLE_DEVICES=GPU_NUMS python main.py --exp_name YOUR_EXP_NAME
```

## Testing Instructions

To Test the project, use the following command:

```bash
CUDA_VISIBLE_DEVICES=GPU_NUMS python main.py --exp_name YOUR_EXP_NAME --resume CHECKPOINT_PATH --test --test_data_dir DATA_PATH
```


