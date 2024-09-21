# Hanwha System AI Challenge - Infrared Sensor Dataset Detection

This project modifies the original [DETR](https://github.com/facebookresearch/detr) (DEtection TRansformers) to better handle the Infrared Sensor Dataset from the Hanwha System AI Challenge.

## Dataset Preparation

Download the dataset from [here](https://www.hscaichallenge.com/datasets) and organize it as follows:

/data │ ├── test_images/ # Test image files ├── train_images/ # Training image files ├── val_images/ # Validation image files ├── train_annotations.json # Training annotations └── val_annotations.json # Validation annotations

## Installation

To set up the environment for the Modified DETR project, follow these steps:

Step 1: Create a new Conda environment
conda create -n modified_detr python=3.8

Step 2: Activate the environment
conda activate modified_detr

Step 3: Install the required dependencies
pip install -r requirements.txt



## Running the Code
1. Using Pretrained DETR
To start training from the original DETR pretrained weights, run:
python main.py --coco_path data \
               --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth \
               --want_class 8 \
               --output_dir finetuning_detr4
   
2. Resuming from a Checkpoint
To continue training from a previous checkpoint:
python main.py --coco_path data \
               --resume finetuning_detr4/checkpoint.pth \
               --want_class 9 \
               --output_dir finetuning_detr4 \
               --batch_size 6

   
## Model Modifications
Backbone: Added FPN (Feature Pyramid Network) and ASFF (Adaptive Spatial Feature Fusion). The performance with a batch size of 1 seems promising.
fourier는 효과가 없는거 같아서 뺏음

todo
1) 이대로 학습돌려보기
2) detr.py변경하고

학습돌린결과
Accuracy : 39.87
