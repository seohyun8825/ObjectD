
python main.py --coco_path data --resume https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth --want_class 8 --output_dir finetuning_detr4



modified_backbone --> FPN이랑 ASFF 추가 완료
배치 1로만해봤는데 성능 꽤 괜찮은듯


fourier는 효과가 없는거 같아서 뺏음

todo
1) 이대로 학습돌려보기
2) detr.py변경하고

