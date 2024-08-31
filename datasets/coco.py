from pathlib import Path
import os
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask
import datasets.transforms as T

class CustomCocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks=False):
        super(CustomCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

        # 이미지 ID와 어노테이션을 매핑하는 딕셔너리 생성
        self.img_id_to_annotations = {img['id']: [] for img in self.coco.dataset['images']}
        for ann in self.coco.dataset['annotations']:
            self.img_id_to_annotations[ann['image_id']].append(ann)

    def __getitem__(self, idx):
        # 실제로 로드할 이미지의 정보를 가져옴
        image_id = self.ids[idx]
        img_info = self.coco.imgs[image_id]  # loadImgs를 사용하지 않고 직접 접근
        file_name = img_info['file_name']

        # 이미지 로드
        img = Image.open(os.path.join(self.root, file_name)).convert("RGB")

        # 해당 이미지 ID에 해당하는 어노테이션 로드
        annotations = self.img_id_to_annotations.get(image_id, [])

        # 주석이 없는 경우 건너뜀
        if not annotations:
            # 다음 인덱스로 넘어감, idx + 1이 범위를 벗어나지 않는지 확인 필요
            if idx + 1 < len(self.ids):
                return self.__getitem__(idx + 1)
            else:
                # 마지막 인덱스라면 None을 반환하여 데이터 로더에서 필터링 가능
                return None, None
        
        # 이미지와 타겟 준비
        target = {'image_id': image_id, 'annotations': annotations}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        
        return img, target

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size
        image_id = target["image_id"]

        # image_id를 torch.tensor로 변환하지 않고 그대로 사용
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {
            "boxes": boxes,
            "labels": classes,
            "image_id": image_id,  # image_id는 문자열로 그대로 사용
            "area": torch.tensor([obj["area"] for obj in anno])[keep],
            "iscrowd": torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])[keep],
            "orig_size": torch.as_tensor([int(h), int(w)]),
            "size": torch.as_tensor([int(h), int(w)])
        }

        if self.return_masks:
            target["masks"] = masks
        if keypoints is not None:
            target["keypoints"] = keypoints

        return image, target

def make_coco_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    
    PATHS = {
        "train": (root / "train_images", root / "train_annotations.json"),
        "val": (root / "val_images", root / "val_annotations.json"),
        "test": (root / "test_images", None)  # Adjust as needed for your test setup
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CustomCocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    return dataset
