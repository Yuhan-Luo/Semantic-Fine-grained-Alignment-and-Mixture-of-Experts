'''
# author: Zhiyuan Yan
# email: zhiyuanyan@link.cuhk.edu.cn
# date: 2024-01-26

The code is designed for self-blending method (SBI, CVPR 2024).
'''

import sys
sys.path.append('.')

import cv2
import yaml
import torch
import numpy as np
from copy import deepcopy
import albumentations as A
from training.dataset.albu import IsotropicResize
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.sbi_api import SBI_API


class SBIDataset(DeepfakeAbstractBaseDataset):
    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        
        # Get real lists
        # Fix the label of real images to be 0
        self.real_imglist = [(img, label) for img, label in zip(self.image_list, self.label_list) if label == 0]

        # Init SBI
        self.sbi = SBI_API(phase=mode,image_size=config['resolution'])

        # Init data augmentation method
        self.transform = self.init_data_aug_method()

    def __getitem__(self, index):
        # Get the real image paths and labels
        real_image_path, real_label = self.real_imglist[index]

        # Get the landmark paths for real images
        real_landmark_path = real_image_path.replace('frames', 'landmarks').replace('.png', '.npy')
        landmark = self.load_landmark(real_landmark_path).astype(np.int32)

        # Load the real images
        real_image = self.load_rgb(real_image_path)
        real_image = np.array(real_image)  # Convert to numpy array, shape: H x W x 3

        # Generate the corresponding SBI sample
        fake_image, real_image_out, fake_mask = self.sbi(real_image, landmark)

        # 兜底：SBI 生成失败时，fake=real，label=0，mask=全 0
        if fake_image is None or real_image_out is None:
            fake_image = deepcopy(real_image)
            real_image_out = real_image
            fake_label = 0
            fake_mask_np = np.zeros(real_image.shape[:2], dtype=np.float32)  # H x W
        else:
            fake_label = 1
            # 防止 mask 为 None
            if fake_mask is None:
                fake_mask_np = np.zeros(real_image_out.shape[:2], dtype=np.float32)
            else:
                fake_mask_np = np.asarray(fake_mask, dtype=np.float32)
                if fake_mask_np.ndim == 3:
                    fake_mask_np = fake_mask_np[..., 0]

        # real mask 固定全 0
        real_mask_np = np.zeros(real_image_out.shape[:2], dtype=np.float32)  # H x W

        # To tensor and normalize for fake and real images
        fake_image_trans = self.normalize(self.to_tensor(fake_image))
        real_image_trans = self.normalize(self.to_tensor(real_image_out))

        # 把 mask 也转成 1 x H x W 的 float tensor
        fake_mask_tensor = torch.from_numpy(fake_mask_np).unsqueeze(-1)  # [1, H, W]
        real_mask_tensor = torch.from_numpy(real_mask_np).unsqueeze(-1)  # [1, H, W]

        return {
            "fake": (fake_image_trans, fake_label, fake_mask_tensor),
            "real": (real_image_trans, real_label, real_mask_tensor),
        }

        
        
    def __len__(self):
        return len(self.real_imglist)

    @staticmethod
    def collate_fn(batch):
        """
        Collate a batch of data points.

        Returns:
            data_dict: {
                'image': Tensor [B, 3, H, W],
                'label': LongTensor [B],
                'landmark': None,
                'mask': Tensor [B, 1, H, W],
            }
        """
        # Separate fake & real
        fake_images, fake_labels, fake_masks = zip(*[data["fake"] for data in batch])
        real_images, real_labels, real_masks = zip(*[data["real"] for data in batch])

        # Stack tensors
        fake_images = torch.stack(fake_images, dim=0)       # [B, 3, H, W]
        fake_labels = torch.LongTensor(fake_labels)         # [B]
        fake_masks  = torch.stack(fake_masks, dim=0)        # [B, 1, H, W]

        real_images = torch.stack(real_images, dim=0)
        real_labels = torch.LongTensor(real_labels)
        real_masks  = torch.stack(real_masks, dim=0)

        # Concatenate fake + real along batch dim
        images = torch.cat([real_images, fake_images], dim=0)   # [2B, 3, H, W]
        labels = torch.cat([real_labels, fake_labels], dim=0)   # [2B]
        masks  = torch.cat([real_masks, fake_masks], dim=0)     # [2B, 1, H, W]

        data_dict = {
            'image': images,
            'label': labels,
            'landmark': None,
            'mask': masks,
        }
        return data_dict


    def init_data_aug_method(self):
        trans = A.Compose([           
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(blur_limit=self.config['data_aug']['blur_limit'], p=self.config['data_aug']['blur_prob']),
            A.OneOf([                
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR),
                IsotropicResize(max_side=self.config['resolution'], interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR),
            ], p = 0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=self.config['data_aug']['brightness_limit'], contrast_limit=self.config['data_aug']['contrast_limit']),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(quality_lower=self.config['data_aug']['quality_lower'], quality_upper=self.config['data_aug']['quality_upper'], p=0.5)
        ], 
            additional_targets={'real': 'sbi'},
        )
        return trans


if __name__ == '__main__':
    with open('/data/home/zhiyuanyan/DeepfakeBench/training/config/detector/sbi.yaml', 'r') as f:
        config = yaml.safe_load(f)
    train_set = SBIDataset(config=config, mode='train')
    train_data_loader = \
        torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=config['train_batchSize'],
            shuffle=True, 
            num_workers=0,
            collate_fn=train_set.collate_fn,
        )
    from tqdm import tqdm
    for iteration, batch in enumerate(tqdm(train_data_loader)):
        print(iteration)
        if iteration > 10:
            break
