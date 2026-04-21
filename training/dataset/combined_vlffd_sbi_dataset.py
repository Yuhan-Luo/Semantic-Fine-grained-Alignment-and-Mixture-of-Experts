# training/dataset/combined_vlffd_sbi_dataset.py

import random
import torch

from training.dataset.vlffd_region_dataset import VLFFDRegionDataset
from training.dataset.sbi_dataset import SBIDataset 
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
import dlib

class CombinedVLFFDSBIDataset(DeepfakeAbstractBaseDataset):
    """
    把 VLFFDRegionDataset + SBIDataset 组合在一起：
    - 统一 __getitem__ 输出：{"real": (...), "fake": (...)}
    - 统一 collate_fn：输出 image / label / mask / landmark
    - 通过 config['sbi_prob'] 控制使用 SBI 的比例（默认 0.5）
    """
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("/media/ubuntu/3c90d67b-86b3-4bcc-b52e-138d569789d9/new/DeepfakeBench/preprocessing/dlib_tools/shape_predictor_81_face_landmarks.dat")

    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        self.config = config or {}
        self.mode = mode

        # 子数据集
        self.vlffd = VLFFDRegionDataset(config, mode)
        self.sbi   = SBIDataset(config, mode)

        self.vlffd_len = len(self.vlffd)
        self.sbi_len   = len(self.sbi)
        self.transform = self.init_data_aug_method()
        
        # 使用 SBI 的概率，默认 0.5
        self.sbi_prob = self.config.get('sbi_prob', 0.5)


    def __len__(self):
        # 取两者长度的最大值，用取模循环索引
        return max(self.vlffd_len, self.sbi_len)

    def __getitem__(self, index):
        """
        返回格式与两个子数据集保持一致：
        {
            "real": (real_img_t, real_label, real_mask_t),
            "fake": (fake_img_t, fake_label, fake_mask_t),
        }
        """
        if self.mode == 'train':
            use_sbi = (random.random() < self.sbi_prob)
        else:
            # 验证 / 测试阶段你可以按需调策略，这里仍然用同样概率
            use_sbi = (random.random() < self.sbi_prob)

        if use_sbi:
            idx = index % self.sbi_len
            sample = self.sbi[idx]
            # 可选：标记一下来源，方便 debug
            sample["source"] = "sbi"
        else:
            idx = index % self.vlffd_len
            sample = self.vlffd[idx]
            sample["source"] = "vlffd"
        return sample
      
        
    # ------------------------------------------------------------------
    # 一些小工具函数：在 numpy / tensor 间来回转
    # ------------------------------------------------------------------
    def _to_numpy_img_and_mask(self, img, mask):
        """
        img: np.ndarray [H,W,3] 或 torch.Tensor [3,H,W]
        mask: np.ndarray [H,W] or [H,W,1] 或 torch.Tensor [1,H,W] / [H,W]
        """
        # image
        if isinstance(img, torch.Tensor):
            # [3,H,W] -> [H,W,3]
            img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        else:
            img_np = img

        # mask
        if isinstance(mask, torch.Tensor):
            m = mask.detach().cpu().numpy()
        else:
            m = mask

        # 统一 [H,W] or [H,W,1]
        if m is not None and m.ndim == 3 and m.shape[0] == 1:
            # [1,H,W] -> [H,W]
            m = m[0]
        return img_np, m

    def _to_tensor_img_and_mask(self, img_np, mask_np):
        """
        把 numpy 的 img / mask 转回 torch.Tensor：
        img_np: [H,W,3]
        mask_np: [H,W] 或 [H,W,1]
        """
        # image: [H,W,3] -> [3,H,W]
        img_t = torch.from_numpy(img_np).permute(2, 0, 1).float()

        # mask: [H,W] -> [1,H,W]
        if mask_np is None:
            # 给一个全 0 mask，避免后续出错
            h, w = img_np.shape[:2]
            mask_np = np.zeros((h, w), dtype=np.float32)

        if mask_np.ndim == 2:
            mask_np = mask_np[None, ...]  # [1,H,W]
        elif mask_np.ndim == 3 and mask_np.shape[-1] == 1:
            # [H,W,1] -> [1,H,W]
            mask_np = np.transpose(mask_np, (2, 0, 1))

        mask_t = torch.from_numpy(mask_np).float()
        return img_t, mask_t


    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        Args:
            img: An Image object containing the image to be augmented.
            landmark: A numpy array containing the 2D facial landmarks to be augmented.
            mask: A numpy array containing the binary mask to be augmented.

        Returns:
            The augmented image, landmark, and mask.
        """

        # Set the seed for the random number generator
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)
        
        # Create a dictionary of arguments
        kwargs = {'image': img}
        
        # Check if the landmark and mask are not None
        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')
        if mask is not None:
            mask = mask.squeeze(2)
            if mask.max() > 0:
                kwargs['mask'] = mask

        # Apply data augmentation
        transformed = self.transform(**kwargs)
        
        # Get the augmented image, landmark, and mask
        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints')
        augmented_mask = transformed.get('mask',mask)

        # Convert the augmented landmark to a numpy array
        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        # Reset the seeds to ensure different transformations for different videos
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask


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




    # @staticmethod
    # def collate_fn(batch):
    #     """
    #     期望 batch 里每个元素都是：
    #       {"real": (real_img, real_label, real_mask),
    #        "fake": (fake_img, fake_label, fake_mask),
    #        "source": ... (可选)}
    #     输出:
    #       data_dict = {
    #         'image': [2B,3,H,W],
    #         'label': [2B],
    #         'mask':  [2B,1,H,W],
    #         'landmark': None,
    #       }
    #     """
    #     # 忽略 "source"，只看 real / fake
    #     real_imgs, real_labels, real_masks = zip(*[b["real"] for b in batch])
    #     fake_imgs, fake_labels, fake_masks = zip(*[b["fake"] for b in batch])
    #
    #     real_imgs  = torch.stack(real_imgs,  dim=0)       # [B, 3, H, W]
    #     fake_imgs  = torch.stack(fake_imgs,  dim=0)
    #     real_masks = torch.stack(real_masks, dim=0)       # [B, 1, H, W]
    #     fake_masks = torch.stack(fake_masks, dim=0)
    #
    #     real_labels = torch.LongTensor(real_labels)       # [B]
    #     fake_labels = torch.LongTensor(fake_labels)
    #
    #     images = torch.cat([real_imgs,  fake_imgs],  dim=0)  # [2B, 3, H, W]
    #     labels = torch.cat([real_labels, fake_labels], dim=0)  # [2B]
    #     masks  = torch.cat([real_masks,  fake_masks],  dim=0)  # [2B, 1, H, W]
    #
    #     data_dict = {
    #         'image': images,
    #         'label': labels,
    #         'mask':  masks,
    #         'landmark': None,
    #     }
    #     return data_dict
    @staticmethod
    def get_landmarks(img):
        img = img.cpu().numpy()
        img = np.transpose(img, (1, 2, 0))  # 转化成np格式，[H,W,3]
        if img.max() <= 1 and img.min() >= 0:
            img = (img * 255).astype(np.uint8)
        elif img.max() <= 1 and img.min() >= -1:
            img = ((img + 1) / 2 * 255).astype(np.uint8)
        gray_img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

        faces = CombinedVLFFDSBIDataset.detector(gray_img)
        if len(faces) == 0:
            return np.zeros((81, 2), dtype=np.int32)  # 没检测到脸就返回全0landmarks

        shape = CombinedVLFFDSBIDataset.predictor(gray_img, faces[0])  # 有多个脸只处理第一个
        landmarks = np.zeros((81, 2), dtype=np.int32)
        for i in range(81):
            landmarks[i, 0] = shape.part(i).x
            landmarks[i, 1] = shape.part(i).y

        return landmarks
        
    @staticmethod
    def save_mask_as_black_white(mask_tensor, save_dir, prefix="mask"):
        """
        将形状为 [2B, 1, H, W] 的 mask 张量保存为黑白灰度图
        Args:
            mask_tensor (torch.Tensor): 输入mask张量，形状为 [2B, 1, H, W]，值范围建议0~1（自动归一化）
            save_dir (str): 保存图片的目录（不存在则自动创建）
            prefix (str): 图片文件名前缀，最终文件名格式：prefix_索引.png
        """
        # 1. 校验输入张量形状
        if len(mask_tensor.shape) != 4 or mask_tensor.shape[1] != 1:
            raise ValueError(f"mask_tensor形状必须为 [2B, 1, H, W]，当前为 {mask_tensor.shape}")
    
        # 2. 张量预处理：CPU -> numpy -> 压缩1通道维度
        mask_np = mask_tensor.detach().cpu().numpy()  # 转到CPU并脱离计算图
        mask_np = np.squeeze(mask_np, axis=1)  # 形状变为 [2B, H, W]
    
        # 3. 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
        # 4. 遍历每个mask，保存为黑白图
        for idx in range(mask_np.shape[0]):
            single_mask = mask_np[idx]
    
            # 5. 归一化到0~255（兼容mask值为0~1或其他范围的情况）
            single_mask = (single_mask - single_mask.min()) / (single_mask.max() - single_mask.min() + 1e-8)  # 归一化到0~1
            single_mask = (single_mask * 255).astype(np.uint8)  # 转成0~255的uint8格式
    
            # 6. 保存为黑白灰度图（cv2.IMWRITE_PNG_COMPRESSION=0 无压缩，可选）
            save_path = os.path.join(save_dir, f"{prefix}_{idx:04d}.png")  # 补零命名，方便排序
            cv2.imwrite(save_path, single_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
            if idx % 10 == 0:  # 每保存10个打印一次进度
                print(f"已保存第 {idx + 1}/{mask_np.shape[0]} 个mask：{save_path}")


    @staticmethod
    def collate_fn(batch):
        """
        期望 batch 里每个元素都是：
          {"real": (real_img, real_label, real_mask),
           "fake": (fake_img, fake_label, fake_mask),
           "source": ... (可选)}
        输出:
          data_dict = {
            'image': [2B,3,H,W],
            'label': [2B],
            'mask':  [2B,1,H,W],
            'landmark': None,
          }
        """
        # 忽略 "source"，只看 real / fake
        real_imgs, real_labels, real_masks = zip(*[b["real"] for b in batch])
        fake_imgs, fake_labels, fake_masks = zip(*[b["fake"] for b in batch])

        real_imgs  = torch.stack(real_imgs,  dim=0)       # [B, 3, H, W]
        fake_imgs  = torch.stack(fake_imgs,  dim=0)
        real_masks = torch.stack(real_masks, dim=0)       # [B, 1, H, W]
        fake_masks = torch.stack(fake_masks, dim=0)
        real_labels = torch.LongTensor(real_labels)       # [B]
        fake_labels = torch.LongTensor(fake_labels)

        real_landmarks = []
        for img in real_imgs:
            landmarks = CombinedVLFFDSBIDataset.get_landmarks(img)
            real_landmarks.append(landmarks)
        real_landmarks = np.stack(real_landmarks, axis=0)  # [B,81,2]

        fake_landmarks = []
        for img in fake_imgs:
            landmarks = CombinedVLFFDSBIDataset.get_landmarks(img)
            fake_landmarks.append(landmarks)
        fake_landmarks = np.stack(fake_landmarks, axis=0)

        images = torch.cat([real_imgs,  fake_imgs],  dim=0)  # [2B, 3, H, W]
        labels = torch.cat([real_labels, fake_labels], dim=0)  # [2B]
        masks  = torch.cat([real_masks,  fake_masks],  dim=0)  # [2B, 1, H, W]
        #CombinedVLFFDSBIDataset.save_mask_as_black_white(masks, 'mask_img/')
        #input()
        landmarks = np.concatenate([real_landmarks, fake_landmarks], axis=0)  # [2B,81,2] np
        landmarks = torch.from_numpy(landmarks)
        #print(landmarks,landmarks.shape)
        #print(landmarks[0])
        #input()

        data_dict = {
            'image': images,
            'label': labels,
            'mask':  masks,
            'landmark': landmarks,
        }
        return data_dict

