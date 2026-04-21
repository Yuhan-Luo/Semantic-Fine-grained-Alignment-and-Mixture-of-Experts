import os
import cv2
import yaml
import torch
import numpy as np
from copy import deepcopy
import random          

import albumentations as A

from training.dataset.albu import IsotropicResize
from training.dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from training.dataset.vlffd_region_api import VLFFDRegionAPI
from detectors import DETECTOR  # 如果你想在 config 里直接用这个模块名


class VLFFDRegionDataset(DeepfakeAbstractBaseDataset):
    """
    VLFFD 风格的 region 替换数据集：
    - 输入：DeepfakeBench 原本的 image_list / label_list (0=real,1=fake)
    - 逻辑：对每个 fake 样本找到它对应的 real 样本，做局部替换生成 mixed_fake + mask
    - __getitem__ 返回 real + fake，两者都能进 batch
    """

    def __init__(self, config=None, mode='train'):
        super().__init__(config, mode)
        self.config = config or {}
        self.mode = mode

        # 构建 real / fake 列表
        self.real_imglist = [(img, lbl) for img, lbl in
                             zip(self.image_list, self.label_list) if lbl == 0]
        self.fake_imglist = [(img, lbl) for img, lbl in
                             zip(self.image_list, self.label_list) if lbl == 1]

        # real-fake 配对（这里给一个 FF++ 风格的默认实现）
        self.pairs = self.build_pairs_ffpp(self.real_imglist, self.fake_imglist)
        print("=== DEBUG PAIRS ===")
        for i in range(5):
            print("REAL:", self.pairs[i][0])
            print("FAKE:", self.pairs[i][1])
        # exit(0)

        # Region 替换 API
        self.region_api = VLFFDRegionAPI(
            image_size=self.config.get('resolution', 256),
            regions=self.config.get('vlffd_regions', ("eyes", "nose", "mouth", "face"))
        )

        # 这里初始化 Albumentations 数据增强（如果 config 里没配 data_aug，则为 None，不做增强）
        self.transform = self.init_data_aug_method()

    def build_pairs_ffpp(self, real_list, fake_list):
        """
        以 fake 为主，根据 FF++ 的路径规则精确找到对应 real：
          fake: FaceForensics++\\manipulated_sequences\\<method>\\c23\\frames\\<src>_<tgt>\\<frame>.png
          real: FaceForensics++\\original_sequences\\youtube\\c23\\frames\\<tgt>\\<frame>.png
        返回:
          pairs: list[(real_rel, fake_rel)]
        """
        # 1) 整理所有 real，建索引: (quality, vid, frame) -> real_path
        real_index = {}
        for r, _ in real_list:
            r_norm = r.replace('/', '\\')
            parts = r_norm.split('\\')

            # 期待: [..., 'original_sequences', 'youtube', 'c23', 'frames', '<vid>', '<frame>.png']
            if 'original_sequences' not in parts:
                continue
            i = parts.index('original_sequences')
            if i + 5 >= len(parts):
                continue

            quality = parts[i + 2]          # c23 / c40
            vid     = parts[i + 4]          # '374'
            frame_f = parts[i + 5]          # '264.png'
            frame   = frame_f.replace('.png', '')

            key = (quality, vid, frame)
            real_index[key] = r_norm

        # 2) 遍历 fake，为每个 fake 找 real
        pairs = []
        for f, _ in fake_list:
            f_norm = f.replace('/', '\\')
            parts = f_norm.split('\\')

            # 期待: [..., 'manipulated_sequences', '<method>', 'c23', 'frames', '<src>_<tgt>', '<frame>.png']
            if 'manipulated_sequences' not in parts:
                continue

            j = parts.index('manipulated_sequences')

            method  = parts[j + 1]          # Deepfakes / Face2Face / ...
            quality = parts[j + 2]          # c23 / c40
            pair_id = parts[j + 4]          # '897_969'
            frame_f = parts[j + 5]          # '148.png'
            frame   = frame_f.replace('.png', '')

            # <src>_<tgt> 取左边这个作为目标视频（你之前写的是 split('_')[0]）
            if '_' in pair_id:
                tgt_vid = pair_id.split('_')[0]
            else:
                tgt_vid = pair_id

            key = (quality, tgt_vid, frame)

            if key in real_index:
                real_rel = real_index[key]
                pairs.append((real_rel, f_norm))
            # 没找到的 fake 直接丢掉

        print(f"[VLFFD] real={len(real_list)}, fake={len(fake_list)}, pairs={len(pairs)}")

        if len(pairs) == 0:
            raise RuntimeError("[VLFFD] 没有成功匹配到任何 real-fake 对，请检查路径是否真的是 FF++ 格式。")

        # 再打印几对确认一下
        for k in range(min(5, len(pairs))):
            print("PAIR_DEBUG REAL:", pairs[k][0])
            print("PAIR_DEBUG FAKE:", pairs[k][1])

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # 1) 原始相对路径（lmdb key），保持原样
        real_rel, fake_rel = self.pairs[index]

        # 2) 用原始 key 从 LMDB / 磁盘中读图片 -> numpy, H x W x 3
        real_img = np.array(self.load_rgb(real_rel))
        fake_img = np.array(self.load_rgb(fake_rel))

        # 3) 构造 landmark 的 key（注意：不是文件路径，是 LMDB key）
        real_lm_key = real_rel.replace('frames', 'landmarks').replace('.png', '.npy')

        # 4) 从 LMDB 读取 landmark
        landmark = self.load_landmark(real_lm_key).astype(np.int32)  # shape [81,2]

        # ====== VLFFD region 替换（局部混合伪造） ======
        use_region_aug = (self.mode == 'train') and (random.random() < 0.5)

        if use_region_aug:
            mixed_fake, mask = self.region_api(real_img, fake_img, landmark)
            if mixed_fake is None or mask is None:
                mixed_fake = fake_img.copy()
                mask = np.zeros(real_img.shape[:2], dtype=np.float32)
        else:
            mixed_fake = fake_img.copy()
            mask = np.zeros(real_img.shape[:2], dtype=np.float32)

        real_mask = np.zeros(real_img.shape[:2], dtype=np.float32)

        # ====== Albumentations 图像级增强（新加） ======
        # 只在 train 时做；且 real/fake 独立增强即可（不需要几何对齐）
        if self.mode == 'train' and self.transform is not None:
            # real: img + mask
            real_img, _, real_mask_aug = self.data_aug(
                img=real_img,
                landmark=None,
                mask=real_mask
            )
            real_mask = real_mask_aug

            # fake: img + mask
            mixed_fake, _, fake_mask_aug = self.data_aug(
                img=mixed_fake,
                landmark=None,
                mask=mask
            )
            mask = fake_mask_aug

        # 5) 转 tensor + normalize
        # real_img / mixed_fake 现在仍然是 numpy [H,W,3]
        real_img_t = self.normalize(self.to_tensor(real_img))       # [3,H,W]
        fake_img_t = self.normalize(self.to_tensor(mixed_fake))     # [3,H,W]

        # mask: numpy [H,W] -> [H,W,1] 再转 tensor，保持你原来的格式
        real_mask_t = torch.from_numpy(real_mask.astype(np.float32)).unsqueeze(-1)  # [H,W,1]
        fake_mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(-1)       # [H,W,1]

        real_label = 0
        fake_label = 1

        # ===== Debug Save（保持你原来的逻辑） =====
        DEBUG_SAVE = False
        DEBUG_SAVE_DIR = "/media/ubuntu/3c90d67b-86b3-4bcc-b52e-138d569789d9/new/DeepfakeBench/training/dataset/111"

        if DEBUG_SAVE:
            os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

            real_bgr = real_img[..., ::-1].copy()      # 如果 load_rgb 是 RGB，这里转 BGR 给 cv2
            fake_bgr = mixed_fake[..., ::-1].copy()
            mask_img = (mask * 255).astype(np.uint8)

            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{index:04d}_real.png"), real_bgr)
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{index:04d}_fake_region.png"), fake_bgr)
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{index:04d}_mask.png"), mask_img)

        return {
            "real": (real_img_t, real_label, real_mask_t),
            "fake": (fake_img_t, fake_label, fake_mask_t),
        }

    # ===================== Albumentations 部分 =====================

    def data_aug(self, img, landmark=None, mask=None, augmentation_seed=None):
        """
        Apply data augmentation to an image, landmark, and mask.

        img: numpy [H,W,3]
        landmark: (N,2) or None
        mask: numpy [H,W] 或 [H,W,1] 或 None

        返回增强后的 (image, landmark, mask)，mask 返回 [H,W] 形式
        """
        if self.transform is None:
            # 没配置增强，直接返回原数据
            return img, landmark, mask

        # 控制随机种子（可选）
        if augmentation_seed is not None:
            random.seed(augmentation_seed)
            np.random.seed(augmentation_seed)

        kwargs = {'image': img}

        if landmark is not None:
            kwargs['keypoints'] = landmark
            kwargs['keypoint_params'] = A.KeypointParams(format='xy')

        if mask is not None:
            m = mask
            # 保证传给 albumentations 时是 [H,W]
            if m.ndim == 3 and m.shape[2] == 1:
                m = m[:, :, 0]
            kwargs['mask'] = m

        transformed = self.transform(**kwargs)

        augmented_img = transformed['image']
        augmented_landmark = transformed.get('keypoints', landmark)
        augmented_mask = transformed.get('mask', mask)

        if augmented_landmark is not None:
            augmented_landmark = np.array(augmented_landmark)

        # 重置随机种子（避免影响外部）
        if augmentation_seed is not None:
            random.seed()
            np.random.seed()

        return augmented_img, augmented_landmark, augmented_mask

    def init_data_aug_method(self):
        """
        初始化 Albumentations 的增强 pipeline。
        需要 config 中有：
          - resolution
          - with_landmark (bool，可选)
          - data_aug: dict, 如
            data_aug:
              flip_prob: 0.5
              rotate_limit: 10
              rotate_prob: 0.5
              blur_limit: 3
              blur_prob: 0.2
              brightness_limit: 0.1
              contrast_limit: 0.1
              quality_lower: 40
              quality_upper: 100
        """
        if 'data_aug' not in self.config:
            return None

        da_cfg = self.config['data_aug']
        resolution = self.config.get('resolution', 256)
        with_landmark = self.config.get('with_landmark', False)

        trans = A.Compose(
            [
                A.HorizontalFlip(p=da_cfg.get('flip_prob', 0.5)),
                A.Rotate(
                    limit=da_cfg.get('rotate_limit', 10),
                    p=da_cfg.get('rotate_prob', 0.5)
                ),
                A.GaussianBlur(
                    blur_limit=da_cfg.get('blur_limit', 3),
                    p=da_cfg.get('blur_prob', 0.2)
                ),
                A.OneOf([
                    IsotropicResize(
                        max_side=resolution,
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_CUBIC
                    ),
                    IsotropicResize(
                        max_side=resolution,
                        interpolation_down=cv2.INTER_AREA,
                        interpolation_up=cv2.INTER_LINEAR
                    ),
                    IsotropicResize(
                        max_side=resolution,
                        interpolation_down=cv2.INTER_LINEAR,
                        interpolation_up=cv2.INTER_LINEAR
                    ),
                ], p=0 if with_landmark else 1),
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=da_cfg.get('brightness_limit', 0.1),
                        contrast_limit=da_cfg.get('contrast_limit', 0.1)
                    ),
                    A.FancyPCA(),
                    A.HueSaturationValue()
                ], p=0.5),
                A.ImageCompression(
                    quality_lower=da_cfg.get('quality_lower', 40),
                    quality_upper=da_cfg.get('quality_upper', 100),
                    p=0.5
                )
            ],
            keypoint_params=A.KeypointParams(format='xy') if with_landmark else None
        )
        return trans

    # ===================== collate 不动 =====================

    @staticmethod
    def collate_fn(batch):
        """
        输出:
          data_dict = {
            'image': [2B,3,H,W],
            'label': [2B],
            'mask':  [2B,H,W,1],
            'landmark': None,
          }
        """
        real_imgs, real_labels, real_masks = zip(*[b["real"] for b in batch])
        fake_imgs, fake_labels, fake_masks = zip(*[b["fake"] for b in batch])

        real_imgs = torch.stack(real_imgs, dim=0)
        fake_imgs = torch.stack(fake_imgs, dim=0)
        real_labels = torch.LongTensor(real_labels)
        fake_labels = torch.LongTensor(fake_labels)
        real_masks = torch.stack(real_masks, dim=0)
        fake_masks = torch.stack(fake_masks, dim=0)

        images = torch.cat([real_imgs, fake_imgs], dim=0)
        labels = torch.cat([real_labels, fake_labels], dim=0)
        masks = torch.cat([real_masks, fake_masks], dim=0)

        data_dict = {
            'image': images,
            'label': labels,
            'mask': masks,
            'landmark': None,
        }
        return data_dict

