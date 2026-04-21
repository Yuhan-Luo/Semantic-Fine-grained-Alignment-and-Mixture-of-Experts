# training/dataset/vlffd_region_dataset.py

import os
import cv2
import yaml
import torch
import numpy as np
from copy import deepcopy
import random          

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
        self.config = config
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
            image_size=config.get('resolution', 256),
            regions=config.get('vlffd_regions', ("eyes", "nose", "mouth", "face"))
        )

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
            # print(key,r_norm)
        # print(real_list,real_index)

        # 2) 遍历 fake，为每个 fake 找 real
        pairs = []
        for f, _ in fake_list:
            # sprint(f)
            f_norm = f.replace('/', '\\')
            parts = f_norm.split('\\')

            # 期待: [..., 'manipulated_sequences', '<method>', 'c23', 'frames', '<src>_<tgt>', '<frame>.png']

            if 'manipulated_sequences' not in parts:
                continue
            
            # print(f_norm,parts)
            j = parts.index('manipulated_sequences')

            method  = parts[j + 1]          # Deepfakes / Face2Face / ...
            quality = parts[j + 2]          # c23 / c40
            pair_id = parts[j + 4]          # '897_969'
            frame_f = parts[j + 5]          # '148.png'
            frame   = frame_f.replace('.png', '')

            # <src>_<tgt> 取右边这个作为目标视频
            if '_' in pair_id:
                tgt_vid = pair_id.split('_')[0]
            else:
                tgt_vid = pair_id

            key = (quality, tgt_vid, frame)
            # print('fake',key,f_norm)

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
        # 例子：'FaceForensics++\\original_sequences\\youtube\\c23\\frames\\022\\198.png'

        # 2) 用原始 key 从 LMDB / 磁盘中读图片
        real_img = self.load_rgb(real_rel)  # 这里必须用原始 key，不能 replace，也不能加 rgb_dir
        fake_img = self.load_rgb(fake_rel)
    
        real_img = np.array(real_img)  # H x W x 3
        fake_img = np.array(fake_img)

        # 3) 构造 landmark 的 key（注意：不是文件路径，是 LMDB key）
        #    只做 frames -> landmarks & .png -> .npy 的替换
        real_lm_key = real_rel.replace('frames', 'landmarks').replace('.png', '.npy')
        # 千万别加 rgb_dir / replace('\\', '/')，否则 key 就和 LMDB 不匹配了

        # 4) 从 LMDB 读取 landmark
        landmark = self.load_landmark(real_lm_key).astype(np.int32)  # shape [81,2]

        # ====== VLFFD region 替换 ======
        use_augment = (self.mode == 'train') and (random.random() < 0.5)

        if use_augment:
            # 做 VLFFD 区域替换增强
            mixed_fake, mask = self.region_api(real_img, fake_img, landmark)
            if mixed_fake is None or mask is None:
                mixed_fake = fake_img.copy()
                mask = np.zeros(real_img.shape[:2], dtype=np.float32)
        else:
            mixed_fake = fake_img.copy()
            mask = np.zeros(real_img.shape[:2], dtype=np.float32)
        
        real_mask = np.zeros(real_img.shape[:2], dtype=np.float32)
        # 5) 转 tensor + normalize
        real_img_t = self.normalize(self.to_tensor(real_img))
        fake_img_t = self.normalize(self.to_tensor(mixed_fake))

        real_mask_t = torch.from_numpy(real_mask).unsqueeze(-1)  # [1,H,W]
        fake_mask_t = torch.from_numpy(mask).unsqueeze(-1)       # [1,H,W]

        real_label = 0
        fake_label = 1
        
        # ===== Debug Save =====
        DEBUG_SAVE = False   # 你想关掉就改 False
        DEBUG_SAVE_DIR = "/media/ubuntu/3c90d67b-86b3-4bcc-b52e-138d569789d9/new/DeepfakeBench/training/dataset/111"

        if DEBUG_SAVE:  # 避免每张都保存，保存前10张即可
            os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)

            # BGR->RGB (cv2读取) 如果你的 load_rgb 是 PIL 则不用转换
            real_bgr = real_img.copy()
            fake_bgr = mixed_fake.copy()
            mask_img = (mask * 255).astype(np.uint8)

            # 保存
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{index:04d}_real.png"), real_bgr)
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{index:04d}_fake_region.png"), fake_bgr)
            cv2.imwrite(os.path.join(DEBUG_SAVE_DIR, f"{index:04d}_mask.png"), mask_img)
            # exit(0)
        
        

        return {
            "real": (real_img_t, real_label, real_mask_t),
            "fake": (fake_img_t, fake_label, fake_mask_t),
        }




    @staticmethod
    def collate_fn(batch):
        """
        输出:
          data_dict = {
            'image': [2B,3,H,W],
            'label': [2B],
            'mask':  [2B,1,H,W],
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

