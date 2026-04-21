# training/dataset/vlffd_region_api.py

import random
import numpy as np
import cv2


def landmarks_to_int(landmarks):
    lm = np.asarray(landmarks, dtype=np.int32)
    if lm.ndim == 1:
        lm = lm.reshape(-1, 2)
    return lm


class VLFFDRegionAPI:
    """
    半脸(face) 用多边形，鼻子/眼睛/嘴巴用矩形框。
    real_img, fake_img: H x W x 3, uint8
    landmark: N x 2
    """

    def __init__(self, image_size=256, regions=("eyes", "nose", "mouth", "face")):
        self.image_size = image_size
        self.regions = regions

    def __call__(self, real_img, fake_img, landmark):
        try:
            h, w, _ = real_img.shape
            lm = landmarks_to_int(landmark)

            mask = np.zeros((h, w), dtype=np.float32)

            # 随机选择要用哪些 region
            num_regions = random.randint(1, len(self.regions))
            regions_to_use = random.sample(self.regions, num_regions)

            for region_name in regions_to_use:
                # 先拿到这个 region 对应的 polygon 点
                region_poly = self._get_region_polygon(region_name, lm, h, w)
                if region_poly is None:
                    continue

                if region_name == "face":
                    # 半脸：保持原来的多边形填充
                    cv2.fillConvexPoly(mask, region_poly, 1.0)
                else:
                    # 鼻子 / 眼睛 / 嘴巴：用外接矩形 + 适当扩张
                    x_min = int(np.min(region_poly[:, 0]))
                    x_max = int(np.max(region_poly[:, 0]))
                    y_min = int(np.min(region_poly[:, 1]))
                    y_max = int(np.max(region_poly[:, 1]))

                    expand_ratio = 0.15  # 想要框大一些就调大
                    x_pad = int((x_max - x_min) * expand_ratio)
                    y_pad = int((y_max - y_min) * expand_ratio)

                    x_min = max(0, x_min - x_pad)
                    x_max = min(w - 1, x_max + x_pad)
                    y_min = max(0, y_min - y_pad)
                    y_max = min(h - 1, y_max + y_pad)

                    if x_max <= x_min or y_max <= y_min:
                        continue

                    mask[y_min:y_max + 1, x_min:x_max + 1] = 1.0

            # 全 0 的话就让外层兜底
            if mask.max() < 0.5:
                return None, None

            # 合成
            mask_3 = np.repeat(mask[:, :, None], 3, axis=2)
            mixed = (mask_3 * fake_img.astype(np.float32) +
                     (1.0 - mask_3) * real_img.astype(np.float32))
            mixed = np.clip(mixed, 0, 255).astype(np.uint8)

            return mixed, mask

        except Exception as e:
            print("[VLFFDRegionAPI] error:", e)
            return None, None

    # ---------- region 对应的 landmark 点集 ----------
    def _get_region_polygon(self, name, lm, h, w):
        """
        dlib 68 点:
        - 鼻子: 27-35
        - 眼睛: 36-41(左) + 42-47(右)
        - 嘴巴: 48-60 (外圈)
        - 下颌线: 0-16
        """
        if name == "nose":
            idx = list(range(27, 36))
        elif name == "mouth":
            idx = list(range(48, 61))  # 外圈
        elif name == "eyes":
            idx = list(range(36, 48))
        elif name == "face":
            idx = list(range(0, 17))
        else:
            return None

        if lm.shape[0] < max(idx) + 1:
            return None

        pts = lm[idx, :].copy()
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        return pts.astype(np.int32)

