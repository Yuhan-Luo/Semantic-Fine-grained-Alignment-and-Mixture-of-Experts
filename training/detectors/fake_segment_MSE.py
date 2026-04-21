import os
import time
import logging
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, CLIPModel
from .base_detector import AbstractDetector
from detectors import DETECTOR
from loss import LOSSFUNC
from metrics.base_metrics_class import calculate_metrics_for_train

logger = logging.getLogger(__name__)


@DETECTOR.register_module(module_name='clip_fine_text_MSE')
class CLIP16x16Segmentor_MSE(AbstractDetector):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.get('train_batchSize')
        self.lmdb = config.get('lmdb', False)


        # 1) 初始化 CLIP
        self.processor, self.model = self.build_clip_model(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.margin = torch.tensor(0.4, device=self.device)
        self.model = self.model.to(self.device)

        # 2) 从 CLIP 配置中读 image_size / patch_size
        vision_cfg = self.model.vision_model.config
        self.image_size = vision_cfg.image_size  # 通常 224
        self.patch_size = vision_cfg.patch_size  # 通常 16
        self.seg_size = self.image_size // self.patch_size  # 224/16=14
        self.target_num_patches = self.seg_size * self.seg_size  # 14*14=196

        # hidden_size / projection_dim
        self.feat_dim = vision_cfg.hidden_size  # 768
        self.proj_dim = self.model.config.projection_dim  # 512
        clip_proj = self.model.visual_projection  # Parameter: [hidden, proj]
        self.projection = nn.Linear(768, 512, bias=(clip_proj.bias is not None)).to(self.device)
        # self.projection.load_state_dict(clip_proj.state_dict())

        clip_text_proj = self.model.text_projection  # 可能是 Linear，也可能是 Parameter
        hidden_dim = self.model.text_model.config.hidden_size
        proj_dim = self.model.config.projection_dim

        # 1) 两个 projector 结构完全一致
        self.patch_text_proj = nn.Linear(hidden_dim, proj_dim, bias=False).to(self.device)
        self.face_text_proj = nn.Linear(hidden_dim, proj_dim, bias=False).to(self.device)

        # 2) 从 CLIP 的 text_projection 拷权重
        with torch.no_grad():
            if isinstance(clip_text_proj, nn.Linear):
                # open_clip 这类实现
                w = clip_text_proj.weight.data.clone()  # [proj_dim, hidden_dim]
                self.patch_text_proj.weight.copy_(w)
                self.face_text_proj.weight.copy_(w)
            else:
                # HF CLIP: text_projection 是 nn.Parameter [hidden_dim, proj_dim]
                # CLIP 原始做法是 x @ text_projection
                w = clip_text_proj.data.clone().T  # 变成 [proj_dim, hidden_dim]
                self.patch_text_proj.weight.copy_(w)
                self.face_text_proj.weight.copy_(w)

        self.loss_func = self.build_loss(config)

        # 文本 prompt
        self.real_face_text = "this is a real face"
        self.fake_face_text = "this is a fake face"
        self.real_text = "this is a real region"
        self.fake_text = "this is a fake region"

        # mask 保存路径
        self.mask_save_path = '/media/ubuntu/3c90d67b-86b3-4bcc-b52e-138d569789d9/LYH-data/DeepfakeBench/mask_image'

    # =================== CLIP 构建 ===================

    def build_clip_model(self, config):
        model_name = "openai/clip-vit-base-patch16"
        processor = AutoProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
        return processor, model

    # 文本特征：已经过 text_projection，是 projection_dim
    # def get_text_features(self):
    #     inputs = self.processor(
    #         text=[self.real_text, self.fake_text, self.real_face_text, self.fake_face_text],
    #         return_tensors="pt",
    #         padding=True
    #     )
    #     inputs = {k: v.to(self.device) for k, v in inputs.items()}
    #     text_features = self.model.get_text_features(**inputs)      # [4, proj_dim]
    #     text_features = F.normalize(text_features, dim=-1)

    #     real_patch = text_features[0]   # [D]
    #     fake_patch = text_features[1]
    #     real_face = text_features[2]
    #     fake_face = text_features[3]
    #      return real_patch, fake_patch, real_face, fake_face

    def get_text_features(self):
        inputs = self.processor(
            text=[self.real_text, self.fake_text, self.real_face_text, self.fake_face_text],
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 原始 text encoder 输出（未 projection）
        raw = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True
        ).pooler_output  # [4, hidden]

        patch_raw = raw[:2, :]  # [2, hidden]  -> region 文本
        face_raw = raw[2:, :]  # [2, hidden]  -> face 文本

        # 分别用不同 projector
        patch_feat = self.patch_text_proj(patch_raw)  # [2, proj_dim]
        face_feat = self.face_text_proj(face_raw)  # [2, proj_dim]

        # 归一化
        patch_feat = F.normalize(patch_feat, dim=-1)
        face_feat = F.normalize(face_feat, dim=-1)

        real_patch = patch_feat[0]
        fake_patch = patch_feat[1]
        real_face = face_feat[0]
        fake_face = face_feat[1]

        return real_patch, fake_patch, real_face, fake_face

    # 图像 patch / CLS 特征：用 vision_model + visual_projection
    def get_image_patch_features(self, image):
        if isinstance(image, torch.Tensor) and image.device != self.device:
            image = image.to(self.device)

        # 这里假设 data_dict["image"] 已经按 CLIP 要求预处理了
        outputs = self.model.vision_model(image, output_hidden_states=True)
        hidden = outputs.last_hidden_state  # [B, 1+P, hidden_size]

        cls_features = hidden[:, 0, :]  # [B, H]
        patch_features = hidden[:, 1:, 0:768]  # [B, P, H]

        # 用 CLIP 自带的 visual_projection：hidden_size -> proj_dim
        visual_proj = self.model.visual_projection  # Linear(H, proj_dim)
        patch_features = self.projection(patch_features)  # [B, P, D]
        cls_features = visual_proj(cls_features)  # [B, D]

        # 归一化
        patch_features = F.normalize(patch_features, dim=-1)
        cls_features = F.normalize(cls_features, dim=-1)

        return patch_features, cls_features

    # =================== 前向 ===================

    def forward(self, data_dict: dict, inference=False) -> dict:
        
        
        images = data_dict['image']
        patch_features, cls_features = self.features(images)
        patch_logits, patch_prob, cls_logits, cls_prob = self.classifier(
            patch_features, cls_features
        )
        pred_dict = {
            'cls': cls_logits,
            'prob': cls_prob,
            'feat': cls_features,
            'patch_cls': patch_logits,
            'patch_prob': patch_prob,
            'patch_feat': patch_features,
        }
        return pred_dict

    def features(self, x):
        return self.get_image_patch_features(x)

    def get_masks(self, data_dict: dict):
        return data_dict['mask']

    def build_backbone(self, config):
        return self.model.vision_model

    def build_loss(self, config):
        loss_class = LOSSFUNC[config['loss_func']]
        return loss_class()

    # =================== classifier：对比 patch/text ===================

    def classifier(self, patch_features, cls_features):
        """
        patch_features: [B, P, D]
        cls_features:   [B, D]
        返回：
          patch_logits_flat: [B*P, 2]
          patch_prob:        [B*P] (fake 概率)
          cls_logits:        [B, 2]
          cls_prob:          [B]   (fake 概率)
        """
        B, P, D = patch_features.shape

        real_patch, fake_patch, real_face, fake_face = self.get_text_features()
        real_patch = real_patch.to(self.device)
        fake_patch = fake_patch.to(self.device)
        real_face = real_face.to(self.device)
        fake_face = fake_face.to(self.device)

        # 为广播 reshape
        real_patch = real_patch.view(1, 1, D)  # [1,1,D]
        fake_patch = fake_patch.view(1, 1, D)
        real_face = real_face.view(1, D)  # [1,D]
        fake_face = fake_face.view(1, D)

        # ---- patch 级别相似度 ----
        real_sim = torch.sum(patch_features * real_patch, dim=-1)  # [B,P]
        fake_sim = torch.sum(patch_features * fake_patch, dim=-1)  # [B,P]

        patch_logits = torch.stack([real_sim, fake_sim], dim=-1)  # [B,P,2]
        patch_logits_flat = patch_logits.reshape(-1, 2)  # [B*P,2]
        patch_prob = torch.softmax(patch_logits_flat, dim=1)[:, 1]  # fake 概率 [2B*P]


        # ---- CLS 级别相似度 ----
        real_cls_sim = torch.sum(cls_features * real_face, dim=-1)  # [B]
        fake_cls_sim = torch.sum(cls_features * fake_face, dim=-1)  # [B]

        cls_logits = torch.stack([real_cls_sim, fake_cls_sim], dim=-1)  # [B,2]
        cls_prob = torch.softmax(cls_logits, dim=1)[:, 1]  # fake 概率

        return patch_logits_flat, patch_prob, cls_logits, cls_prob

    # =================== mask / loss / metric ===================

    def process_tensor(self, label):
        """
        输入:
          label: [B, H, W] 或 [B, H, W, 1]
        返回:
          [B, P] 的 0/1 标签, P = seg_size * seg_size
        """
        P = self.seg_size * self.seg_size
        # print(label.shape)
        # ---------- 1. 基本合法性检查 ----------
        if not isinstance(label, torch.Tensor):
            # 连 tensor 都不是，构一个最小的 0
            return torch.zeros(1, P, dtype=torch.long)

        device = label.device
        # print(label.dim(),label.shape)
        # print(torch.sum(label.reshape(-1)))

        label = label[:, :, :, 0].unsqueeze(1)  # [B,1,H,W]
        B = label.shape[0]

        # print(torch.sum(label.reshape(-1)))
        # 下采样到 seg_size x seg_size
        downsampled = F.interpolate(
            label,
            size=(self.seg_size, self.seg_size),
            mode='area'
        )  # [B,1,S,S]
        # print(downsampled.shape,torch.sum(downsampled.reshape(-1)))
        label = (downsampled.squeeze(1) > 0.1).view(B, -1).long()  # [B,P]
        return label


    def get_L_rank_intra(self, data_dict: dict, pred_dict: dict):
        patch_label  =self.get_masks(data_dict)
        patch_label = self.process_tensor(patch_label)  # [2B, P]
        # B = self.batch_size
        s_fake = pred_dict['patch_prob']  # [2B*P = 12544]
        B = int((len(s_fake) / self.target_num_patches) /2)  # 最后一个batch为46张
        s_fake = s_fake.reshape(2*B, self.target_num_patches)  #[2B, P]
        L_rank_intra = torch.tensor(0.0, device=self.device)
        for b in range(B, 2*B):
            current_patch_label = patch_label[b]
            fg_indices = torch.where(current_patch_label == 1)[0]
            bg_indices = torch.where(current_patch_label == 0)[0]
            s_fg = s_fake[b, fg_indices]  # [len(fg)]
            s_bg = s_fake[b, bg_indices]  # [len(bg)]
            if len(s_fg) == 0 or len(s_bg) == 0:
                continue

            loss_matrix = s_fg.unsqueeze(1) - s_bg.unsqueeze(0)  # [len(fg), len(bg)]
            loss = torch.max(torch.zeros_like(loss_matrix), self.margin - loss_matrix)  # [len(fg), len(bg)]
            L_rank_intra += torch.mean(loss.reshape(-1))/B
            # for i in range(len(s_fg)):
            #     for j in range(len(s_bg)):
            #         loss = torch.max(torch.tensor(0.0, device=self.device), self.margin - (s_fg[i] - s_bg[j]))
            #         L_rank_intra += loss

        return L_rank_intra

    def get_L_rank_real_fake(self, data_dict: dict, pred_dict: dict):
        patch_label = self.get_masks(data_dict)
        patch_label = self.process_tensor(patch_label)  # [2B, P]
        # B = self.batch_size
        s_fake = pred_dict['patch_prob']  # [2B*P=12544]
        B = int((len(s_fake) / self.target_num_patches) / 2)
        s_fake = s_fake.reshape(2*B, self.target_num_patches)
        L_rank_real_fake = torch.tensor(0.0, device=self.device)
        for b in range(B):
            fake_patch_label = patch_label[b+B]
            fake_fg_indices = torch.where(fake_patch_label == 1)[0]
            if len(fake_fg_indices) == 0:
                continue
            s_real_fake = s_fake[b, fake_fg_indices]
            s_fake_fake = s_fake[b+B, fake_fg_indices]

            loss_matrix = s_fake_fake - s_real_fake  # [len(fake_fg_indices)]
            loss = torch.max(torch.zeros_like(loss_matrix), self.margin - loss_matrix)
            L_rank_real_fake += torch.mean(loss.reshape(-1))/B
            # for i in range(len(fake_fg_indices)):
            #     loss = torch.max(torch.tensor(0.0, device=self.device), self.margin - (s_fake_fake[i] - s_real_fake[i]))
            #     L_rank_real_fake += loss

        return L_rank_real_fake

    def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
        cls_label = data_dict['label']                      # [B]
        cls_pred = pred_dict['cls']                         # [B,2]
        loss = self.loss_func(cls_pred, cls_label)
        if self.training:
            L_rank_intra = self.get_L_rank_intra(data_dict, pred_dict)
            L_rank_real_fake = self.get_L_rank_real_fake(data_dict, pred_dict)
            # print(f'L_rank_intra:{L_rank_intra}, L_rank_real_fake:{L_rank_real_fake}')
            loss = L_rank_intra * 0.35 + L_rank_real_fake * 0.15 + loss  # 0.3 0.2

        return {'overall': loss}
        
    def save_mask_as_black_white(self, mask_tensor, save_dir, prefix="mask"):
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

    # def get_losses(self, data_dict: dict, pred_dict: dict) -> dict:
    #     # patch loss
    #     cls_label = data_dict['label']  # [B]
    #     # print('cls',cls_label.shape[0],torch.sum(cls_label.reshape(-1)))
    #     cls_pred = pred_dict['cls']  # [B,2]
    #     loss = self.loss_func(cls_pred, cls_label)
    #     loss_patch = 0
    #     if (self.training):
    #         patch_label = self.get_masks(data_dict)
    #         patch_label = self.process_tensor(patch_label)  # [B,P]
    #         B, P = patch_label.shape
    #         patch_label = patch_label.reshape(-1)  # [B*P]
    #         patch_pred = pred_dict['patch_cls']  # [B*P,2]
    #         patch_loss = self.loss_func(patch_pred, patch_label)
    #         loss = loss + patch_loss * 0.25  # * 0.2
    #     return {'overall': loss}
    #     # return {'overall': loss,'loss_patch':loss_patch}


    def get_train_metrics(self, data_dict: dict, pred_dict: dict) -> dict:
        with torch.no_grad():
            cls_label = data_dict['label']
            cls_pred = pred_dict['cls']
            auc, eer, acc, ap = calculate_metrics_for_train(
                cls_label.detach(), cls_pred.detach()
            )
            return {'acc': acc, 'auc': auc, 'eer': eer, 'ap': ap}

    # =================== 可视化辅助 ===================

    def create_fake_segmentation_map(self, pred_dict: dict):
        """
        从 patch logits 中取第一个样本，生成 seg_size×seg_size 的 0/255 mask
        """
        patch_logits = pred_dict['patch_cls']  # [B*P,2]
        P = self.target_num_patches

        data = patch_logits[:P, :]  # [P,2]
        fake_mask = (data[:, 1].detach().cpu().numpy() >
                     data[:, 0].detach().cpu().numpy()).astype(np.uint8)  # [P]

        fake_mask_2d = fake_mask.reshape(self.seg_size, self.seg_size)  # [S,S]
        segmentation_map = fake_mask_2d * 255
        return segmentation_map

    def visualize_segmentation(self, segmentation_map):
        os.makedirs(self.mask_save_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"seg_map_{timestamp}.png"
        save_path = os.path.join(self.mask_save_path, filename)

        plt.figure(figsize=(6, 6))
        plt.imshow(segmentation_map, cmap='gray', vmin=0, vmax=255)
        plt.title(f'{self.seg_size}x{self.seg_size} 伪造分割图 (白色为伪造，黑色为真实)')
        plt.axis('off')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
