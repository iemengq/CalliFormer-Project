import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
import numpy as np
import cv2
from PIL import Image
import os
from pathlib import Path
from tqdm import tqdm
import datetime  # 新增时间模块
import json
import torch.nn.functional as F
import numpy as np
from dynamic_structural_bias import DynamicStructuralBiasGenerator
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import Tuple
from collections import defaultdict
import random


class CalliGANDataset(Dataset):
    """PyTorch 版 CalliGAN 数据集 (保持与 TensorFlow 版相同的数据流)"""

    def __init__(self,
                 images_path: str,
                 json_path: str,
                 max_comp_len: int = 28,
                 max_struct_len: int = 3,
                 img_size: int = 256,
                 augment: bool = False,
                 samples_per_class: int = None,  # 新增参数：每类样本数
                 random_seed: int = 42
                 ):
        """
        参数:
            images_path: 图像目录路径
            json_path: 汉字部件编码文件路径
            img_size: 输出图像尺寸
            augment: 是否启用数据增强
            samples_per_class: 每个类别选取的样本数 (None表示使用全部数据)
        """
        super().__init__()
        # 预处理参数
        self.max_comp_len = max_comp_len  # 组件序列最大长度
        self.max_struct_len = max_struct_len  # 结构序列最大长度
        self.pad_idx = 0

        self.img_size = img_size
        self.augment = augment

        # 加载汉字到部件的映射
        self.hanzi2components = self._load_component_mapping(json_path)
        self.hanzi2structural = self._load_structural_mapping(json_path)
        self.image_files = []  # 最终使用的图像路径列表

        # 加载所有图像路径并按类别分组
        all_files = self._load_all_image_paths(images_path)
        class_files = self._group_files_by_class(all_files)

        # 分层采样逻辑
        if samples_per_class is not None:
            self._validate_sampling_params(class_files, samples_per_class)
            self._stratified_sampling(class_files, samples_per_class, random_seed)
        else:
            self.image_files = all_files  # 使用全部数据

        print(f'数据集初始化完成，总样本数: {len(self)}')

        # 图像预处理流水线
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为Tensor [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到 [-1,1]
        ])

        # 定义用于“风格参考图像”的转换 (输入给ResNet)
        # 输出符合ImageNet预训练要求
        self.style_transform = transforms.Compose([
            transforms.ToTensor()  # 转换为Tensor [0, 1]
        ])

    def _load_component_mapping(self, path: str) -> dict:
        """加载汉字到部件编码的映射表"""
        mapping = {}
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                hanzi, components = item['character'], item['components']
                mapping[hanzi] = components
        return mapping

    def _load_structural_mapping(self, path: str) -> dict:
        """加载汉字到部件编码的映射表"""
        mapping = {}
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                hanzi, structural = item['character'], item['structures_code']
                mapping[hanzi] = structural
        return mapping

    def _load_all_image_paths(self, path: str) -> list:
        """加载目录下所有PNG图像路径"""
        return [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith('.png')
        ]

    def _group_files_by_class(self, files: list) -> dict:
        """按类别分组文件路径"""
        class_dict = defaultdict(list)
        for f in files:
            class_idx = self._parse_filename(f)[1]  # 获取0-based类别
            class_dict[class_idx].append(f)
        return class_dict

    def _validate_sampling_params(self, class_files: dict, n: int):
        """验证采样参数有效性"""
        min_samples = min(len(v) for v in class_files.values())
        if n > min_samples:
            raise ValueError(
                f"无法从所有类别中选取{n}个样本，"
                f"最小可用类别样本数为{min_samples}"
            )

    def _stratified_sampling(self, class_files: dict, n: int, seed: int):
        """执行分层采样"""
        random.seed(seed)  # 固定随机种子

        # 清空并重新填充图像路径列表
        self.image_files.clear()
        for cls in sorted(class_files.keys()):  # 保证顺序一致
            sampled = random.sample(class_files[cls], n)
            self.image_files.extend(sampled)

        # 打乱整体顺序以避免类别连续
        random.shuffle(self.image_files)

    def __len__(self) -> int:
        return len(self.image_files)

    def _parse_filename(self, path: str) -> Tuple[str, int]:
        """解析文件名获取汉字和字体类别
        文件名格式: {字体编号}_{汉字}_{序号}.png
        示例: 1_阿_206903.png
        """
        filename = os.path.basename(path)
        parts = filename.split('_')
        font_idx = int(parts[0])  # 字体编号 (1-based)
        hanzi = parts[1]  # 汉字字符
        return hanzi, font_idx - 1  # 转为 0-based

    def _get_components(self, hanzi: str) -> torch.Tensor:
        """获取汉字部件编码 (填充/截断到长度28)"""
        # 从字典中返回汉字对应的部件编码，如果没有则返回28位全0向量
        components = self.hanzi2components.get(hanzi, [0] * self.max_comp_len)
        # 填充或截断到固定长度28
        if len(components) < self.max_comp_len:
            components += [0] * (self.max_comp_len - len(components))
        else:
            components = components[:self.max_comp_len]
        return torch.tensor(components, dtype=torch.long)

    def _get_structural(self, hanzi: str) -> torch.Tensor:
        """获取汉字部件编码 (填充/截断到长度3)"""
        # 从字典中返回汉字对应的部件编码，如果没有则返回3位全0向量
        structural = self.hanzi2structural.get(hanzi, [0] * self.max_struct_len)
        if len(structural) < self.max_struct_len:
            structural += [0] * (self.max_struct_len - len(structural))
        else:
            structural = structural[:self.max_struct_len]
        return torch.tensor(structural, dtype=torch.long)

    def _get_category(self, font_idx: int) -> torch.Tensor:
        """生成字体类别的 one-hot 编码"""
        return torch.nn.functional.one_hot(
            torch.tensor(font_idx),
            num_classes=7
        ).float()

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        # 读取图像
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert('L')  # 转为灰度图

        # 分割源图像和目标图像 (左右拼接)
        width = img.width // 2
        source = img.crop((0, 0, width, img.height))
        target = img.crop((width, 0, img.width, img.height))
        target_style = img.crop((width, 0, img.width, img.height))

        # 数据增强 (此处可扩展)
        if self.augment:
            # 示例: 随机亮度调整
            brightness = np.random.uniform(0.6, 1.4)
            source = transforms.functional.adjust_brightness(source, brightness)
            target = transforms.functional.adjust_brightness(target, brightness)
            target_style = transforms.functional.adjust_brightness(target_style, brightness)

        # 调整尺寸并应用变换
        source = self.transform(source.resize((self.img_size, self.img_size)))
        target = self.transform(target.resize((self.img_size, self.img_size)))
        style_image = self.style_transform(target_style.resize((self.img_size, self.img_size)))

        # 解析元数据
        hanzi, font_idx = self._parse_filename(img_path)
        components = self._get_components(hanzi)
        structural = self._get_structural(hanzi)
        # category = self._get_category(font_idx)

        assert torch.isfinite(source).all(), f"Source image含非法值: {img_path}"
        assert torch.isfinite(target).all(), f"Target image含非法值: {img_path}"

        return (components, structural, source, style_image), target


class ResNetStyleEncoder(nn.Module):
    """
    使用预训练的 ResNet-34 从图像中提取风格向量。
    """

    def __init__(self, pretrained=True, freeze=True):
        super().__init__()
        # 加载预训练的 ResNet-34 模型
        weights = ResNet34_Weights.IMAGENET1K_V1
        resnet = resnet34(weights=weights)

        # 我们需要 ResNet 在全局平均池化后的特征，而不是最终的分类结果
        # 因此，我们取除了最后一个全连接层(fc)之外的所有层
        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # 将ImageNet归一化层定义为模型的一部分
        # 这样模型就能处理[0, 1]范围的输入图像
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        # 获取风格向量的维度 (对于ResNet-34是512)
        self.output_dim = resnet.fc.in_features
        if freeze:
            # 冻结所有参数，因为我们只用它作特征提取器，不进行训练
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        输入 x: 风格参考图像，形状为 [batch_size, 3, H, W] (ResNet需要3通道输入)
        输出: 风格向量，形状为 [batch_size, output_dim]
        """
        # 设置为评估模式，这会禁用 Dropout 和 BatchNorm 的更新
        self.features.eval()

        # 如果输入是单通道 (C=1)
        if x.shape[1] == 1:
            # 将单通道复制三次，变为三通道
            # (N, 1, H, W) -> (N, 3, H, W)
            x = x.repeat(1, 3, 1, 1)
        # --- 新增：在模型内部进行归一化 ---
        # 现在 x 保证是三通道了，可以应用ImageNet的归一化
        x = self.normalize(x)
        # 提取特征
        features = self.features(x)

        # 将输出从 [batch, 512, 1, 1] 展平为 [batch, 512]
        style_vector = torch.flatten(features, 1)

        return style_vector


class StructuralAttention(nn.Module):
    def __init__(self, d_model=256, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.embed_dim = d_model
        self.batch_first = True  # 强制batch维度在前

        # 显式定义投影层
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, struct_bias=None, attn_mask=None):
        # 添加参数检查
        assert query is not None, "必须提供query参数"
        assert key is not None, "必须提供key参数"
        assert value is not None, "必须提供value参数"
        assert query.size() == key.size() == value.size(), "输入形状不一致"
        if attn_mask is not None:
            assert attn_mask.dim() == 4, f"attn_mask应为4D，实际是{attn_mask.dim()}D"

        # 输入形状: [batch, seq_len, d_model]
        batch_size, tgt_len, _ = query.shape
        src_len = key.shape[1]

        # 1. 投影到Q/K/V
        q = self.q_proj(query)  # [batch, tgt_len, d_model]
        k = self.k_proj(key)  # [batch, src_len, d_model]
        v = self.v_proj(value)  # [batch, src_len, d_model]

        # 2. 分头处理
        q = q.view(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        k = k.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]
        v = v.view(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, S, D/H]

        # 3. 计算注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, T, S]

        # 4. 添加结构偏置
        if struct_bias is not None:
            attn_scores += struct_bias  # struct_bias形状需为 [B, H, T, S]

        # 5. 处理掩码（可选）
        if attn_mask is not None:
            attn_scores += attn_mask

        # 6. Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, H, T, S]

        # 7. 计算上下文向量
        context = torch.matmul(attn_weights, v)  # [B, H, T, D/H]

        # 8. 合并多头输出
        context = context.transpose(1, 2).contiguous().view(batch_size, tgt_len, -1)  # [B, T, D]

        # 9. 最终投影
        output = self.out_proj(context)  # [B, T, D]
        return output, attn_weights


class CalliGAN(nn.Module):
    """CalliGAN 训练器 (包含完整训练流程和可视化功能)

        功能：
        - 混合精度训练
        - 自动设备检测 (GPU/CPU)
        - 训练过程监控
        - 模型检查点保存
        - 训练可视化

        参数：
        num_classes: 字体类别数
        l1_coef: L1损失系数
        const_coef: 特征一致性损失系数
        device: 训练设备 (自动检测)
        """

    def __init__(self, style_vector_dim=512, image_size=256, gen_train_steps=2, cns_coder='transformer', lr=0.0002):
        super().__init__()
        self.style_vector_dim = style_vector_dim
        self.image_size = image_size
        self.gen_train_steps = gen_train_steps  # 生成器训练次数控制
        self.cns_coder = cns_coder
        self.lr = lr

        # 设备检测
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化组件
        self.generator = CalliGenerator(style_vector_dim, self.cns_coder).to(self.device)
        self.discriminator = CalliDiscriminator().to(self.device)

        self.g_optim = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr * 0.2, betas=(0.5, 0.999))

        # 损失权重系数
        self.L1_penalty = 100.0
        self.Lconst_penalty = 15.0
        self.Lcategory_penalty = 1.0
        self.Ltv_penalty = 0.0
        self.Ladv_d_penalty = 1.0
        self.Ladv_g_penalty = 5.0
        self.Lstyle_penalty = 20.0 # 风格损失的权重，需要调试

        # 混合精度训练
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))

        # 训练历史记录
        self.history = {
            'g_loss': [],
            'd_loss': [],
            # 'val_loss': []
        }

        self._init_weights()

        # 将此行添加到您的训练器类的 __init__ 方法中
        self.normalize_for_resnet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])

    def _init_weights(self):
        """权重初始化 (He初始化)"""
        for m in self.generator.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.trunc_normal_(m.weight, 0, 0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        for m in self.discriminator.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.trunc_normal_(m.weight, 0, 0.02, a=-0.04, b=0.04)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def train_step(self, real_X, real_y):
        # 数据转移到设备
        real_components = real_X[0].to(self.device)
        real_structural = real_X[1].to(self.device)
        content_images = real_X[2].to(self.device)
        style_images = real_X[3].to(self.device)
        ground_truth_images = real_y.to(self.device)

        # 训练判别器
        d_loss, real_adv_loss, fake_adv_loss = self._train_discriminator(real_components, real_structural, content_images, style_images, ground_truth_images)

        # 训练生成器（根据参数控制次数）
        g_loss_total = 0.0
        pixel_loss_total = 0.0
        const_loss_total = 0.0
        adv_loss_total = 0.0
        style_loss_total = 0.0
        tv_loss_total = 0.0
        for _ in range(self.gen_train_steps):
            g_loss, pixel_loss, const_loss, adv_loss, style_loss, tv_loss = self._train_generator(real_components, real_structural, content_images, style_images, ground_truth_images)
            g_loss_total += g_loss
            pixel_loss_total += pixel_loss
            const_loss_total += const_loss
            adv_loss_total += adv_loss
            style_loss_total += style_loss
            tv_loss_total += tv_loss
        avg_g_total_loss = g_loss_total / self.gen_train_steps
        avg_pixel_loss_total = pixel_loss_total / self.gen_train_steps
        avg_const_loss_total = const_loss_total / self.gen_train_steps
        avg_adv_loss_total = adv_loss_total / self.gen_train_steps
        avg_style_loss_total = style_loss_total / self.gen_train_steps
        avg_tv_loss_total = tv_loss_total / self.gen_train_steps

        losses_dict = {
            # 总损失
            'G_total': avg_g_total_loss,
            'D_total': d_loss,
            # 判别器原始损失 (Raw Discriminator Losses)
            'raw_adv_d_real': real_adv_loss,
            'raw_adv_d_fake': fake_adv_loss,
            # 生成器原始损失 (Raw Generator Losses) - 这是模型最原始的反馈
            'raw_pixel': avg_pixel_loss_total,
            'raw_const': avg_const_loss_total,
            'raw_adv_g': avg_adv_loss_total,
            'raw_style': avg_style_loss_total,
            'raw_tv': avg_tv_loss_total,
            # 生成器加权损失 (Weighted Generator Losses) - 这是决定梯度方向的关键
            'weighted_pixel': avg_pixel_loss_total * self.L1_penalty,
            'weighted_const': avg_const_loss_total * self.Lconst_penalty,
            'weighted_adv_g': avg_adv_loss_total * self.Ladv_g_penalty,
            'weighted_style': avg_style_loss_total * self.Lstyle_penalty,
            'weighted_tv': avg_tv_loss_total * self.Ltv_penalty,
        }
        self.history['g_loss'].append(avg_g_total_loss)
        self.history['d_loss'].append(d_loss)
        # return {'g_loss': self.history['g_loss'][-1], 'd_loss': self.history['d_loss'][-1]}
        return losses_dict

    def _train_discriminator(self, real_component, real_structural, content_images, style_images, ground_truth_images):
        self.d_optim.zero_grad()

        # 生成假图像
        with torch.no_grad():
            fake_images = self.generator(real_component, real_structural, content_images, style_images)

        # 混合精度上下文
        with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu',
                      enabled=(self.device.type == 'cuda')):
            # 真实样本判别
            real_input = torch.cat([content_images, ground_truth_images], dim=1)
            real_pred = self.discriminator(real_input)
            real_adv_loss = nn.MSELoss()(real_pred, torch.ones_like(real_pred))
            # real_cat_loss = nn.CrossEntropyLoss()(real_cat, real_categories.argmax(dim=1))

            # 假样本判别
            fake_input = torch.cat([content_images, fake_images], dim=1)
            fake_pred = self.discriminator(fake_input)
            fake_adv_loss = nn.MSELoss()(fake_pred, torch.zeros_like(fake_pred))

            # 总损失计算
            d_loss = self.Ladv_d_penalty * (torch.mean(torch.relu(1.0 - real_pred)) + torch.mean(torch.relu(1.0 + fake_pred)))


        # 反向传播
        self.scaler.scale(d_loss).backward()
        self.scaler.step(self.d_optim)
        self.scaler.update()

        return d_loss.item(), real_adv_loss.item(), fake_adv_loss.item()

    def _train_generator(self, rc, rs, content_images, style_images, ground_truth_images):
        self.g_optim.zero_grad()

        # --- 新增 ---: 提取目标风格向量，作为后续比较的“黄金标准”
        # 我们在这里计算它，并使用 .detach()，因为它不应参与生成器的梯度计算
        target_style_vector = self.generator.style_encoder(style_images).detach()

        # 混合精度上下文
        with autocast(device_type='cuda' if self.device.type == 'cuda' else 'cpu',
                      enabled=(self.device.type == 'cuda')):
            # 生成图像
            fake_images_for_g = self.generator(rc, rs, content_images, style_images)

            # 编码特征一致性损失
            encode_real = self.generator.encoder(content_images)
            encode_fake = self.generator.encoder(fake_images_for_g)
            const_loss = nn.MSELoss()(encode_real[-1], encode_fake[-1])

            # 像素级重建损失
            pixel_loss = nn.L1Loss()(fake_images_for_g, ground_truth_images)

            # 对抗损失（使用LSGAN的MSE损失）
            fake_input_for_g = torch.cat([content_images, fake_images_for_g], dim=1)
            pred_g = self.discriminator(fake_input_for_g)
            # adv_loss = nn.MSELoss()(pred_g, torch.ones_like(pred_g)) # LSGAN
            adv_loss = -torch.mean(pred_g)   # 谱归一化之后对抗损失

            # 分类损失
            # cat_loss = nn.CrossEntropyLoss()(cat_g, style_images.argmax(dim=1))
            # 风格一致性损失 (Style Consistency Loss)
            # 1. 将生成图像的范围从 [-1, 1] 转换到 [0, 1]
            fake_images_0_1 = (fake_images_for_g + 1.0) / 2.0
            # 3. 提取生成图像的风格向量
            output_style_vector = self.generator.style_encoder(fake_images_0_1)
            # 4. 计算与目标风格向量的 L1 距离
            style_loss = nn.L1Loss()(output_style_vector, target_style_vector)

            # 总变差损失
            tv_loss = (torch.sum((fake_images_for_g[:, :, 1:, :] - fake_images_for_g[:, :, :-1, :]) ** 2) +
                       torch.sum((fake_images_for_g[:, :, :, 1:] - fake_images_for_g[:, :, :, :-1]) ** 2))

            # print(
            #     f"\tpixel_loss: {pixel_loss.item():.4f}\tconst_loss: {const_loss.item():.8f}\tadv_loss: {adv_loss.item():.6f}\tstyle_loss: {style_loss.item():.6f}")

            # 总损失计算
            g_loss = (self.L1_penalty * pixel_loss +
                      self.Lconst_penalty * const_loss +
                      self.Ladv_g_penalty * adv_loss +
                      self.Lstyle_penalty * style_loss +  # --- 修改点 ---
                      self.Ltv_penalty * tv_loss)

        # 反向传播
        self.scaler.scale(g_loss).backward()
        self.scaler.step(self.g_optim)
        self.scaler.update()

        return g_loss.item(), pixel_loss.item(), const_loss.item(), adv_loss.item(), style_loss.item(), tv_loss.item()
