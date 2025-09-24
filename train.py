import os
import datetime
import numpy as np

import torch
from torch.utils.data import DataLoader, Subset
from model_torch import CalliGAN
from dataset import CalliGANDataset
from torch.utils.tensorboard import SummaryWriter

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

if __name__ == "__main__":
    train_image_dir = '../data/Calliformer-datasets/images/train'
    json_path = '../data/Calliformer-datasets/components/processed_datasets.json'
    output_dir = "../data/training_output"

    LOAD_MODEL = True  # 设置为True来加载模型
    GEN_MODEL_PATH = "../data/training_output/20250831_111239/transformer_generator_epoch_60_64_spc_6000.pth"
    DISC_MODEL_PATH = "../data/training_output/20250831_111239/transformer_discriminator_epoch_60_64_spc_6000.pth"

    # 初始化实验时间戳（统一使用此时间）
    exp_time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.datetime.now()

    # 创建带时间戳的输出目录
    output_dir = os.path.join(output_dir, exp_time_str)
    sample_dir = os.path.join(output_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    # 初始化 TensorBoard 记录器
    writer = SummaryWriter(log_dir)

    # 超参数配置
    batch_size = 64
    samples_per_class = 6000
    epochs = 100
    initial_epoch = 60
    # SAVE_INTERVAL = 10
    vis_interval = 100
    log_interval = 100

    # 初始化训练器
    trainer = CalliGAN(gen_train_steps=1, cns_coder='transformer')

    dataset = CalliGANDataset(train_image_dir, json_path, samples_per_class=samples_per_class)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # 加速GPU传输
        persistent_workers=True,
        drop_last=True  # 丢弃最后不完整的batch
    )

    # ------------------------ 开始训练 ------------------------
    print(f"Using device: {trainer.device}")
    print(f"Number of training batches: {len(train_loader)}")

    # 加载预训练模型
    if LOAD_MODEL:
        try:
            print(f"Loading Generator from: {GEN_MODEL_PATH}")
            trainer.generator.load_state_dict(torch.load(GEN_MODEL_PATH, map_location=trainer.device))

            print(f"Loading Discriminator from: {DISC_MODEL_PATH}")
            trainer.discriminator.load_state_dict(torch.load(DISC_MODEL_PATH, map_location=trainer.device))
            print(f"Models loaded successfully. Resuming training from epoch {initial_epoch}.")
        except FileNotFoundError:
            print(f"Warning: Model files not found at specified paths. Starting training from scratch.")
            initial_epoch = 0  # 如果找不到文件，就从头开始
        except Exception as e:
            print(f"An error occurred while loading models: {e}. Starting training from scratch.")
            initial_epoch = 0  # 如果加载出错，也从头开始
    else:
        print("Starting training from scratch.")

    # 训练循环
    best_loss = float('inf')
    patience = 5
    no_improve = 0

    global_step = 0  # 新增全局step计数器

    try:
        for epoch in range(initial_epoch, epochs):
            # 训练阶段
            trainer.generator.train()
            trainer.discriminator.train()
            batch_count = 0

            for batch_idx, batch in enumerate(train_loader):
                (components, structural, sources, categories), targets = batch
                metrics = trainer.train_step((components, structural, sources, categories), targets)
                batch_count += 1
                global_step += 1

                if batch_count % log_interval == 0:
                    print(f"\t[Epoch {epoch + 1}][Batch {batch_count}/{len(train_loader)}]"
                          f"\tLoss_D: {metrics['D_total']:.4f}\tLoss_G: {metrics['G_total']:.4f}")

                    # --- TensorBoard 日志记录 ---
                    # 总损失
                    writer.add_scalar('Loss/Total/Generator', metrics['G_total'], global_step)
                    writer.add_scalar('Loss/Total/Discriminator', metrics['D_total'], global_step)

                    # 记录原始损失 (Raw)
                    for key in ['raw_pixel', 'raw_const', 'raw_adv_g', 'raw_style', 'raw_tv']:
                        writer.add_scalar(f'Loss/Raw/G/{key.split("_")[1]}', metrics[key], global_step)
                    writer.add_scalar('Loss/Raw/D/real', metrics['raw_adv_d_real'], global_step)
                    writer.add_scalar('Loss/Raw/D/fake', metrics['raw_adv_d_fake'], global_step)
                    # 记录加权损失 (Weighted)
                    for key in ['weighted_pixel', 'weighted_const', 'weighted_adv_g', 'weighted_style', 'weighted_tv']:
                        writer.add_scalar(f'Loss/Weighted/G/{key.split("_")[1]}', metrics[key], global_step)

                # ========== 新增样本可视化 ==========
                if (batch_idx + 1) % vis_interval == 0:
                    trainer.save_comparison_figures(
                        batch=batch,
                        output_dir=output_dir,
                        step=global_step,
                        n_samples=3  # 保存前3个样本
                    )

                if global_step % 5000 == 0:  # 每5000步保存一次
                    trainer.save_checkpoint(global_step=global_step, directory=checkpoint_dir)

            # 学习率衰减
            if (epoch + 1) >= 40 and (epoch + 1) % 20 == 0:
                for param_group in trainer.g_optim.param_groups:
                    param_group['lr'] = max(param_group['lr'] / 2, 0.000001)
                for param_group in trainer.d_optim.param_groups:
                    param_group['lr'] = max(param_group['lr'] / 2, 0.000001)

        # ====================================================================
        # 3. 在每个 Epoch 结束后，进行验证和低频记录 (学术最佳实践)
        # ====================================================================
        # print(f"--- Epoch {global_step + 1} finished. Running validation... ---")

        # 切换到评估模式
        # trainer.generator.eval()

    finally:
        # 训练结束处理
        total_time = (datetime.datetime.now() - start_time).total_seconds()
        print(f"Total training time: {total_time:.2f}")
        # trainer.save_training_report(output_dir, exp_time_str, total_time)

    # 保存最终模型
    torch.save(trainer.generator.state_dict(), os.path.join(output_dir, f"{trainer.cns_coder}_generator_epoch_{epochs}_{batch_size}_spc_{samples_per_class}.pth"))
    torch.save(trainer.discriminator.state_dict(), os.path.join(output_dir, f"{trainer.cns_coder}_discriminator_epoch_{epochs}_{batch_size}_spc_{samples_per_class}.pth"))

    # 绘制训练曲线
    trainer.plot_training_history(output_dir)

    print("Training completed!")
    writer.close()
