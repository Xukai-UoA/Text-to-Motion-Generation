"""
GAN训练脚本 - PyTorch版本
对应train_GAN.ipynb
使用方法: python train_gan.py
"""

import os
import sys
import numpy as np
import scipy.io as scio
import torch

# 添加模型路径
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from model.gan_trainer import GANTrainer
from model.gan_structure import GANModel
from utils.my_functions import load_metadata


def main():
    # ==================== 配置 ====================
    # GPU设置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 数据路径
    metadata_path = './data/metadata.npz'
    mean_pose_path = './data/mean_pose.mat'

    # 模型保存路径
    seq2seq_model_dir = './seq2seq_model/model_epoch_500.pth'  # 预训练的seq2seq模型路径
    gan_model_dir = './gan_model'
    dis_model_dir = ''  # 如果有单独的discriminator预训练模型，填入路径

    # ==================== 加载数据 ====================
    print("\n加载数据...")
    train_action, train_script, train_length, sentence_steps = load_metadata(metadata_path)

    # 加载初始pose
    init_pose = scio.loadmat(mean_pose_path)['mean_vector']

    num_data_train = train_action.shape[0]

    print(f"训练数据: {num_data_train} 样本")
    print(f"动作形状: {train_action.shape}")
    print(f"文本形状: {train_script.shape}")
    print(f"最大句子长度: {sentence_steps}")

    # ==================== 超参数 ====================
    # 对应TensorFlow训练代码中的参数
    dim_sentence = 300
    dim_char_enc = 300
    dim_gen = 300
    dim_dis = 300
    batch_size = 32
    dim_random = 10
    action_steps = 32

    # 训练参数
    max_epoch = 500
    save_stride = 5
    gen_learning_rate = 0.000002  # 2e-6
    dis_learning_rate = 0.000002  # 2e-6

    # 恢复训练设置
    restore = 0  # 设为1表示从checkpoint恢复训练
    restore_path = ''  # 如果restore=1，填入checkpoint路径，例如'./gan_model/model_epoch_200.pth'
    restore_step = 0  # 如果从checkpoint恢复，填入之前训练的epoch数

    print(f"\n超参数:")
    print(f"  dim_sentence: {dim_sentence}")
    print(f"  dim_char_enc: {dim_char_enc}")
    print(f"  dim_gen: {dim_gen}")
    print(f"  dim_dis: {dim_dis}")
    print(f"  batch_size: {batch_size}")
    print(f"  dim_random: {dim_random}")
    print(f"  action_steps: {action_steps}")
    print(f"  max_epoch: {max_epoch}")
    print(f"  gen_learning_rate: {gen_learning_rate}")
    print(f"  dis_learning_rate: {dis_learning_rate}")

    # ==================== 创建模型 ====================
    print("\n创建GAN模型...")
    model = GANModel(
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_dis=dim_dis,
        dim_random=dim_random  # 添加dim_random参数
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 统计generator和discriminator的参数
    gen_params = sum(p.numel() for name, p in model.named_parameters()
                     if 'char_encoder' in name or 'char2action' in name)
    dis_params = sum(p.numel() for name, p in model.named_parameters()
                     if 'discriminator' in name)
    print(f"  - Generator参数: {gen_params:,}")
    print(f"  - Discriminator参数: {dis_params:,}")

    # ==================== 创建训练器 ====================
    print("\n创建训练器...")
    train_module = GANTrainer(
        gan_model=model,
        train_script=train_script,
        train_script_len=train_length,
        train_action=train_action,
        init_pose=init_pose,
        num_data=num_data_train,
        batch_size=batch_size,
        gan_model_dir=gan_model_dir,
        seq2seq_model_dir=seq2seq_model_dir,
        dis_model_dir=dis_model_dir,
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random,
        restore=restore,
        restore_path=restore_path,
        restore_step=restore_step,
        max_epoch=max_epoch,
        save_stride=save_stride,
        gen_learning_rate=gen_learning_rate,
        dis_learning_rate=dis_learning_rate,
        device=device
    )

    # ==================== 开始训练 ====================
    print("\n" + "="*60)
    print("开始训练 GAN 模型")
    print("="*60)
    print("\n训练说明:")
    print("  - 首先会加载预训练的seq2seq模型权重（char_encoder和char2action部分）")
    print("  - Discriminator使用随机初始化")
    print("  - 训练过程中会交替更新Generator和Discriminator")
    print("  - 每个epoch结束后会保存checkpoint")
    print("\n提示:")
    print("  - Generator loss应该逐渐下降（目标是欺骗discriminator）")
    print("  - Discriminator loss应该在1.0-1.5之间波动（表示平衡的对抗）")
    print("  - 如果discriminator loss过低(<0.5)，说明discriminator过强")
    print("  - 如果generator loss过高(>2.0)，说明generator太弱")
    print("")

    try:
        train_module.train()
    except KeyboardInterrupt:
        print("\n\n训练被用户中断!")
        print("当前进度已保存，可以使用restore参数继续训练")

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"模型保存在: {gan_model_dir}")
    print("\n使用训练好的模型:")
    print("  1. 加载最佳checkpoint")
    print("  2. 使用model.char2action()生成动作序列")
    print("  3. 可以通过改变random_noise生成不同的动作变体")


if __name__ == "__main__":
    main()