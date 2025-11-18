"""
Seq2Seq训练脚本 - 对应TensorFlow版本
使用方法: python train_seq2seq.py
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

from model.seq2seq_trainer import Seq2SeqTrainer
from model.seq2seq_structure import Seq2SeqModel


def load_metadata(metadata_path):
    """加载预处理的metadata"""
    npzfile = np.load(metadata_path, allow_pickle=True)

    train_action = npzfile['arr_0']
    train_script = npzfile['arr_1']
    train_length = npzfile['arr_2']
    sentence_steps = int(npzfile['arr_3'])

    return train_action, train_script, train_length, sentence_steps


def main():
    # ==================== 配置 ====================
    # GPU设置
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 数据路径
    metadata_path = './data/metadata.npz'
    mean_pose_path = './data/mean_pose.mat'
    model_dir = './seq2seq_model'

    # ==================== 加载数据 ====================
    print("\n加载数据...")
    train_action, train_script, train_length, sentence_steps = load_metadata(metadata_path)

    # 加载初始pose
    init_pose = scio.loadmat(mean_pose_path)['mean_vector']

    num_data_train = train_action.shape[0]

    # 自动从数据中推断动作维度
    dim_action = train_action.shape[1]  # [num_data, dim_action, action_steps]

    print(f"训练数据: {num_data_train} 样本")
    print(f"动作形状: {train_action.shape}")
    print(f"动作维度: {dim_action}维 (HumanML3D)")
    print(f"文本形状: {train_script.shape}")
    print(f"最大句子长度: {sentence_steps}")

    # ==================== 超参数 ====================
    # 对应TensorFlow训练代码中的参数
    dim_sentence = 300
    dim_char_enc = 300
    dim_gen = 300
    batch_size = 32
    dim_random = 10
    action_steps = 32

    # 训练参数
    max_epoch = 500
    save_stride = 5
    learning_rate = 0.00005

    # 恢复训练设置
    restore = 0
    restore_path = ''
    restore_step = 0

    print(f"\n超参数:")
    print(f"  dim_action: {dim_action} (从数据自动推断)")
    print(f"  dim_sentence: {dim_sentence}")
    print(f"  dim_char_enc: {dim_char_enc}")
    print(f"  dim_gen: {dim_gen}")
    print(f"  batch_size: {batch_size}")
    print(f"  dim_random: {dim_random}")
    print(f"  action_steps: {action_steps}")
    print(f"  max_epoch: {max_epoch}")
    print(f"  learning_rate: {learning_rate}")

    # ==================== 创建模型 ====================
    print("\n创建模型...")
    model = Seq2SeqModel(
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_action=dim_action,  # 从数据自动推断的动作维度
        dim_random=dim_random
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # ==================== 创建训练器 ====================
    print("\n创建训练器...")
    train_module = Seq2SeqTrainer(
        model=model,
        train_script=train_script,
        train_script_len=train_length,
        train_action=train_action,
        init_pose=init_pose,
        num_data=num_data_train,
        batch_size=batch_size,
        model_dir=model_dir,
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
        learning_rate=learning_rate,
        device=device
    )

    # ==================== 开始训练 ====================
    print("\n" + "="*60)
    print("开始训练 Seq2Seq 模型")
    print("="*60)

    train_module.train()

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60)
    print(f"模型保存在: {model_dir}")


if __name__ == "__main__":
    main()