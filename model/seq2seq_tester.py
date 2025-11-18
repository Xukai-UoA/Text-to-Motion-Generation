"""
PyTorch version of seq2seq_tester.py
用于测试训练好的seq2seq模型
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Seq2SeqTester:
    """
    Seq2Seq测试器 - 对应TensorFlow版本
    用于加载训练好的模型并生成动作序列
    """

    def __init__(self, model, init_pose, model_path,
                 sentence_steps, action_steps, dim_sentence,
                 dim_char_enc, dim_gen, dim_random,
                 device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.action_steps = action_steps
        self.dim_action = 24
        self.num_data = 1  # 测试时batch_size=1

        self.init_pose = init_pose
        self.batch_init = np.transpose(np.tile(self.init_pose, (1, self.num_data)), [1, 0])

        self.model_path = model_path
        self.sentence_steps = sentence_steps
        self.dim_sentence = dim_sentence
        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_random = dim_random

        # 加载训练好的模型
        self._load_model()

    def _load_model(self):
        """加载训练好的模型权重"""
        print(f"Loading model from {self.model_path}...")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()  # 设置为评估模式
        print(f"Model loaded successfully! (Epoch: {checkpoint.get('epoch', 'unknown')})")

    def test(self, test_script, test_script_len):
        """
        测试函数 - 从文本生成动作序列

        Args:
            test_script: [1, dim_sentence, sentence_steps] 输入文本embedding
            test_script_len: [1] 文本长度

        Returns:
            test_esti: [1, dim_action, action_steps] 生成的动作序列
        """
        with torch.no_grad():  # 测试时不需要梯度
            # 转换为torch tensor
            script_tensor = torch.FloatTensor(test_script).to(self.device)
            length_tensor = torch.LongTensor(test_script_len)

            # 转置以匹配模型输入格式 [batch, sentence_steps, dim_sentence]
            script_batch = script_tensor.transpose(1, 2)

            # 准备初始输入
            curr_init_input = torch.FloatTensor(self.batch_init).to(self.device)

            # 使用零随机噪声（与训练时一致）
            curr_random = torch.zeros(self.num_data, self.sentence_steps,
                                      self.dim_random).to(self.device)

            # 1. 编码文本
            char_enc_out = self.model.char_encoder(script_batch, length_tensor)

            # 2. 从文本生成动作
            action_gen_out, action_enc_out = self.model.char2action(
                char_enc_out, curr_init_input, curr_random, self.num_data
            )

            # 转换回numpy并转置为 [1, dim_action, action_steps]
            test_esti = action_gen_out.cpu().numpy().transpose(0, 2, 1)

        return test_esti


def visualize_action_sequence(action_seq, title="Generated Action", save_path=None):
    """
    可视化动作序列（3D骨架动画）

    Args:
        action_seq: [dim_action, action_steps] 或 [1, dim_action, action_steps]
        title: 图表标题
        save_path: 保存路径（可选）
    """
    if action_seq.ndim == 3:
        action_seq = action_seq[0]  # [dim_action, action_steps]

    action_steps = action_seq.shape[1]

    # 选择几个关键帧显示
    num_frames = min(8, action_steps)
    frame_indices = np.linspace(0, action_steps - 1, num_frames, dtype=int)

    fig = plt.figure(figsize=(16, 6))

    # 定义骨架连接关系（根据论文图5）
    # 颈部位置: action_seq[0:3, t]
    # 关节向量: action_seq[3:24, t] 分为7个关节，每个3维
    connections = [
        (0, 1), (0, 2),  # 颈部到左右肩
        (1, 3), (2, 4),  # 肩到肘
        (3, 5), (4, 6),  # 肘到手腕
        (0, 7)  # 颈部到头部（如果有）
    ]

    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(2, 4, i + 1, projection='3d')

        # 提取当前帧的pose
        pose = action_seq[:, frame_idx]

        # 颈部位置
        neck_pos = pose[0:3]

        # 计算各关节位置（从颈部开始累加向量）
        joint_positions = [neck_pos]
        for j in range(7):
            joint_vec = pose[3 + j * 3:3 + (j + 1) * 3]
            # 这里简化处理，实际应根据骨架结构累加
            joint_pos = joint_positions[-1] + joint_vec * 0.3  # 缩放因子
            joint_positions.append(joint_pos)

        joint_positions = np.array(joint_positions)

        # 绘制骨架
        ax.scatter(joint_positions[:, 0], joint_positions[:, 1],
                   joint_positions[:, 2], c='red', s=50)

        # 绘制连接线
        for conn in connections:
            if conn[1] < len(joint_positions):
                ax.plot([joint_positions[conn[0], 0], joint_positions[conn[1], 0]],
                        [joint_positions[conn[0], 1], joint_positions[conn[1], 1]],
                        [joint_positions[conn[0], 2], joint_positions[conn[1], 2]],
                        'b-', linewidth=2)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Frame {frame_idx}')

        # 设置相同的坐标轴范围
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


def save_action_to_file(action_seq, save_path):
    """
    保存动作序列到文件

    Args:
        action_seq: [1, dim_action, action_steps]
        save_path: 保存路径
    """
    np.save(save_path, action_seq)
    print(f"Action sequence saved to {save_path}")


if __name__ == "__main__":
    """测试示例"""
    import sys
    import os
    from seq2seq_structure import Seq2SeqModel
    from my_functions import load_w2v, get_init_pose, load_metadata
    import scipy.io as scio

    # ==================== 配置 ====================
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")

    # 模型参数（需要与训练时一致）
    sentence_steps = 30
    action_steps = 32
    dim_sentence = 300
    dim_char_enc = 300
    dim_gen = 300
    dim_random = 10

    # 路径配置
    model_path = './seq2seq_model/model_epoch_500.pth'
    mean_pose_path = './data/mean_pose.mat'
    w2v_path = './data/GoogleNews-vectors-negative300.bin'

    # ==================== 加载模型 ====================
    print("\n创建模型...")
    model = Seq2SeqModel(
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random
    )

    # 加载初始pose
    print("\n加载初始pose...")
    init_pose = scio.loadmat(mean_pose_path)['mean_vector']

    # 创建测试器
    print("\n创建测试器...")
    tester = Seq2SeqTester(
        model=model,
        init_pose=init_pose,
        model_path=model_path,
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random,
        device=device
    )

    # ==================== 准备测试输入 ====================
    print("\n准备测试输入...")

    # 加载Word2Vec模型
    print("加载Word2Vec模型...")
    w2v_model = load_w2v(w2v_path)

    # 测试句子
    test_sentence = "a woman is dancing"
    print(f"\n测试句子: '{test_sentence}'")

    # 将句子转换为embedding
    words = test_sentence.lower().split()
    test_script = np.zeros((1, dim_sentence, sentence_steps))

    for i, word in enumerate(words):
        if i >= sentence_steps:
            break
        if word in w2v_model:
            test_script[0, :, i] = w2v_model[word]
        else:
            print(f"警告: 词 '{word}' 不在词汇表中")

    test_script_len = np.array([len(words)])

    # ==================== 运行测试 ====================
    print("\n生成动作序列...")
    generated_action = tester.test(test_script, test_script_len)

    print(f"生成的动作序列形状: {generated_action.shape}")
    print(f"动作序列统计:")
    print(f"  - Mean: {generated_action.mean():.4f}")
    print(f"  - Std: {generated_action.std():.4f}")
    print(f"  - Min: {generated_action.min():.4f}")
    print(f"  - Max: {generated_action.max():.4f}")

    # ==================== 保存和可视化 ====================
    print("\n保存结果...")
    output_dir = './test_results'
    os.makedirs(output_dir, exist_ok=True)

    # 保存到文件
    save_action_to_file(generated_action,
                        os.path.join(output_dir, 'generated_action.npy'))

    # 可视化
    print("\n可视化动作序列...")
    visualize_action_sequence(
        generated_action,
        title=f"Generated Action: '{test_sentence}'",
        save_path=os.path.join(output_dir, 'action_visualization.png')
    )

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    print(f"结果保存在: {output_dir}")