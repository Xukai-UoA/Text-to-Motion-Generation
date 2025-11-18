import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os


class Seq2SeqTrainer:
    """
    Seq2Seq训练器 - 严格对应TensorFlow版本
    只训练seq2seq模型，不包含GAN
    """

    def __init__(self, model, train_script, train_script_len, train_action, init_pose,
                 num_data, batch_size, model_dir, sentence_steps, action_steps,
                 dim_sentence, dim_char_enc, dim_gen, dim_random,
                 restore=0, restore_path='', restore_step=0,
                 max_epoch=500, save_stride=5, learning_rate=0.00005,
                 device='cuda'):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)
        self.train_script = train_script  # [num_data, dim_sentence, sentence_steps]
        self.train_script_len = train_script_len
        self.train_action = train_action  # [num_data, dim_action, action_steps]
        self.init_pose = init_pose
        self.num_data = num_data
        self.batch_size = batch_size

        # 准备batch初始pose
        self.batch_init = np.transpose(np.tile(self.init_pose, (1, batch_size)), [1, 0])

        self.num_batch = num_data // batch_size
        self.model_dir = model_dir
        self.sentence_steps = sentence_steps
        self.action_steps = action_steps
        self.dim_sentence = dim_sentence
        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_random = dim_random

        self.restore = restore
        self.restore_path = restore_path
        self.restore_step = restore_step

        self.max_epoch = max_epoch
        self.save_stride = save_stride
        self.learning_rate = learning_rate

        # 创建模型目录
        os.makedirs(self.model_dir, exist_ok=True)

        # 设置optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train(self):
        """主训练循环 - 对应TensorFlow版本的train()方法"""

        # 如果需要恢复训练
        if self.restore == 1:
            checkpoint = torch.load(self.restore_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f'Restored from {self.restore_path}')

        # 转换numpy数组为torch tensors
        train_script_tensor = torch.FloatTensor(self.train_script).to(self.device)
        train_script_len_tensor = torch.LongTensor(self.train_script_len)
        train_action_tensor = torch.FloatTensor(self.train_action).to(self.device)

        for epoch in range(self.max_epoch):
            # Shuffle数据
            batch_shuffle = list(range(self.num_data))
            random.shuffle(batch_shuffle)

            for i in range(self.num_batch):
                # 获取batch索引
                batch_idx = batch_shuffle[i * self.batch_size:(i + 1) * self.batch_size]

                # 准备batch数据
                # 注意: 转置以匹配期望的输入格式 [batch, steps, dim]
                script_batch = train_script_tensor[batch_idx].transpose(1, 2)  # [batch, sentence_steps, dim_sentence]
                length_batch = train_script_len_tensor[batch_idx]
                action_batch = train_action_tensor[batch_idx].transpose(1, 2)  # [batch, action_steps, dim_action]

                # 准备输入
                curr_action_init = torch.FloatTensor(self.batch_init).to(self.device)
                curr_char_init = torch.zeros(self.batch_size, self.dim_sentence).to(self.device)

                # 使用零随机噪声（对应TensorFlow版本）
                curr_random_c2a = torch.zeros(self.batch_size, self.sentence_steps, self.dim_random).to(self.device)
                curr_random_a2c = torch.zeros(self.batch_size, self.action_steps, self.dim_random).to(self.device)

                # 前向传播
                self.optimizer.zero_grad()

                # 1. 编码文本
                char_enc_out = self.model.char_encoder(script_batch, length_batch)

                # 2. 从文本生成动作
                action_gen_out, action_enc_out = self.model.char2action(
                    char_enc_out, curr_action_init, curr_random_c2a, self.batch_size
                )

                # 3. 从动作重建文本
                # 关键: 使用action_enc_out（char2action decoder的隐藏状态），不是重新编码action!
                char_recon_out = self.model.action2char(
                    action_enc_out, curr_char_init, curr_random_a2c, self.batch_size
                )

                # 4. 计算损失
                total_loss, action_loss, char_loss = self.model.seq2seq_loss(
                    action_gen_out, action_batch, char_recon_out, script_batch
                )

                # 反向传播
                total_loss.backward()

                # 梯度裁剪（可选但推荐）
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                # 更新权重
                self.optimizer.step()

                # 打印进度（对应TensorFlow版本的print频率）
                if i % 100 == 0:
                    print(f'current epoch : {epoch + self.restore_step}, current loss : {total_loss.item():.6f}, '
                          f'{action_loss.item():.6f}, {char_loss.item():.6f}')

            # 保存checkpoint
            if (epoch + 1) % self.save_stride == 0:
                checkpoint_path = os.path.join(
                    self.model_dir,
                    f'model_epoch_{epoch + 1 + self.restore_step}.pth'
                )
                torch.save({
                    'epoch': epoch + 1 + self.restore_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': total_loss.item(),
                }, checkpoint_path)
                print(f'Model saved in file : {checkpoint_path}')


if __name__ == "__main__":
    # 测试训练器
    from seq2seq_structure import Seq2SeqModel

    # 超参数
    sentence_steps = 30
    action_steps = 32
    dim_sentence = 300
    dim_char_enc = 256
    dim_gen = 256
    dim_action = 263  # HumanML3D标准版维度
    dim_random = 16
    batch_size = 32

    # 生成测试数据
    num_data = 1000
    train_script = np.random.randn(num_data, dim_sentence, sentence_steps).astype(np.float32)
    train_script_len = np.random.randint(10, sentence_steps, size=(num_data,))
    train_action = np.random.randn(num_data, dim_action, action_steps).astype(np.float32)
    init_pose = np.random.randn(dim_action, 1).astype(np.float32)

    # 创建模型
    model = Seq2SeqModel(
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_action=dim_action,
        dim_random=dim_random
    )

    # 创建训练器
    trainer = Seq2SeqTrainer(
        model=model,
        train_script=train_script,
        train_script_len=train_script_len,
        train_action=train_action,
        init_pose=init_pose,
        num_data=num_data,
        batch_size=batch_size,
        model_dir='./checkpoints',
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random,
        max_epoch=10,
        save_stride=2,
        learning_rate=0.00005
    )

    # 开始训练
    print("开始训练...")
    trainer.train()
    print("训练完成!")