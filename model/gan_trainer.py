"""
PyTorch implementation of GAN trainer from Text2Action
Corresponds to trainer_GAN.py from TensorFlow version
"""

import numpy as np
import torch
import torch.optim as optim
import random
import os


class GANTrainer:
    """
    GAN训练器 - 对应TensorFlow版本的GAN_trainer类
    """
    def __init__(self, gan_model, train_script, train_script_len, train_action, init_pose,
                 num_data, batch_size, gan_model_dir, seq2seq_model_dir, dis_model_dir,
                 sentence_steps, action_steps, dim_sentence, dim_char_enc, dim_gen, dim_random,
                 dim_action=263, restore=0, restore_path='', restore_step=0,
                 max_epoch=500, save_stride=5, gen_learning_rate=0.000002, dis_learning_rate=0.000002,
                 device='cuda'):

        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.action_steps = action_steps
        self.dim_action = dim_action  # 支持可配置的动作维度，默认263（HumanML3D标准版）

        self.gan_model = gan_model.to(self.device)

        self.train_script = train_script  # [num_data, dim_sentence, sentence_steps]
        self.train_script_len = train_script_len
        self.train_action = train_action  # [num_data, dim_action, action_steps]
        self.init_pose = init_pose

        self.num_data = num_data
        self.batch_size = batch_size

        # 准备batch初始pose
        self.batch_init = np.transpose(np.tile(self.init_pose, (1, batch_size)), [1, 0])

        self.num_batch = num_data // batch_size
        self.gan_model_dir = gan_model_dir
        self.seq2seq_model_dir = seq2seq_model_dir
        self.dis_model_dir = dis_model_dir

        self.sentence_steps = sentence_steps
        self.dim_sentence = dim_sentence
        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_gen_inp = self.dim_action
        self.dim_random = dim_random

        self.restore = restore
        self.restore_path = restore_path
        self.restore_step = restore_step

        self.max_epoch = max_epoch
        self.save_stride = save_stride
        self.gen_learning_rate = gen_learning_rate
        self.dis_learning_rate = dis_learning_rate

        # 创建模型目录
        os.makedirs(self.gan_model_dir, exist_ok=True)

        # 分离generator和discriminator的参数
        # Generator包含: char_encoder + char2action部分
        self.gen_params = []
        self.dis_params = []

        for name, param in self.gan_model.named_parameters():
            if 'char_encoder' in name or 'char2action' in name:
                self.gen_params.append(param)
            elif 'discriminator' in name:
                self.dis_params.append(param)

        # 设置optimizers - 对应TensorFlow的AdamOptimizer
        self.gen_optimizer = optim.Adam(self.gen_params, lr=gen_learning_rate)
        self.dis_optimizer = optim.Adam(self.dis_params, lr=dis_learning_rate)

    def load_seq2seq_pretrained(self, seq2seq_path):
        """
        加载预训练的seq2seq模型权重
        对应TensorFlow版本中的seq2seq_saver.restore()
        """
        print(f"\n加载预训练seq2seq模型: {seq2seq_path}")

        try:
            checkpoint = torch.load(seq2seq_path, map_location=self.device)

            # 加载模型状态
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 只加载generator相关的权重（char_encoder和char2action）
            model_state = self.gan_model.state_dict()
            pretrained_dict = {}

            for k, v in state_dict.items():
                # 将seq2seq的权重映射到GAN模型
                # seq2seq中的char_encoder对应GAN中的char_encoder
                # seq2seq中的char2action对应GAN中的char2action
                if 'char_encoder' in k or 'char2action' in k:
                    pretrained_dict[k] = v

            # 更新模型权重
            model_state.update(pretrained_dict)
            self.gan_model.load_state_dict(model_state, strict=False)

            print(f"✓ 成功加载 {len(pretrained_dict)} 个预训练权重")

        except Exception as e:
            print(f"警告: 无法加载预训练模型 - {e}")
            print("将使用随机初始化的权重开始训练")

    def train(self):
        """
        主训练循环 - 对应TensorFlow版本的train()方法
        """

        # 加载预训练的seq2seq权重
        if self.seq2seq_model_dir and os.path.exists(self.seq2seq_model_dir):
            self.load_seq2seq_pretrained(self.seq2seq_model_dir)

        # 如果需要恢复GAN训练
        if self.restore == 1 and self.restore_path:
            checkpoint = torch.load(self.restore_path, map_location=self.device)
            self.gan_model.load_state_dict(checkpoint['model_state_dict'])
            if 'gen_optimizer_state_dict' in checkpoint:
                self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer_state_dict'])
            if 'dis_optimizer_state_dict' in checkpoint:
                self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer_state_dict'])
            print(f'Restored {self.restore_path}')

        # 转换numpy数组为torch tensors
        train_script_tensor = torch.FloatTensor(self.train_script).to(self.device)
        train_script_len_tensor = torch.LongTensor(self.train_script_len)
        train_action_tensor = torch.FloatTensor(self.train_action).to(self.device)

        # 训练循环
        for epoch in range(self.max_epoch - self.restore_step):
            # Shuffle数据
            batch_shuffle = list(range(self.num_data))
            random.shuffle(batch_shuffle)

            for i in range(self.num_batch):
                # 获取batch索引
                batch_idx = batch_shuffle[i * self.batch_size:(i + 1) * self.batch_size]

                # 准备batch数据
                # 转置以匹配期望的输入格式 [batch, steps, dim]
                script_batch = train_script_tensor[batch_idx].transpose(1, 2)  # [batch, sentence_steps, dim_sentence]
                length_batch = train_script_len_tensor[batch_idx]
                action_batch = train_action_tensor[batch_idx].transpose(1, 2)  # [batch, action_steps, dim_action]

                # 准备输入
                curr_init_input = torch.FloatTensor(self.batch_init).to(self.device)

                # 生成随机噪声 - 对应TensorFlow的np.random.normal
                curr_random = torch.randn(
                    self.batch_size, self.sentence_steps, self.dim_random,
                    device=self.device
                )

                # ==================== 前向传播 ====================
                # 1. 编码文本
                with torch.no_grad():  # char_encoder不需要梯度（已预训练）
                    char_enc_out = self.gan_model.char_encoder(script_batch, length_batch)

                # 2. Generator生成假动作
                fake_actions_list = self.gan_model.char2action(
                    char_enc_out, curr_init_input, curr_random, self.batch_size
                )

                # 转换为tensor
                fake_actions_tensor = torch.stack(fake_actions_list, dim=1)  # [batch, action_steps, dim_action]

                # ==================== 训练Discriminator ====================
                self.dis_optimizer.zero_grad()

                # 判别真实动作
                label_real = self.gan_model.discriminator(
                    char_enc_out, action_batch, self.batch_size
                )

                # 判别生成动作（detach以避免通过generator反向传播）
                label_fake_for_dis = self.gan_model.discriminator(
                    char_enc_out, fake_actions_tensor.detach(), self.batch_size
                )

                # 计算discriminator loss
                dis_loss = self.gan_model.dis_loss(label_real, label_fake_for_dis)

                # 反向传播和更新
                dis_loss.backward()
                self.dis_optimizer.step()

                # ==================== 训练Generator ====================
                self.gen_optimizer.zero_grad()

                # 重新生成动作（需要梯度）
                fake_actions_list_for_gen = self.gan_model.char2action(
                    char_enc_out, curr_init_input, curr_random, self.batch_size
                )
                fake_actions_tensor_for_gen = torch.stack(fake_actions_list_for_gen, dim=1)

                # 判别生成动作
                label_fake_for_gen = self.gan_model.discriminator(
                    char_enc_out, fake_actions_tensor_for_gen, self.batch_size
                )

                # 计算generator loss
                gen_loss = self.gan_model.gen_loss(label_fake_for_gen)

                # 反向传播和更新
                gen_loss.backward()
                self.gen_optimizer.step()

                # ==================== 打印进度 ====================
                if i % 100 == 0:
                    print(f'{epoch + self.restore_step}: batch_gen_loss : {gen_loss.item():.6f}, '
                          f'dis_loss :{dis_loss.item():.6f}')

            # ==================== 保存checkpoint ====================
            if (epoch + 1 + self.restore_step) % self.save_stride == 0:
                checkpoint_path = os.path.join(
                    self.gan_model_dir,
                    f'model_epoch_{epoch + 1 + self.restore_step}.pth'
                )
                torch.save({
                    'epoch': epoch + 1 + self.restore_step,
                    'model_state_dict': self.gan_model.state_dict(),
                    'gen_optimizer_state_dict': self.gen_optimizer.state_dict(),
                    'dis_optimizer_state_dict': self.dis_optimizer.state_dict(),
                    'gen_loss': gen_loss.item(),
                    'dis_loss': dis_loss.item(),
                }, checkpoint_path)
                print(f'Model saved in file : {checkpoint_path}')


if __name__ == "__main__":
    # 测试训练器
    from gan_structure import GANModel

    # 超参数
    sentence_steps = 30
    action_steps = 32
    dim_sentence = 300
    dim_char_enc = 300
    dim_gen = 300
    dim_dis = 300
    dim_action = 263  # HumanML3D标准版维度
    dim_random = 10
    batch_size = 32

    # 生成测试数据
    num_data = 1000
    train_script = np.random.randn(num_data, dim_sentence, sentence_steps).astype(np.float32)
    train_script_len = np.random.randint(10, sentence_steps, size=(num_data,))
    train_action = np.random.randn(num_data, dim_action, action_steps).astype(np.float32)
    init_pose = np.random.randn(dim_action, 1).astype(np.float32)

    # 创建模型
    model = GANModel(
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_dis=dim_dis,
        dim_action=dim_action,
        dim_random=dim_random
    )

    # 创建训练器
    trainer = GANTrainer(
        gan_model=model,
        train_script=train_script,
        train_script_len=train_script_len,
        train_action=train_action,
        init_pose=init_pose,
        num_data=num_data,
        batch_size=batch_size,
        gan_model_dir='./gan_checkpoints',
        seq2seq_model_dir='',  # 如果有预训练模型，填入路径
        dis_model_dir='',
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_random=dim_random,
        dim_action=dim_action,
        max_epoch=5,  # 测试用小epoch
        save_stride=2,
        gen_learning_rate=0.000002,
        dis_learning_rate=0.000002
    )

    # 开始训练
    print("开始训练...")
    trainer.train()
    print("训练完成!")