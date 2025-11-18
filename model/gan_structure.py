"""
PyTorch implementation of GAN structure from Text2Action
Corresponds to struct_GAN.py from TensorFlow version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDecoder(nn.Module):
    """
    Attention decoder for GAN - reuses the implementation from seq2seq
    """
    def __init__(self, hidden_size, attention_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.attention_size = attention_size

        # Attention mechanism weights
        self.W_a = nn.Linear(hidden_size, attention_size, bias=False)
        self.U_a = nn.Linear(attention_size, attention_size)
        self.v_a = nn.Linear(attention_size, 1, bias=False)

        # Decoder LSTM
        self.lstm_cell = nn.LSTMCell(hidden_size + attention_size, hidden_size)

    def compute_attention(self, hidden, encoder_outputs):
        """计算attention权重和context向量"""
        batch_size = hidden.size(0)
        seq_len = encoder_outputs.size(1)

        # 扩展hidden用于broadcasting
        hidden_expanded = hidden.unsqueeze(1).expand(-1, seq_len, -1)

        # 计算attention scores
        scores = self.v_a(torch.tanh(
            self.W_a(hidden_expanded) + self.U_a(encoder_outputs)
        ))

        # Softmax得到attention权重
        attention_weights = F.softmax(scores, dim=1)

        # 计算context向量
        context = torch.sum(attention_weights * encoder_outputs, dim=1)

        return context, attention_weights

    def forward(self, inputs, initial_state, attention_states, loop_function=None):
        """
        前向传播 - 关键：正确应用loop_function

        inputs: list of [batch_size, input_dim] tensors
        initial_state: (h_0, c_0) tuple
        attention_states: [batch_size, seq_len, attention_size]
        loop_function: 可选的函数，将当前输出转换为下一个输入
        """
        outputs = []
        hidden, cell = initial_state

        for i in range(len(inputs)):
            # 在第一步使用提供的输入，之后如果有loop_function就使用它
            if i == 0 or loop_function is None:
                inp = inputs[i]
            else:
                # 使用上一步的输出通过loop_function得到当前输入
                inp = loop_function(outputs[-1], i)

            # 计算attention
            context, _ = self.compute_attention(hidden, attention_states)

            # 拼接输入和context
            lstm_input = torch.cat([inp, context], dim=1)

            # LSTM step
            hidden, cell = self.lstm_cell(lstm_input, (hidden, cell))
            outputs.append(hidden)

        return outputs, (hidden, cell)


class GANModel(nn.Module):
    """
    GAN model for Text2Action
    对应TensorFlow版本的GAN_model类
    """
    def __init__(self, sentence_steps, action_steps, dim_sentence, dim_char_enc, dim_gen, dim_dis, dim_random=10):
        super(GANModel, self).__init__()

        self.action_steps = action_steps
        self.dim_action = 24
        self.stddev = 0.01

        self.sentence_steps = sentence_steps
        self.dim_sentence = dim_sentence

        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_dis = dim_dis
        self.dim_gen_inp = dim_gen
        self.dim_random = dim_random

        # Text encoder LSTM
        self.char_encoder_lstm = nn.LSTM(
            input_size=dim_sentence,
            hidden_size=dim_char_enc,
            batch_first=True
        )

        # Generator (char2action) components
        # 关键修复：attention_size需要包含random noise维度，与seq2seq模型一致
        self.char2action_attention_size = dim_char_enc + (dim_random if dim_random > 0 else 0)
        self.char2action_decoder = AttentionDecoder(dim_gen, self.char2action_attention_size)

        # Generator的输入输出转换层
        self.char2action_W_out = nn.Linear(dim_gen, self.dim_action)
        self.char2action_W_in = nn.Linear(self.dim_action, dim_gen)

        # Discriminator components
        # Discriminator不使用random noise，所以attention_size只是char_enc
        self.discriminator_attention_size = dim_char_enc
        self.discriminator_decoder = AttentionDecoder(dim_dis, self.discriminator_attention_size)

        # Discriminator的输入转换层（将action转换到dim_dis维度）
        self.discriminator_W_in = nn.Linear(self.dim_action, dim_dis)

        # Discriminator的最后输出层
        self.discriminator_W_out = nn.Linear(dim_dis, 1)

        # 初始化权重 - 对应TensorFlow的stddev=0.01
        self._init_weights()

    def _init_weights(self):
        """用小的随机值初始化权重，对应TensorFlow的stddev=0.01"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=0.01)
            elif isinstance(m, nn.LSTM) or isinstance(m, nn.LSTMCell):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.normal_(param, std=0.01)
                    elif 'bias' in name:
                        nn.init.normal_(param, std=0.01)

    def char_encoder(self, x, seq_len=None):
        """
        编码文本序列
        x: [batch_size, sentence_steps, dim_sentence]
        seq_len: [batch_size] 序列长度 (可选)
        返回: [batch_size, sentence_steps, dim_char_enc]
        """
        if seq_len is not None:
            # Pack padded sequence以提高效率
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_len.cpu(), batch_first=True, enforce_sorted=False
            )
            output, _ = self.char_encoder_lstm(x_packed)
            # Pad回固定长度
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=self.sentence_steps
            )
        else:
            output, _ = self.char_encoder_lstm(x)

        return output

    def char2action(self, char_enc, init_input, random_noise, batch_size):
        """
        Generator: 从文本编码生成动作序列

        char_enc: [batch_size, sentence_steps, dim_char_enc]
        init_input: [batch_size, dim_action] 初始pose
        random_noise: [batch_size, sentence_steps, dim_random] 随机噪声

        返回: list of [batch_size, dim_action] tensors (长度为action_steps)
        """
        device = char_enc.device

        # 准备decoder输入（全部用转换后的init_input初始化）
        dec_input_list = []
        for _ in range(self.action_steps):
            dec_input = self.char2action_W_in(init_input)
            dec_input_list.append(dec_input)

        # 准备attention states
        # 关键修复：random noise必须concat到char_enc上，这样attention会在整个拼接向量上计算
        # 这与TensorFlow版本完全一致，也与seq2seq模型训练时一致
        if random_noise.size(-1) > 0:
            # 确保random_noise的长度与char_enc匹配
            if random_noise.size(1) == char_enc.size(1):
                attn_states = torch.cat([char_enc, random_noise], dim=2)
            else:
                # 调整random_noise长度
                actual_seq_len = char_enc.size(1)
                if random_noise.size(1) > actual_seq_len:
                    random_noise_adjusted = random_noise[:, :actual_seq_len, :]
                else:
                    padding = torch.zeros(batch_size, actual_seq_len - random_noise.size(1),
                                          random_noise.size(-1), device=device)
                    random_noise_adjusted = torch.cat([random_noise, padding], dim=1)
                attn_states = torch.cat([char_enc, random_noise_adjusted], dim=2)
        else:
            attn_states = char_enc

        # 初始hidden state
        h_0 = torch.zeros(batch_size, self.dim_gen, device=device)
        c_0 = torch.zeros(batch_size, self.dim_gen, device=device)

        # 定义loop function（对应TensorFlow的loop_function）
        def loop_fn(prev_output, i):
            # prev_output是decoder的hidden state
            # 转换为action，再转回input空间
            action = self.char2action_W_out(prev_output)
            next_input = self.char2action_W_in(action)
            return next_input

        # 运行attention decoder
        outputs, _ = self.char2action_decoder(
            dec_input_list, (h_0, c_0), attn_states, loop_function=loop_fn
        )

        # 将输出转换为actions
        actions = []
        for output in outputs:
            action = self.char2action_W_out(output)
            actions.append(action)

        return actions  # list of [batch_size, dim_action]

    def discriminator(self, char_seq, action_seq, batch_size, reuse_flag=False):
        """
        Discriminator: 判别action序列是真还是假

        char_seq: [batch_size, sentence_steps, dim_char_enc] 文本编码
        action_seq: [batch_size, action_steps, dim_action] 或 list of [batch_size, dim_action]

        返回: [batch_size, 1] 判别结果 (sigmoid激活后)
        """
        device = char_seq.device

        # 如果action_seq是tensor，转换为list
        if isinstance(action_seq, torch.Tensor):
            # [batch_size, action_steps, dim_action] -> list of [batch_size, dim_action]
            action_list = [action_seq[:, i, :] for i in range(action_seq.size(1))]
        else:
            action_list = action_seq

        # 关键修复：将action转换到dim_dis维度
        # 这对应TensorFlow版本中decoder_inputs的预期维度
        action_list_transformed = []
        for action in action_list:
            action_transformed = self.discriminator_W_in(action)
            action_list_transformed.append(action_transformed)

        # 准备attention states (只使用char_seq)
        attn_states = char_seq

        # 初始hidden state
        h_0 = torch.zeros(batch_size, self.dim_dis, device=device)
        c_0 = torch.zeros(batch_size, self.dim_dis, device=device)

        # 运行attention decoder (没有loop function)
        outputs, _ = self.discriminator_decoder(
            action_list_transformed, (h_0, c_0), attn_states, loop_function=None
        )

        # 取最后一个输出，通过全连接层和sigmoid
        last_output = outputs[-1]  # [batch_size, dim_dis]
        result = torch.sigmoid(self.discriminator_W_out(last_output))  # [batch_size, 1]

        return result

    def dis_loss(self, real_label, fake_label):
        """
        Discriminator loss
        对应TensorFlow: -mean(log(real_label) + log(1 - fake_label))
        """
        loss = -torch.mean(torch.log(real_label + 1e-8) + torch.log(1.0 - fake_label + 1e-8))
        return loss

    def gen_loss(self, fake_label):
        """
        Generator loss
        对应TensorFlow: -mean(log(fake_label))
        """
        loss = -torch.mean(torch.log(fake_label + 1e-8))
        return loss


if __name__ == "__main__":
    # 测试GAN模型
    print("测试GAN模型...")

    batch_size = 4
    sentence_steps = 30
    action_steps = 32
    dim_sentence = 300
    dim_char_enc = 300
    dim_gen = 300
    dim_dis = 300
    dim_random = 10

    model = GANModel(
        sentence_steps=sentence_steps,
        action_steps=action_steps,
        dim_sentence=dim_sentence,
        dim_char_enc=dim_char_enc,
        dim_gen=dim_gen,
        dim_dis=dim_dis,
        dim_random=dim_random
    )

    # 创建测试数据
    script = torch.randn(batch_size, sentence_steps, dim_sentence)
    seq_len = torch.tensor([25, 28, 30, 22])
    real_action = torch.randn(batch_size, action_steps, 24)
    init_action = torch.randn(batch_size, 24)
    random_noise = torch.randn(batch_size, sentence_steps, dim_random)

    # 测试
    print("\n1. 编码文本...")
    char_enc = model.char_encoder(script, seq_len)
    print(f"   Char encoding: {char_enc.shape}")

    print("\n2. 生成动作...")
    fake_actions = model.char2action(char_enc, init_action, random_noise, batch_size)
    print(f"   Generated {len(fake_actions)} action frames")
    print(f"   Each action shape: {fake_actions[0].shape}")

    # 转换为tensor用于discriminator
    fake_action_tensor = torch.stack(fake_actions, dim=1)
    print(f"   Stacked action tensor: {fake_action_tensor.shape}")

    print("\n3. 判别真实动作...")
    label_real = model.discriminator(char_enc, real_action, batch_size)
    print(f"   Real labels: {label_real.shape}, values: {label_real.squeeze()}")

    print("\n4. 判别生成动作...")
    label_fake = model.discriminator(char_enc, fake_action_tensor, batch_size)
    print(f"   Fake labels: {label_fake.shape}, values: {label_fake.squeeze()}")

    print("\n5. 计算损失...")
    dis_loss = model.dis_loss(label_real, label_fake)
    gen_loss = model.gen_loss(label_fake)
    print(f"   Discriminator loss: {dis_loss.item():.4f}")
    print(f"   Generator loss: {gen_loss.item():.4f}")

    print("\n✅ GAN模型测试通过!")
    print(f"\n维度信息:")
    print(f"   char2action attention_size: {model.char2action_attention_size}")
    print(f"   discriminator attention_size: {model.discriminator_attention_size}")
    print(f"\n预期的LSTM输入维度:")
    print(f"   Generator LSTM input: decoder_input({model.dim_gen}) + context({model.char2action_attention_size}) = {model.dim_gen + model.char2action_attention_size}")
    print(f"   Discriminator LSTM input: decoder_input({model.dim_dis}) + context({model.discriminator_attention_size}) = {model.dim_dis + model.discriminator_attention_size}")