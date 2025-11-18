import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionDecoder(nn.Module):
    """
    自定义attention decoder，对应TensorFlow的attention_decoder
    关键: 必须在解码过程中应用loop_function
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

        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, attention_size]

        # 扩展hidden用于broadcasting
        hidden_expanded = hidden.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_size]

        # 计算attention scores
        scores = self.v_a(torch.tanh(
            self.W_a(hidden_expanded) + self.U_a(encoder_outputs)
        ))  # [batch_size, seq_len, 1]

        # Softmax得到attention权重
        attention_weights = F.softmax(scores, dim=1)  # [batch_size, seq_len, 1]

        # 计算context向量
        context = torch.sum(attention_weights * encoder_outputs, dim=1)  # [batch_size, attention_size]

        return context, attention_weights

    def forward(self, inputs, initial_state, attention_states, loop_function=None):
        """
        前向传播 - 关键：正确应用loop_function

        inputs: list of [batch_size, input_dim] tensors (初始输入列表)
        initial_state: (h_0, c_0) tuple
        attention_states: [batch_size, seq_len, attention_size]
        loop_function: 可选的函数，将当前输出转换为下一个输入

        注意：loop_function在TensorFlow中的作用是：
        - 在第一步使用提供的input
        - 从第二步开始，使用loop_function(上一步的输出)作为输入
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


class Seq2SeqModel(nn.Module):
    """
    Seq2Seq模型 - 严格对应TensorFlow版本
    注意：不包含action_encoder，因为TF版本中没有
    """

    def __init__(self, sentence_steps, action_steps, dim_sentence, dim_char_enc, dim_gen, dim_action=263, dim_random=10):
        super(Seq2SeqModel, self).__init__()

        self.action_steps = action_steps
        self.dim_action = dim_action  # 支持可配置的动作维度，默认263（HumanML3D标准版）
        self.sentence_steps = sentence_steps
        self.dim_sentence = dim_sentence
        self.dim_char_enc = dim_char_enc
        self.dim_gen = dim_gen
        self.dim_random = dim_random

        # Character encoder LSTM
        self.char_encoder_lstm = nn.LSTM(
            input_size=dim_sentence,
            hidden_size=dim_char_enc,
            batch_first=True
        )

        # Char2Action decoder
        # attention_size = char_enc维度 + random_noise维度
        # 如果dim_random > 0，则attention_size包含random noise
        char2action_attention_size = dim_char_enc + (dim_random if dim_random > 0 else 0)
        self.char2action_decoder = AttentionDecoder(dim_gen, char2action_attention_size)
        self.char2action_W_out = nn.Linear(dim_gen, self.dim_action)
        self.char2action_W_in = nn.Linear(self.dim_action, dim_gen)

        # Action2Char decoder
        # attention_size = action_enc维度 + random_noise维度
        action2char_attention_size = dim_gen + (dim_random if dim_random > 0 else 0)
        self.action2char_decoder = AttentionDecoder(dim_gen, action2char_attention_size)
        self.action2char_W_out = nn.Linear(dim_gen, dim_sentence)
        self.action2char_W_in = nn.Linear(dim_sentence, dim_gen)

        # 初始化权重
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

        注意: 即使提供seq_len，输出长度仍然固定为sentence_steps，
        以匹配TensorFlow版本的行为
        """
        if seq_len is not None:
            # Pack padded sequence以提高效率
            # 重要: seq_len必须在CPU上
            x_packed = nn.utils.rnn.pack_padded_sequence(
                x, seq_len.cpu(), batch_first=True, enforce_sorted=False
            )
            output, _ = self.char_encoder_lstm(x_packed)
            # Pad回固定长度，确保输出是[batch_size, sentence_steps, dim_char_enc]
            output, _ = nn.utils.rnn.pad_packed_sequence(
                output, batch_first=True, total_length=self.sentence_steps
            )
        else:
            output, _ = self.char_encoder_lstm(x)

        return output

    def char2action(self, char_enc, init_input, random_noise, batch_size):
        """
        从文本编码生成动作序列

        char_enc: [batch_size, sentence_steps, dim_char_enc] (固定长度)
        init_input: [batch_size, dim_action] 初始pose
        random_noise: [batch_size, sentence_steps, dim_random]

        返回:
        - actions: [batch_size, action_steps, dim_action]
        - action_features: [batch_size, action_steps, dim_gen] decoder的隐藏状态
        """
        device = char_enc.device

        # 准备decoder输入（全部用转换后的init_input初始化）
        dec_input_list = []
        for _ in range(self.action_steps):
            dec_input = self.char2action_W_in(init_input)
            dec_input_list.append(dec_input)

        # 拼接char_enc和random noise作为attention states
        if random_noise.size(-1) > 0:
            # char_enc现在是固定长度sentence_steps
            # 确保random_noise也是sentence_steps长度
            if random_noise.size(1) == char_enc.size(1):
                attn_states = torch.cat([char_enc, random_noise], dim=2)
            else:
                # 安全检查：如果长度不匹配，截断或填充
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

        # Stack outputs
        actions = torch.stack(actions, dim=1)  # [batch_size, action_steps, dim_action]
        action_features = torch.stack(outputs, dim=1)  # [batch_size, action_steps, dim_gen]

        return actions, action_features

    def action2char(self, action_enc, init_input, random_noise, batch_size):
        """
        从动作编码重建文本序列

        关键：action_enc是char2action decoder的输出隐藏状态，不是重新编码的action!

        action_enc: [batch_size, actual_action_steps, dim_gen] 来自char2action的decoder hidden states
        init_input: [batch_size, dim_sentence] 初始字符输入
        random_noise: [batch_size, action_steps, dim_random]

        返回: [batch_size, sentence_steps, dim_sentence]
        """
        device = action_enc.device

        # 准备decoder输入
        dec_input_list = []
        for _ in range(self.sentence_steps):
            dec_input = self.action2char_W_in(init_input)
            dec_input_list.append(dec_input)

        # 拼接action_enc和random noise作为attention states
        # 注意: random_noise需要匹配action_enc的实际长度
        if random_noise.size(-1) > 0:
            # 取action_enc的实际长度
            actual_action_len = action_enc.size(1)
            # 调整random_noise的长度以匹配action_enc
            if random_noise.size(1) != actual_action_len:
                # 截断或填充random_noise
                if random_noise.size(1) > actual_action_len:
                    random_noise_adjusted = random_noise[:, :actual_action_len, :]
                else:
                    # 填充零
                    padding = torch.zeros(batch_size, actual_action_len - random_noise.size(1),
                                          random_noise.size(-1), device=device)
                    random_noise_adjusted = torch.cat([random_noise, padding], dim=1)
            else:
                random_noise_adjusted = random_noise

            attn_states = torch.cat([action_enc, random_noise_adjusted], dim=2)
        else:
            attn_states = action_enc

        # 初始hidden state
        h_0 = torch.zeros(batch_size, self.dim_gen, device=device)
        c_0 = torch.zeros(batch_size, self.dim_gen, device=device)

        # 定义loop function
        def loop_fn(prev_output, i):
            char = self.action2char_W_out(prev_output)
            next_input = self.action2char_W_in(char)
            return next_input

        # 运行attention decoder
        outputs, _ = self.action2char_decoder(
            dec_input_list, (h_0, c_0), attn_states, loop_function=loop_fn
        )

        # 将输出转换为字符
        chars = []
        for output in outputs:
            char = self.action2char_W_out(output)
            chars.append(char)

        # Stack outputs
        chars = torch.stack(chars, dim=1)  # [batch_size, sentence_steps, dim_sentence]

        return chars

    def seq2seq_loss(self, fake_action, real_action, fake_char, real_char):
        """
        计算seq2seq loss
        对应TensorFlow版本的loss计算

        fake_action: [batch_size, action_steps, dim_action]
        real_action: [batch_size, action_steps, dim_action]
        fake_char: [batch_size, sentence_steps, dim_sentence]
        real_char: [batch_size, sentence_steps, dim_sentence]
        """
        action_loss = F.mse_loss(fake_action, real_action)
        char_loss = F.mse_loss(fake_char, real_char)

        # 对应TensorFlow: return action_loss + 5.0 * enc_loss
        total_loss = action_loss + 5.0 * char_loss

        return total_loss, action_loss, char_loss


if __name__ == "__main__":
    # 测试模型
    print("测试Seq2Seq模型...")

    batch_size = 4
    sentence_steps = 30
    action_steps = 32
    dim_sentence = 300
    dim_char_enc = 256
    dim_gen = 256
    dim_action = 263  # HumanML3D标准版维度
    dim_random = 16

    model = Seq2SeqModel(sentence_steps, action_steps, dim_sentence, dim_char_enc, dim_gen, dim_action, dim_random)

    # 创建测试数据
    script = torch.randn(batch_size, sentence_steps, dim_sentence)
    seq_len = torch.tensor([25, 28, 30, 22])
    action = torch.randn(batch_size, action_steps, dim_action)
    init_action = torch.randn(batch_size, dim_action)
    init_char = torch.randn(batch_size, dim_sentence)
    random_c2a = torch.randn(batch_size, sentence_steps, dim_random)
    random_a2c = torch.randn(batch_size, action_steps, dim_random)

    # 前向传播
    print("\n1. 编码文本...")
    char_enc = model.char_encoder(script, seq_len)
    print(f"   Char encoding: {char_enc.shape}")  # [4, 30, 256]

    print("\n2. 生成动作...")
    fake_action, action_enc = model.char2action(char_enc, init_action, random_c2a, batch_size)
    print(f"   Generated action: {fake_action.shape}")  # [4, 32, 263]
    print(f"   Action features (decoder states): {action_enc.shape}")  # [4, 32, 256]

    print("\n3. 重建文本...")
    fake_char = model.action2char(action_enc, init_char, random_a2c, batch_size)
    print(f"   Reconstructed text: {fake_char.shape}")  # [4, 30, 300]

    print("\n4. 计算损失...")
    loss, action_loss, char_loss = model.seq2seq_loss(fake_action, action, fake_char, script)
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   Action loss: {action_loss.item():.4f}")
    print(f"   Char loss: {char_loss.item():.4f}")

    print("\n✅ 模型测试通过!")