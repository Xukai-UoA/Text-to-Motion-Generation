"""
优化版动作可视化工具
改进：
1. 头部关键点使用红色
2. 胳膊和手部关键点使用绿色
3. 颈部关键点使用蓝色（作为基准点）
4. 自动对齐骨架到标准视角（头部在上，躯干居中，正面展示）
   - 智能检测头部-颈部方向确定"上"方向
   - 根据左右肩膀位置确定"左右"方向
   - 自动构建标准坐标系，确保每个动作都以一致的正面视角展示
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
import os


# ==================== 骨架定义 ====================
def get_skeleton_connections():
    """
    定义骨架连接关系
    关节索引：
    0: 颈部 (neck) - 蓝色
    1: 左肩 (left shoulder) - 绿色
    2: 右肩 (right shoulder) - 绿色
    3: 左肘 (left elbow) - 绿色
    4: 右肘 (right elbow) - 绿色
    5: 左手腕 (left wrist) - 绿色
    6: 右手腕 (right wrist) - 绿色
    7: 头部 (head) - 红色
    """
    connections = [
        (0, 1),  # 颈部 -> 左肩
        (0, 2),  # 颈部 -> 右肩
        (1, 3),  # 左肩 -> 左肘
        (2, 4),  # 右肩 -> 右肘
        (3, 5),  # 左肘 -> 左手腕
        (4, 6),  # 右肘 -> 右手腕
        (0, 7),  # 颈部 -> 头部
    ]
    return connections


def get_joint_colors():
    """
    定义每个关节的颜色
    返回: 关节索引 -> 颜色的字典
    """
    colors = {
        0: 'blue',      # 颈部 - 蓝色（基准点）
        1: 'green',     # 左肩 - 绿色
        2: 'green',     # 右肩 - 绿色
        3: 'green',     # 左肘 - 绿色
        4: 'green',     # 右肘 - 绿色
        5: 'green',     # 左手腕 - 绿色
        6: 'green',     # 右手腕 - 绿色
        7: 'red',       # 头部 - 红色
    }
    return colors


def get_connection_colors():
    """
    定义每条连接线的颜色
    返回: 连接 -> 颜色的字典
    """
    connection_colors = {
        (0, 1): 'red',    # 颈部->左肩
        (0, 2): 'red',    # 颈部->右肩
        (1, 3): 'green',    # 左肩->左肘
        (2, 4): 'green',    # 右肩->右肘
        (3, 5): 'green',    # 左肘->左手腕
        (4, 6): 'green',    # 右肘->右手腕
        (0, 7): 'red',      # 颈部->头部
    }
    return connection_colors


def pose_to_joint_positions(pose_vector):
    """
    将24维pose向量转换为8个关节的3D位置

    Args:
        pose_vector: [24] 维度的pose向量
            [0:3]   - 颈部3D位置 (x, y, z)
            [3:6]   - 关节向量1
            [6:9]   - 关节向量2
            ...
            [21:24] - 关节向量7

    Returns:
        joint_positions: [8, 3] - 8个关节的3D坐标
    """
    neck_pos = pose_vector[0:3]
    joint_positions = [neck_pos]

    scale_factor = 0.3

    for i in range(7):
        joint_vec = pose_vector[3+i*3:3+(i+1)*3]
        vec_norm = np.linalg.norm(joint_vec)
        if vec_norm > 0:
            joint_vec = joint_vec / vec_norm * scale_factor

        if i < 2:  # 肩膀
            joint_pos = neck_pos + joint_vec
        elif i < 4:  # 肘部
            shoulder_idx = i - 2 + 1
            joint_pos = joint_positions[shoulder_idx] + joint_vec
        elif i < 6:  # 手腕
            elbow_idx = i - 4 + 3
            joint_pos = joint_positions[elbow_idx] + joint_vec
        else:  # 头部
            joint_pos = neck_pos + joint_vec

        joint_positions.append(joint_pos)

    return np.array(joint_positions)


def align_skeleton_to_standard_view(joint_positions):
    """
    自动对齐骨架到标准视角坐标系

    标准视角定义：
    - Z轴向上（头部在颈部上方）
    - Y轴向前（身体正面朝向观察者）
    - X轴向右（左臂在左侧，右臂在右侧）
    - 颈部位于坐标系中心

    Args:
        joint_positions: [8, 3] - 8个关节的3D坐标
            0: 颈部, 1: 左肩, 2: 右肩, 3-6: 肘部和手腕, 7: 头部

    Returns:
        aligned_positions: [8, 3] - 对齐后的关节坐标
    """
    # 提取关键关节
    neck = joint_positions[0]       # 颈部
    left_shoulder = joint_positions[1]   # 左肩
    right_shoulder = joint_positions[2]  # 右肩
    head = joint_positions[7]       # 头部

    # 1. 计算"上"方向 (Z轴)：从颈部指向头部
    up_vec = head - neck
    up_norm = np.linalg.norm(up_vec)
    if up_norm < 1e-6:
        # 如果头部和颈部重合，使用默认向上方向
        up_vec = np.array([0, 0, 1])
    else:
        up_vec = up_vec / up_norm

    # 2. 计算"右"方向 (X轴)：从左肩指向右肩
    right_vec = right_shoulder - left_shoulder
    right_norm = np.linalg.norm(right_vec)
    if right_norm < 1e-6:
        # 如果左右肩重合，使用默认右方向
        right_vec = np.array([1, 0, 0])
    else:
        right_vec = right_vec / right_norm

    # 3. 计算"前"方向 (Y轴)：右向量 × 上向量
    # 右手坐标系：right × up = forward
    forward_vec = np.cross(right_vec, up_vec)
    forward_norm = np.linalg.norm(forward_vec)
    if forward_norm < 1e-6:
        # 如果向量平行，重新计算
        forward_vec = np.array([0, 1, 0])
    else:
        forward_vec = forward_vec / forward_norm

    # 4. 重新正交化：确保三个轴互相垂直
    # 固定Z轴(up)，重新计算X轴
    right_vec = np.cross(forward_vec, up_vec)
    right_vec = right_vec / np.linalg.norm(right_vec)

    # 再次计算Y轴以确保完全正交
    forward_vec = np.cross(right_vec, up_vec)
    forward_vec = forward_vec / np.linalg.norm(forward_vec)

    # 5. 构建旋转矩阵
    # 目标坐标系：X轴=[1,0,0], Y轴=[0,1,0], Z轴=[0,0,1]
    # 当前坐标系：X轴=right_vec, Y轴=forward_vec, Z轴=up_vec
    # 旋转矩阵R: 将当前坐标系转换到目标坐标系
    rotation_matrix = np.column_stack([right_vec, forward_vec, up_vec])

    # 6. 对所有关节进行变换
    aligned_positions = np.zeros_like(joint_positions)
    for i in range(len(joint_positions)):
        # 先平移：将颈部移到原点
        centered = joint_positions[i] - neck
        # 再旋转：对齐到标准坐标系
        aligned_positions[i] = rotation_matrix.T @ centered

    return aligned_positions


# ==================== 8关键帧可视化 ====================
def visualize_key_frames(action_seq, title="Generated Action", save_path=None, num_frames=8):
    """
    可视化动作序列的关键帧（彩色版，自动对齐视角）

    该函数会自动将骨架对齐到标准视角：
    - Z轴向上（头部在颈部上方）
    - Y轴向前（身体正面朝向观察者）
    - X轴向右（左臂在左侧，右臂在右侧）

    Args:
        action_seq: [dim_action, action_steps] 或 [1, dim_action, action_steps]
        title: 图表标题
        save_path: 保存路径
        num_frames: 显示的关键帧数量（默认8）
    """
    if action_seq.ndim == 3:
        action_seq = action_seq[0]

    action_steps = action_seq.shape[1]
    frame_indices = np.linspace(0, action_steps-1, num_frames, dtype=int)

    cols = 4
    rows = (num_frames + cols - 1) // cols
    fig = plt.figure(figsize=(16, rows * 4))

    connections = get_skeleton_connections()
    joint_colors = get_joint_colors()
    connection_colors = get_connection_colors()

    # 计算全局坐标范围（对齐后）
    all_positions = []
    for frame_idx in frame_indices:
        pose = action_seq[:, frame_idx]
        positions = pose_to_joint_positions(pose)
        # 自动对齐到标准视角
        positions = align_skeleton_to_standard_view(positions)
        all_positions.append(positions)

    all_positions = np.concatenate(all_positions, axis=0)
    x_range = [all_positions[:, 0].min() - 0.2, all_positions[:, 0].max() + 0.2]
    y_range = [all_positions[:, 1].min() - 0.2, all_positions[:, 1].max() + 0.2]
    z_range = [all_positions[:, 2].min() - 0.2, all_positions[:, 2].max() + 0.2]

    # 绘制每个关键帧
    for i, frame_idx in enumerate(frame_indices):
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')

        pose = action_seq[:, frame_idx]
        joint_positions = pose_to_joint_positions(pose)
        # 自动对齐到标准视角
        joint_positions = align_skeleton_to_standard_view(joint_positions)

        # 绘制关节点（按颜色分组）
        for joint_idx in range(len(joint_positions)):
            color = joint_colors.get(joint_idx, 'gray')
            ax.scatter(joint_positions[joint_idx, 0],
                      joint_positions[joint_idx, 1],
                      joint_positions[joint_idx, 2],
                      c=color, s=150, alpha=0.9,
                      edgecolors='darkgray', linewidths=2.5,
                      zorder=10)  # zorder确保点在线上面

        # 绘制骨架连接（彩色）
        for conn in connections:
            if conn[1] < len(joint_positions):
                color = connection_colors.get(conn, 'gray')
                ax.plot([joint_positions[conn[0], 0], joint_positions[conn[1], 0]],
                       [joint_positions[conn[0], 1], joint_positions[conn[1], 1]],
                       [joint_positions[conn[0], 2], joint_positions[conn[1], 2]],
                       color=color, linewidth=3.5, alpha=0.7, zorder=5)

        # 设置坐标轴
        ax.set_xlabel('X', fontsize=10, fontweight='bold')
        ax.set_ylabel('Y', fontsize=10, fontweight='bold')
        ax.set_zlabel('Z', fontsize=10, fontweight='bold')
        ax.set_title(f'Frame {frame_idx}/{action_steps-1}', fontsize=12, fontweight='bold')

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        # 正面视角：elev=10(稍微从上往下看), azim=0(正面)
        ax.view_init(elev=10, azim=0)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=16, fontweight='bold')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Head'),
        Patch(facecolor='green', label='Arms/Hands'),
        Patch(facecolor='blue', label='Neck')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ 关键帧可视化保存到: {save_path}")

    plt.close()


# ==================== 32帧完整动画 ====================
def visualize_full_animation(action_seq, title="Generated Action", save_path=None):
    """
    生成32帧完整动作动画（彩色版，自动对齐视角）

    该函数会自动将骨架对齐到标准视角：
    - Z轴向上（头部在颈部上方）
    - Y轴向前（身体正面朝向观察者）
    - X轴向右（左臂在左侧，右臂在右侧）

    Args:
        action_seq: [dim_action, action_steps] 或 [1, dim_action, action_steps]
        title: 动画标题
        save_path: 保存路径（.gif）
    """
    if action_seq.ndim == 3:
        action_seq = action_seq[0]

    action_steps = action_seq.shape[1]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    connections = get_skeleton_connections()
    joint_colors = get_joint_colors()
    connection_colors = get_connection_colors()

    # 计算全局坐标范围（对齐后）
    all_positions = []
    for t in range(action_steps):
        pose = action_seq[:, t]
        positions = pose_to_joint_positions(pose)
        # 自动对齐到标准视角
        positions = align_skeleton_to_standard_view(positions)
        all_positions.append(positions)

    all_positions = np.concatenate(all_positions, axis=0)
    x_range = [all_positions[:, 0].min() - 0.3, all_positions[:, 0].max() + 0.3]
    y_range = [all_positions[:, 1].min() - 0.3, all_positions[:, 1].max() + 0.3]
    z_range = [all_positions[:, 2].min() - 0.3, all_positions[:, 2].max() + 0.3]

    # 初始化绘图元素 - 为每个关节创建单独的scatter
    scatters = []
    for joint_idx in range(8):
        color = joint_colors.get(joint_idx, 'gray')
        scatter = ax.scatter([], [], [], c=color, s=150, alpha=0.9,
                           edgecolors='darkgray', linewidths=2.5, zorder=10)
        scatters.append(scatter)

    # 为每条连接创建单独的line
    lines = []
    for conn in connections:
        color = connection_colors.get(conn, 'gray')
        line, = ax.plot([], [], [], color=color, linewidth=4, alpha=0.7, zorder=5)
        lines.append(line)

    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    ax.set_xlabel('X', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z', fontsize=12, fontweight='bold')
    # 正面视角：elev=10(稍微从上往下看), azim=0(正面)
    ax.view_init(elev=10, azim=0)
    ax.grid(True, alpha=0.3)

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Head'),
        Patch(facecolor='green', label='Arms/Hands'),
        Patch(facecolor='blue', label='Neck')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    frame_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes,
                          fontsize=14, fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def init():
        """初始化函数"""
        for scatter in scatters:
            scatter._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        frame_text.set_text('')
        return scatters + lines + [frame_text]

    def update(frame):
        """更新函数"""
        pose = action_seq[:, frame]
        joint_positions = pose_to_joint_positions(pose)
        # 自动对齐到标准视角
        joint_positions = align_skeleton_to_standard_view(joint_positions)

        # 更新每个关节点
        for joint_idx, scatter in enumerate(scatters):
            if joint_idx < len(joint_positions):
                scatter._offsets3d = ([joint_positions[joint_idx, 0]],
                                     [joint_positions[joint_idx, 1]],
                                     [joint_positions[joint_idx, 2]])

        # 更新骨架连接线
        for i, conn in enumerate(connections):
            if conn[1] < len(joint_positions):
                xs = [joint_positions[conn[0], 0], joint_positions[conn[1], 0]]
                ys = [joint_positions[conn[0], 1], joint_positions[conn[1], 1]]
                zs = [joint_positions[conn[0], 2], joint_positions[conn[1], 2]]
                lines[i].set_data(xs, ys)
                lines[i].set_3d_properties(zs)

        frame_text.set_text(f'Frame: {frame}/{action_steps-1}')

        return scatters + lines + [frame_text]

    print(f"正在生成彩色动画 (共{action_steps}帧)...")
    anim = FuncAnimation(fig, update, frames=action_steps,
                        init_func=init, blit=True, interval=100)

    if save_path:
        if not save_path.endswith('.gif'):
            save_path = save_path.replace('.png', '.gif')

        writer = PillowWriter(fps=10)
        anim.save(save_path, writer=writer, dpi=100)
        print(f"✓ 彩色动画保存到: {save_path}")
        print(f"  - 总帧数: {action_steps}")
        print(f"  - 时长: {action_steps/10:.1f}秒")
        print(f"  - 颜色方案: 头部(红色) 手臂(绿色) 颈部(蓝色)")

    plt.close()


# ==================== 32帧网格图 ====================
def visualize_all_frames_grid(action_seq, title="All Frames", save_path=None):
    """
    在一张大图中显示所有32帧（彩色版，自动对齐视角）

    该函数会自动将骨架对齐到标准视角：
    - Z轴向上（头部在颈部上方）
    - Y轴向前（身体正面朝向观察者）
    - X轴向右（左臂在左侧，右臂在右侧）

    Args:
        action_seq: [dim_action, action_steps] 或 [1, dim_action, action_steps]
        title: 图表标题
        save_path: 保存路径
    """
    if action_seq.ndim == 3:
        action_seq = action_seq[0]

    action_steps = action_seq.shape[1]

    cols = 8
    rows = 4
    fig = plt.figure(figsize=(24, 12))

    connections = get_skeleton_connections()
    joint_colors = get_joint_colors()
    connection_colors = get_connection_colors()

    # 计算全局坐标范围（对齐后）
    all_positions = []
    for t in range(action_steps):
        pose = action_seq[:, t]
        positions = pose_to_joint_positions(pose)
        # 自动对齐到标准视角
        positions = align_skeleton_to_standard_view(positions)
        all_positions.append(positions)

    all_positions = np.concatenate(all_positions, axis=0)
    x_range = [all_positions[:, 0].min() - 0.2, all_positions[:, 0].max() + 0.2]
    y_range = [all_positions[:, 1].min() - 0.2, all_positions[:, 1].max() + 0.2]
    z_range = [all_positions[:, 2].min() - 0.2, all_positions[:, 2].max() + 0.2]

    # 绘制每一帧
    for frame_idx in range(action_steps):
        ax = fig.add_subplot(rows, cols, frame_idx+1, projection='3d')

        pose = action_seq[:, frame_idx]
        joint_positions = pose_to_joint_positions(pose)
        # 自动对齐到标准视角
        joint_positions = align_skeleton_to_standard_view(joint_positions)

        # 绘制关节（按颜色）
        for joint_idx in range(len(joint_positions)):
            color = joint_colors.get(joint_idx, 'gray')
            ax.scatter(joint_positions[joint_idx, 0],
                      joint_positions[joint_idx, 1],
                      joint_positions[joint_idx, 2],
                      c=color, s=60, alpha=0.9, zorder=10)

        # 绘制骨架（彩色）
        for conn in connections:
            if conn[1] < len(joint_positions):
                color = connection_colors.get(conn, 'gray')
                ax.plot([joint_positions[conn[0], 0], joint_positions[conn[1], 0]],
                       [joint_positions[conn[0], 1], joint_positions[conn[1], 1]],
                       [joint_positions[conn[0], 2], joint_positions[conn[1], 2]],
                       color=color, linewidth=2.5, alpha=0.7, zorder=5)

        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_zlim(z_range)
        ax.set_title(f'F{frame_idx}', fontsize=8)
        # 正面视角：elev=10(稍微从上往下看), azim=0(正面)
        ax.view_init(elev=10, azim=0)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)

    fig.suptitle(title, fontsize=18, fontweight='bold')

    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='Head'),
        Patch(facecolor='green', label='Arms/Hands'),
        Patch(facecolor='blue', label='Neck')
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ 完整32帧彩色网格图保存到: {save_path}")

    plt.close()


# ==================== 便捷函数 ====================
def visualize_action_complete(action_seq, sentence, output_dir='./test_results'):
    """
    一次性生成所有可视化（使用完整句子作为文件名）

    Args:
        action_seq: [1, dim_action, action_steps] 或 [dim_action, action_steps]
        sentence: 句子描述
        output_dir: 输出目录

    Returns:
        生成的文件路径字典
    """
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名：使用完整句子，替换空格为下划线
    filename_prefix = sentence.lower().replace(' ', '_')

    print(f"\n{'='*60}")
    print(f"为句子 '{sentence}' 生成完整可视化")
    print(f"文件名前缀: {filename_prefix}")
    print(f"{'='*60}")

    files = {}

    # 1. 8个关键帧
    print("\n[1/3] 生成8个关键帧（彩色版）...")
    key_frames_path = os.path.join(output_dir, f'{filename_prefix}_8frames.png')
    visualize_key_frames(action_seq, title=f"Key Frames: {sentence}",
                        save_path=key_frames_path)
    files['key_frames'] = key_frames_path

    # 2. 32帧完整动画GIF
    print("\n[2/3] 生成32帧彩色动画GIF...")
    animation_path = os.path.join(output_dir, f'{filename_prefix}_animation.gif')
    visualize_full_animation(action_seq, title=sentence,
                            save_path=animation_path)
    files['animation'] = animation_path

    # 3. 32帧网格图
    print("\n[3/3] 生成32帧彩色网格图...")
    grid_path = os.path.join(output_dir, f'{filename_prefix}_32frames_grid.png')
    visualize_all_frames_grid(action_seq, title=f"All 32 Frames: {sentence}",
                             save_path=grid_path)
    files['grid'] = grid_path

    print(f"\n{'='*60}")
    print("✓ 所有彩色可视化生成完成！")
    print(f"{'='*60}")
    print("\n生成的文件:")
    for key, path in files.items():
        print(f"  - {key}: {os.path.basename(path)}")

    return files


if __name__ == "__main__":
    """测试可视化功能"""
    print("="*60)
    print("测试优化版可视化工具")
    print("="*60)

    # 生成测试数据
    print("\n生成测试数据...")
    test_action = np.random.randn(1, 24, 32) * 0.5

    for t in range(32):
        test_action[0, 0, t] = 0.1 * np.sin(t * 0.2)
        test_action[0, 1, t] = 0.1 * np.cos(t * 0.2)
        test_action[0, 2, t] = t * 0.01

    print(f"测试数据形状: {test_action.shape}")

    # 测试完整句子文件名
    test_sentence = "a woman is dancing gracefully"
    output_dir = "./test_results"

    files = visualize_action_complete(test_action, test_sentence, output_dir)

    print("\n" + "="*60)
    print("测试完成！请查看生成的文件：")
    print("="*60)
    for key, path in files.items():
        print(f"\n{key}:")
        print(f"  {path}")
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  文件大小: {size_kb:.1f} KB")

    print("\n颜色方案:")
    print("  🔴 红色: 头部")
    print("  🟢 绿色: 手臂和手部")
    print("  🔵 蓝色: 颈部")
