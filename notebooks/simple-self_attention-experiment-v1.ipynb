import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 设置随机种子以保证结果可复现
torch.manual_seed(0)

# 定义词表和映射
vocab = ["I", "like", "cats", "dogs"]
word2idx = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
embed_dim = 4   # 嵌入维度为4
seq_length = 3  # 序列长度为3

# 构造训练数据: 多个长度为3的序列和对应的目标输出
# 任务: 若序列中包含 "cats"，则预测输出 "dogs"；若包含 "dogs"，则预测输出 "cats"
# （每个序列恰好包含一个 "cats" 或 "dogs"）
train_data = []
others = ["I", "like"]     # 其他词（非 cats/dogs）
animals = ["cats", "dogs"] # 动物词列表
for animal in animals:
    other_animal = "dogs" if animal == "cats" else "cats"
    for w1 in others:
        for w2 in others:
            # 确保这两个位置都不包含动物词
            if w1 in animals or w2 in animals:
                continue
            # 将当前 animal 插入序列的不同位置来构造长度为3的序列
            base_seq = [w1, w2]
            for pos in range(seq_length):
                seq = base_seq.copy()
                seq.insert(pos, animal)
                seq = seq[:seq_length]  # 插入后可能长度为4，截断为3
                # 确认序列中恰好包含一个 animal 且不包含 other_animal
                if seq.count(animal) != 1 or other_animal in seq:
                    continue
                train_data.append(([word2idx[w] for w in seq], word2idx[other_animal]))

# 移除重复的样本
train_data_unique = []
seen = set()
for inp, tgt in train_data:
    key = (tuple(inp), tgt)
    if key not in seen:
        train_data_unique.append((inp, tgt))
        seen.add(key)
train_data = train_data_unique

print(f"训练样本数: {len(train_data)}")
for i in range(min(5, len(train_data))):
    inp_idx, tgt_idx = train_data[i]
    inp_words = [vocab[idx] for idx in inp_idx]
    print(f"样本{i}: 输入{inp_words} -> 目标 {vocab[tgt_idx]}")

# 定义包含单头自注意力机制的模型（无位置编码、无前馈网络）
class SelfAttentionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SelfAttentionModel, self).__init__()
        # 词嵌入矩阵
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # 线性变换，用于计算 Q, K, V 向量
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        # 输出层，将注意力输出映射到预测空间（词表大小）
        self.out = nn.Linear(embed_dim, vocab_size)
    def forward(self, x):
        """
        参数:
            x: 张量，形状为 (batch_size, seq_length) 或 (seq_length,)
        返回:
            logits: 张量，形状 (batch_size, vocab_size)，模型的预测输出（未经过 softmax）
            attn_weights: 注意力权重矩阵，形状 (batch_size, seq_length, seq_length)
            Q, K, V: 用于可视化的查询、键、值向量，形状 (batch_size, seq_length, embed_dim)
        """
        # 如果 x 是一维 (seq_length)，则增加一个 batch 维度
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, seq_length)
        # 1. 计算词嵌入表示
        # emb.shape = (batch_size, seq_length, embed_dim)
        emb = self.embedding(x)
        # 2. 计算 Q, K, V 矩阵
        # Q, K, V 形状均为 (batch_size, seq_length, embed_dim)
        Q = self.W_q(emb)
        K = self.W_k(emb)
        V = self.W_v(emb)
        # 3. 计算注意力分数 (没有使用 mask)
        # scores.shape = (batch_size, seq_length, seq_length)
        d_k = Q.shape[-1]
        scores = torch.einsum('bid,bjd->bij', Q, K) / (d_k ** 0.5)
        # 4. 对分数应用 softmax 得到注意力权重矩阵
        # attn_weights.shape = (batch_size, seq_length, seq_length)
        attn_weights = F.softmax(scores, dim=-1)
        # 5. 以注意力权重对 V 加权求和，得到注意力层的输出
        # output.shape = (batch_size, seq_length, embed_dim)
        output = torch.einsum('bij,bjd->bid', attn_weights, V)
        # 6. 这里我们取第一个序列位置的输出作为序列的整体表示（类似于 [CLS]）
        context = output[:, 0, :]  # 形状 (batch_size, embed_dim)
        # 通过输出层获得对下一个词的预测
        logits = self.out(context)  # 形状 (batch_size, vocab_size)
        return logits, attn_weights, Q, K, V

# 初始化模型、损失函数和优化器
model = SelfAttentionModel(vocab_size, embed_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

epochs = 50
# 选择一个输入序列用于每轮训练后的可视化
vis_input_words = ["I", "like", "cats"]
vis_input_idx = torch.tensor([word2idx[w] for w in vis_input_words])

for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    # 遍历每个训练样本
    for inp, tgt in train_data:
        inp_tensor = torch.tensor(inp)
        tgt_tensor = torch.tensor([tgt])
        optimizer.zero_grad()
        logits, attn_weights, Q, K, V = model(inp_tensor)
        loss = criterion(logits, tgt_tensor)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # 打印当前轮次的损失值
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")
    # 每轮训练结束后进行模型评估并记录可视化数据
    model.eval()
    with torch.no_grad():
        logits, attn_weights, Q, K, V = model(vis_input_idx)
    # 提取当前批次（仅一个样本）的注意力权重和 Q,K 向量
    attn = attn_weights[0].cpu().numpy()       # (3, 3)
    Q_mat = Q[0].cpu().numpy()                 # (3, embed_dim)
    K_mat = K[0].cpu().numpy()                 # (3, embed_dim)
    # 提取词嵌入矩阵用于可视化
    embed_mat = model.embedding.weight.detach().cpu().numpy()  # (vocab_size, embed_dim)
    # 计算词嵌入向量的 PCA 前两主成分，以投影到二维平面
    X = embed_mat
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    proj_matrix = Vt.T[:, :2]               # 取前2个主成分方向 (embed_dim, 2)
    embed_2d = X_centered.dot(proj_matrix)  # 投影后的二维坐标 (vocab_size, 2)
    # 将 Q 和 K 向量合在一起做 PCA，以便共同投影到二维平面进行比较
    QK = np.vstack([Q_mat, K_mat])          # 形状 (6, embed_dim)
    QK_centered = QK - QK.mean(axis=0)
    U_qk, S_qk, Vt_qk = np.linalg.svd(QK_centered, full_matrices=False)
    proj_qk = Vt_qk.T[:, :2]                # 前2主成分方向 (embed_dim, 2)
    QK_2d = QK_centered.dot(proj_qk)        # 投影到二维 (6, 2)
    Q_2d = QK_2d[:len(Q_mat), :]            # 前3个为 Q 投影
    K_2d = QK_2d[len(Q_mat):, :]            # 后3个为 K 投影
    # 创建可视化图形 (2x2 子图)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    ax1, ax2, ax3, ax4 = axes[0][0], axes[0][1], axes[1][0], axes[1][1]
    # 子图1: 词嵌入向量的二维投影散点图
    ax1.set_title("Embedding 2D Projection")
    colors = {"I": "red", "like": "green", "cats": "blue", "dogs": "orange"}
    for i, word in enumerate(vocab):
        x, y = embed_2d[i, 0], embed_2d[i, 1]
        ax1.scatter(x, y, c=colors[word], marker='o')
        ax1.text(x + 0.05, y + 0.05, word, fontsize=9)
    ax1.axhline(0, color='gray', linewidth=0.5)
    ax1.axvline(0, color='gray', linewidth=0.5)
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    # 子图2: 输入序列各词的 Q 和 K 向量二维投影
    title_words = " ".join(vis_input_words)
    ax2.set_title(f"Q/K Vectors 2D (for {title_words})")
    for i, word in enumerate(vis_input_words):
        x_q, y_q = Q_2d[i, 0], Q_2d[i, 1]
        x_k, y_k = K_2d[i, 0], K_2d[i, 1]
        # Q 用蓝色方块, K 用红色圆圈表示
        ax2.scatter(x_q, y_q, c='C0', marker='s')
        ax2.text(x_q + 0.03, y_q + 0.03, f"Q_{word}", color='C0', fontsize=8)
        ax2.scatter(x_k, y_k, c='C3', marker='o')
        ax2.text(x_k + 0.03, y_k + 0.03, f"K_{word}", color='C3', fontsize=8)
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    # 子图3: Q·K^T （缩放因子 √d）的得分矩阵热图
    scores_matrix = np.dot(Q_mat, K_mat.T) / (embed_dim ** 0.5)  # 计算 QK^T/√d
    sns.heatmap(scores_matrix, ax=ax3, cmap="vlag", center=0,
                xticklabels=vis_input_words, yticklabels=vis_input_words,
                vmin=-max(abs(scores_matrix.min()), abs(scores_matrix.max())),
                vmax= max(abs(scores_matrix.min()), abs(scores_matrix.max())),
                annot=True, fmt=".2f")
    ax3.set_title("Q·K^T/√d scores")
    # 子图4: 注意力权重矩阵热图 (softmax 之后)
    sns.heatmap(attn, ax=ax4, cmap="Blues", vmin=0, vmax=1,
                xticklabels=vis_input_words, yticklabels=vis_input_words,
                annot=True, fmt=".2f")
    ax4.set_title("Attention Weights")
    # 调整布局并保存当前轮次的可视化结果
    plt.tight_layout()
    plt.savefig(f"attention_vis_epoch_{epoch+1}.png")
    plt.close(fig)
    # 提示当前轮次可视化已保存
    print(f"Saved attention_vis_epoch_{epoch+1}.png")
