import torch
from debug import Rotator

# 设置随机种子，确保结果固定
torch.manual_seed(0)

# 1. 简化参数（更小的维度和序列长度）
batch_size = 1  # 仅1个样本
num_heads = 1   # 仅1个注意力头
seq_len = 2     # 仅2个序列位置
hidden_dim = 4  # 更小的隐藏维度（必须为偶数）

# 2. 生成固定的输入数据（而非随机数）
# 输入向量设为简单固定值：[[[1,2,3,4], [5,6,7,8]]]
x = torch.tensor([
    [
        [1.0, 2.0, 3.0, 4.0],  # 位置0的向量
        [5.0, 6.0, 7.0, 8.0]   # 位置1的向量
    ]
]).unsqueeze(0)  # 增加batch维度，形状变为[1, 1, 2, 4]

# 位置ID固定为[0, 1]
position_ids = torch.arange(seq_len)  # [0, 1]

# 3. 创建Rotator实例
rotator = Rotator(D=hidden_dim, position_ids=position_ids)

# 4. 执行旋转操作
rotated_x = rotator.rotate(x)

# 5. 输出结果（更详细的对比）
print("=== 输入数据 ===")
print("输入形状:", x.shape)
print("位置0的向量:", x[0, 0, 0])  # [1,2,3,4]
print("位置1的向量:", x[0, 0, 1])  # [5,6,7,8]

print("\n=== 旋转参数 ===")
print("位置0的cos值:", rotator.cos[0])
print("位置0的sin值:", rotator.sin[0])
print("位置1的cos值:", rotator.cos[1])
print("位置1的sin值:", rotator.sin[1])

print("\n=== 旋转结果 ===")
print("旋转后形状:", rotated_x.shape)
print("位置0旋转后:", rotated_x[0, 0, 0])
print("位置1旋转后:", rotated_x[0, 0, 1])