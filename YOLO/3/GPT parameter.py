import torch

# 1. 加载权重
model_weights = torch.load('model.pth')
for name, param in model_weights.items():
    # 打印参数名 + 参数形状（避免打印海量数值）
    print(f"参数名：{name:50} | 形状：{param.shape}")