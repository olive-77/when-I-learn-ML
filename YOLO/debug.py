import torch
import intel_extension_for_pytorch as ipex

# 查看是否有可用XPU
print("XPU可用数量：", torch.xpu.device_count())
# 尝试初始化XPU
try:
    device = torch.device("xpu:0")
    print("XPU初始化成功，设备：", device)
except Exception as e:
    print("XPU初始化失败：", e)