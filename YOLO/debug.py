import torch
import intel_extension_for_pytorch as ipex
##
import subprocess
# 目标 Conda 环境名称
conda_env = "env_name"
# 步骤1：激活 Conda 环境的命令（不同系统略有差异，这里以 Linux/macOS 为例）
# Windows 下可改为：f'conda activate {conda_env} && python train.py'
activate_cmd = f'conda activate {conda_env} && python train.py'
# 步骤2：执行命令（通过 shell 方式执行，才能解析 && 等符号）
subprocess.run(activate_cmd, shell=True, check=True)



if hasattr(torch, 'xpu') and torch.xpu.is_available() :
    print('XPU available')
'''
print(f"PyTorch版本: {torch.__version__}")
print(f"IPEX版本: {ipex.__version__}")
print(f"CUDA是否可用? {torch.cuda.is_available()}") # 这行会是False，正常
print(f"XPU是否可用? {torch.xpu.is_available()}")   # 这行必须是True！

if torch.xpu.is_available():
    device = torch.device("xpu")
    print(f"使用的设备: {device}")
    print(f"GPU名称: {torch.xpu.get_device_name(0)}")
    print(f"当前GPU内存使用情况: {torch.xpu.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"总GPU内存: {torch.xpu.get_device_properties(0).total_memory / 1024**3:.2f} GB")
else:
    print("很遗憾，英特尔GPU未就绪。请检查驱动和库的安装。")
'''