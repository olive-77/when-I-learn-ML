import time
import torch
import intel_extension_for_pytorch as ipex

def monitor_xpu(interval=3):
    if not torch.xpu.is_available():
        print("未检测到XPU设备")
        return
    
    device = 0  # 你的XPU设备ID
    print(f"监控XPU设备 {device} ({torch.xpu.get_device_name(device)})...\n")
    
    while True:
        # 多种内存统计方式（不同场景适用）
        allocated = torch.xpu.memory_allocated(device) / (1024**2)  # 显式分配的内存
        cached = torch.xpu.memory_reserved(device) / (1024**2)      # 缓存的内存（包括未使用的预留内存）
        
        print(f"=== 内存状态 ===")
        print(f"显式分配: {allocated:.2f} MB")
        print(f"缓存内存: {cached:.2f} MB")
        print(f"总占用: {allocated + cached:.2f} MB\n")
        
        time.sleep(interval)

if __name__ == "__main__":
    try:
        monitor_xpu()
    except KeyboardInterrupt:
        print("监控停止")
