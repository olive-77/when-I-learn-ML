import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 卷积层 1：输入通道 1（灰度图像），输出通道 6，卷积核大小 5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 池化层 1：采用最大池化，核大小 2x2，步长 2
        self.pool1 = nn.MaxPool2d(2, 2)
        # 卷积层 2：输入通道 6，输出通道 16，卷积核大小 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 池化层 2：采用最大池化，核大小 2x2，步长 2
        self.pool2 = nn.MaxPool2d(2, 2)
        # 全连接层 1：输入特征数 16*4*4，输出 120
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        # 全连接层 2：输入 120，输出 84
        self.fc2 = nn.Linear(120, 84)
        # 全连接层 3：输入 84，输出 10（对应 0-9 十个数字）
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        # 第一层卷积 + 池化 + ReLU 激活
        x = self.pool1(torch.relu(self.conv1(x)))
        # 第二层卷积 + 池化 + ReLU 激活
        x = self.pool2(torch.relu(self.conv2(x)))
        # 调整张量形状，以便输入全连接层
        x = x.view(-1, 16 * 4 * 4)
        # 全连接层 1 + ReLU 激活
        x = torch.relu(self.fc1(x))
        # 全连接层 2 + ReLU 激活
        x = torch.relu(self.fc2(x))
        # 全连接层 3
        x = self.fc3(x)
        return x


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
# 加载训练集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# 加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


# 实例化模型
model = LeNet()
# 损失函数：交叉熵损失
criterion = nn.CrossEntropyLoss()
# 优化器：随机梯度下降，学习率 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)
# 训练轮数
epochs = 5
for epoch in range(epochs):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # 梯度清零
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 优化器更新参数
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {running_loss / 100:.3f}')
            running_loss = 0.0
print('Training finished!')


correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
accuracy = 100 * correct / total
print(f'Test Accuracy: {accuracy:.2f}%')