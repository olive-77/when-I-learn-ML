import torch
import torch.nn as nn
import torch.optim as optim
from  torchvision import datasets, transforms
from torch.utils.data import random_split

#from LR.debug_package import running_loss

device = torch.device("xpu")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d ( 1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 =nn. Conv2d ( 6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.linear = nn.Linear ( 16 *4*4, 48)
        self.linear2 = nn.Linear(48, 10)
    def forward(self, input):
        x = self.pool1(torch.relu(self.conv1(input)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x=x.view(-1, 16 * 4 * 4)
        x=torch.relu(self.linear(x))
        x=self.linear2(x)
        return x
model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 4

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_dataset ,compare_dataset = random_split(train_dataset, [56000, 4000])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
compare_loader = torch.utils.data.DataLoader(compare_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)
running_loss=0.0
best_loss=10000
best=0
for epoch in range(epochs):
    model.train()
    for i , data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 0:
            print('epoch=',epoch,'batch=',i,'loss=',running_loss / 100)
            running_loss = 0.0
    running_loss=0.0
    model.eval()
    with torch.no_grad():
        for data in compare_loader:
            inputs, labels = data
            inputs=inputs.to(device)
            labels=labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
        if running_loss < best_loss:
            best_loss = running_loss
            torch.save(model.state_dict(), './model.pth')
            best =epoch+1
#next:test the point
model.load_state_dict(torch.load('./model.pth'))
tot=0
s=0
with torch.no_grad():
    for data in test_loader:
        inputs, labels = data
        inputs=inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        tot+=labels.size(0)
        predicted=predicted.cpu().numpy()
        s+=(predicted==labels).sum().item()

print(best)
print ("模型准确度：",s/tot)


print("模型参数设备:", next(model.parameters()).device)