import torch.optim as optim
from model_trian import CircleDataset,CircleCenterNet
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#加载数据和模型
dataset = CircleDataset("circle_dataset")

model = CircleCenterNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

#加载训练载数据集
train_loader =  DataLoader(dataset, batch_size=32, shuffle=True)

num_epochs = 250
loss_dict = []
for epoch in range(num_epochs):
     model.train()
     running_loss = 0.0
     for images,labels in train_loader:
         images,labels = images.to(device), labels.to(device)
         optimizer.zero_grad()
         outputs = model(images)
         loss = criterion(outputs,labels)
         loss.backward()
         optimizer.step()
         running_loss += loss.item()
     print(f"Epoch[{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
     loss_dict.append((running_loss/len(train_loader)))

torch.save(model.state_dict(),f"model_best_{min(loss_dict)}.pth")


