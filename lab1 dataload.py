import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os

# 数据集的根目录
data_root = 'path_to_your_image_folder'

# 训练集数据增强
transform_enhanced = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集和测试集数据预处理
transform_standard = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 使用ImageFolder加载数据
dataset = datasets.ImageFolder(root=data_root)

# 划分数据集为训练集和测试集
total_count = len(dataset)
train_count = int(0.8 * total_count)
test_count = total_count - train_count
train_dataset, test_dataset = random_split(dataset, [train_count, test_count])

# 应用数据增强到训练集，验证和测试集使用标准转换
train_dataset.dataset.transform = transform_enhanced
test_dataset.dataset.transform = transform_standard

# 加载数据
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 保存训练集和测试集
torch.save(train_dataset, 'train_dataset.pth')
torch.save(test_dataset, 'test_dataset.pth')

print("Datasets are successfully loaded and saved.")