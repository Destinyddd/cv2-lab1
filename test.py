import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import time
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms



data_dir = '/Localize/lc/homework/work1/dataset/CUB_200_2011/'
save_path = 'best_pr_model.pth'


data_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=data_transforms)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet18()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(test_dataset.classes))
model_ft = model_ft.to(device)

model_ft.load_state_dict(torch.load(save_path))
model_ft.eval()

def evaluate_model(model, dataloader):
    running_corrects = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
    
    accuracy = running_corrects.double() / total
    return accuracy.item()

# Evaluate the model on the test dataset
test_accuracy = evaluate_model(model_ft, test_dataloader)
print(f'Test Accuracy: {test_accuracy:.4f}')