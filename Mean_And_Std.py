import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# Transformers
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomAffine(degrees = 10, translate = (0.05,0.05), shear = 5),
    transforms.ColorJitter(hue = .05, saturation = .05),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Loaders
dataset = ImageFolder(root='./dataset', transform=transform)
loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

# Calculating the mean and standard deviation for our dataset
def get_mean_std(loader):
    channels_sum, channels_squared_sum, num_batches = 0,0,0

    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0,2,3])
        channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
        num_batches +=1

    mean = channels_sum/num_batches
    std = (channels_squared_sum/num_batches-mean**2)**0.5
    return mean, std

mean, std = get_mean_std(loader)

print(mean)
print(std)