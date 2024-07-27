import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

#import time
#start = time.time()

train = 'C:/Users/sange/Downloads/archive/train'
test = 'C:/Users/sange/Downloads/archive/test'

img_width, img_height, channels = 224, 224, 3
batch_size = 1024

nb_train_samples = 28709
nb_test_samples = 7178

data_transforms = {
    'train': transforms.Compose([

        transforms.Resize((img_width, img_height)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.RandomVerticalFlip(p=0.005),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ]),
    'val': transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((img_width, img_height)),
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

image_datasets = {
    'train': datasets.ImageFolder(train, data_transforms['train']),
    'test': datasets.ImageFolder(test, data_transforms['test']),
}

# Create data loaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True, drop_last=True),
    'test': DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, drop_last=True),
}
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#def imshow(img):
   # img = img / 2 + 0.5     # unnormalize
  #  npimg = img.numpy()
 #   plt.imshow(np.transpose(npimg, (1, 2, 0)))
#    plt.show()


# get some random training images
#dataiter = iter(dataloaders['train'])
#images, labels = next(dataiter)

# show images
#imshow(torchvision.utils.make_grid(images[23]))