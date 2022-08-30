# ----- Imports -----
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split
from AlexNet import AlexNet
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



# ----- Variables & Parameters -----
epochs = 50
num_classes = 4
learning_rate = 0.005
dataset_root = "./dataset"
training_set_size = 1200
testing_set_size = 400
batch_size = 32
device = torch.device("cuda")
img_size = 227
classes = ["Cloth", "N95", "NoMask", "Surgical"]

# ----- Initialize the transformation configuration -----
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomAffine(degrees = 10, translate = (0.05,0.05), shear = 5),
    transforms.ColorJitter(hue = .05, saturation = .05),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5287, 0.4877, 0.4666), (0.3204, 0.3142, 0.3182))
])

# ----- Splitting dataset into training and testing sets -----
dataset = ImageFolder(root=dataset_root, transform=transform)
training_set, testing_set = random_split(dataset, [training_set_size, testing_set_size])


# ----- Load Data -----
train_loader = DataLoader(training_set, batch_size, shuffle=True)
test_loader = DataLoader(testing_set, batch_size, shuffle=False)



# ----- Loading The mode arichetecture -----
model = AlexNet()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9)

# Creating variables to store stats
epoch_log = []
loss_log = []
accuracy_log = []
pred_list = torch.zeros(0, dtype=torch.long, device='cpu')
label_list = torch.zeros(0, dtype=torch.long, device='cpu')

# ---------- Starting the Training ------
print('----- Starting training... -----')

for epoch in range(epochs):
    print(f'Starting Epoch: {epoch + 1}...')

    # We keep adding or accumulating our loss after each mini-batch in running_loss
    running_loss = 0.0

    for i, data in enumerate(train_loader):

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero grad
        optimizer.zero_grad()

        # Forward -> backprop + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Print Training statistics - Epoch/Iterations/Loss/Accuracy
        running_loss += loss.item()
        if i % 4 == 3:
            correct = 0
            total = 0


            with torch.no_grad():
                # Iterate through the testloader iterator
                for data in test_loader:
                    images, labels = data
                    # Move our data to GPU
                    images = images.to(device)
                    labels = labels.to(device)

                    # Foward propagate our test data batch through our model
                    outputs = model(images)

                    # Get predictions from the maximum value of the predicted output tensor
                    _, predicted = torch.max(outputs.data, dim=1)
                    # Keep adding the label size or length to the total variable
                    total += labels.size(0)
                    # Keep a running total of the number of predictions predicted correctly
                    correct += (predicted == labels).sum().item()

                accuracy = 100 * correct / total
                epoch_num = epoch + 1
                actual_loss = running_loss / 50
                print(
                    f'Epoch: {epoch_num}, Mini-Batches Completed: {(i + 1)}, Loss: {actual_loss:.3f}, Test Accuracy = {accuracy:.3f}%')
                running_loss = 0.0

                # Store training stats after each epoch
                epoch_log.append(epoch_num)
                loss_log.append(actual_loss)
                accuracy_log.append(accuracy)
                pred_list = torch.cat([pred_list, predicted.view(-1).cpu()])
                label_list = torch.cat([label_list, labels.view(-1).cpu()])


print('Finished Training')
PATH = './models/alexnet_mask_cnn_50ep.pth'
torch.save(model.state_dict(), PATH)
print('Model Saved')

#----- Printing the Confusion Matrix
def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


conf_mat = confusion_matrix(label_list.numpy(), pred_list.numpy())
plot_confusion_matrix(conf_mat, classes)

#---- Printing the Stats
print(classification_report(label_list.numpy(), pred_list.numpy()))


