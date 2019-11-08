import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np
from torch.autograd import Variable

from Discriminator import Discriminator

batch_size = 128
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    transforms.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

model = Discriminator()

model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Begin training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

total_epochs = 100

train_loss = []
train_accu = []
test_accu = []

print("Beginning Training...")

for epoch in range(total_epochs):  # loop over the dataset multiple times
    if(epoch==50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/10.0
    if(epoch==75):
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate/100.0


    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0
    time1 = time.time()

    running_loss = 0.0
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):

        if(Y_train_batch.shape[0] < batch_size):
            continue

        X_train_batch = Variable(X_train_batch).cuda()
        Y_train_batch = Variable(Y_train_batch).cuda()
        _, output = model(X_train_batch)

        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()

        loss.backward()
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'step' in state.keys():
                    if state['step'] >= 1024:
                        state['step'] = 1000
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if batch_idx % 50 == 49:    # print every 50 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, batch_idx + 1, running_loss / (50*batch_size)))
            running_loss = 0.0
        epoch_counter += batch_size
        epoch_loss +=loss.data[0]
        _,predicted = torch.max(output.data,1)
        epoch_acc += (predicted == Y_train_batch)

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))

    # Begin testing accuracy
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()

    running_loss = 0.0
    with torch.no_grad:
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):

            if(Y_test_batch.shape[0] < batch_size):
                continue

            X_test_batch = Variable(X_test_batch).cuda()
            Y_test_batch = Variable(Y_test_batch).cuda()
            _, output = model(X_test_batch)

            loss = criterion(output, Y_test_batch)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if batch_idx % 50 == 49:    # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, batch_idx + 1, running_loss / 50))
                running_loss = 0.0
            epoch_counter += batch_size
            epoch_loss +=loss.data[0]
            _,predicted = torch.max(output.data,1)
            epoch_acc += (predicted == Y_test_batch)

    print(epoch_counter,epoch_acc)
    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter/batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print(epoch, "%.2f" % (epoch_acc*100.0), "%.4f" % epoch_loss, "%.4f" % float(time.time()-time1))



print('Finished Training')

torch.save(model,'cifar10.model')

# Evaluate accuracy

# correct = 0
# total = 0
# with torch.no_grad():
#     for data in testloader:
#         images, labels = data
#         outputs = net(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
# print('Accuracy of the network on the 10000 test images: %d %%' % (
#     100 * correct / total))
