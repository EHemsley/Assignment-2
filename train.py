#Importing packages
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch import optim
from collections import OrderedDict
import argparse    
from get_input_args_train import get_input_args

# Main program function defined below
in_arg = get_input_args()

#Data directory
data_dir = in_arg.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    
    
#Image transformations
train_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])

valid_transforms =  transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485,0.456,0.406],
                                                                 [0.229,0.224,0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])])

# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
test_data = datasets.ImageFolder(test_dir,transform=test_transforms)

# Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)


#Label mapping
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Use GPU if it's available
gpu = in_arg.gpu
if gpu == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
else:
    device = torch.device("cpu")
    
#Choose architecture
arch = in_arg.arch
vgg16 = models.vgg16(pretrained=True)
densenet121 = models.densenet121(pretrained=True)
models = {'densenet121': densenet121, 'vgg16': vgg16}
model = models[arch]
input_sizes = {"vgg16":25088,"densenet121":1024}
input_size = input_sizes[arch]


# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False  

#Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
hidden_units = in_arg.hidden_units
learning_rate = in_arg.learning_rate
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(0.5)),
                          ('fc2', nn.Linear(hidden_units, 102)),    
                          ('output', nn.LogSoftmax(dim=1))
                          ]))


model.classifier = classifier
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
model.to(device);
    
print("Training model...")    
    
#Training model with training and validation datasets
epochs = in_arg.epochs
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps+=1
        inputs, labels = inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Valid loss: {test_loss/len(validloader):.3f}.. "
                  f"Valid accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
 
# TODO: Save the checkpoint 
model.class_to_idx = train_data.class_to_idx

checkpoint = {'arch': arch,
              'classifier': model.classifier,
              'input_size': input_size,
              'output_size': 102,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'epochs': epochs,
              'optimizer': optimizer.state_dict()}

torch.save(checkpoint, 'checkpoint.pth')

            
# Call to main function to run the program
def main():
    warnings.filterwarnings("ignore")

if __name__ == "__main__":
    main()
