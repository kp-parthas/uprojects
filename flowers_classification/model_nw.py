#!/usr/bin/env python3

# Imports
from collections import OrderedDict

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import utils

def get_data_loaders(data_dir):
    # Define transforms for the training, validation, and testing sets
    train_transform = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    valid_test_transform = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])
    
    # Setup dir paths
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_test_transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=valid_test_transform)

    # Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    return train_loader, valid_loader, test_loader, train_dataset.class_to_idx

def get_network(input_size, output_size, hidden_layers, drop_p=0.5):
    layers = OrderedDict()

    layers['fc0'] = nn.Linear(input_size, hidden_layers[0])
    layers['relu0'] = nn.ReLU()
    layers['drop0'] = nn.Dropout(drop_p)
    
    for i, (h1,h2) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
        layers['fc' + str(i+1)] = nn.Linear(h1, h2)
        layers['relu' + str(i+1)] = nn.ReLU()
        layers['drop' + str(i+1)] = nn.Dropout(drop_p)

    layers['output'] = nn.Linear(hidden_layers[-1], output_size)
    layers['softmax'] = nn.LogSoftmax(dim=1)
    return nn.Sequential(layers)

def get_resnet_model(hidden_layers, output_size, drop_p):
    model = models.resnet152(pretrained=True)
    input_size = model.fc.in_features
    model.fc = get_network(input_size, output_size, hidden_layers, drop_p)
    return model

def get_densenet_model(hidden_layers, output_size, drop_p):
    model = models.densenet161(pretrained=True)
    input_size = model.classifier.in_features
    model.classifier = get_network(input_size, output_size, hidden_layers, drop_p)
    return model

def setup_pretrained_model(arch, hidden_layers, output_size, drop_p=0.1):
    if arch == "resnet":
        model = get_resnet_model(hidden_layers, output_size, drop_p)
        classifier = model.fc
    elif arch == "densenet":
        model = get_densenet_model(hidden_layers, output_size, drop_p)
        classifier = model.classifier
    else:
        print ("Error: Unsupported arch {}".format(arch))
        exit(1)

    return model, classifier

def get_criterion_optimizer(classifier, learning_rate):
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(classifier.parameters(), learning_rate)
    return criterion, optimizer

# Get validation loss and accuracy
def validation(model, valid_loader, criterion, device='cpu'):
    valid_loss = 0
    accuracy = 0
    for images, labels in valid_loader:

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        valid_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return (valid_loss/len(valid_loader), accuracy/len(valid_loader))

# Is gpu available
def is_gpu_available():
    return torch.cuda.is_available()

# Train model
def train_model(model, train_loader, valid_loader, epochs, print_every, criterion, optimizer, device='cpu'):
    epochs = epochs
    print_every = print_every
    steps = 0

    # setup device as cpu or cuda
    model.to(device)

    for e in range(epochs):
        model.train()
        
        running_loss = 0
        for ii, (inputs, labels) in enumerate(train_loader):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every))
                
                running_loss = 0
        
        # Validation Loss and Accuracy
        model.eval()
        with torch.no_grad():
            valid_loss, valid_accuracy = validation(model, valid_loader, criterion, device='cuda')
            print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Validation Loss: {:.4f}, Accuracy: {:.4f}".format(valid_loss, valid_accuracy))
        model.train()

def check_accuracy(model, test_loader, device='cpu'):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (total, correct)

# Save the checkpoint
def save_checkpoint(filepath, model, arch, epochs, hidden_layers, drop_p=0.1):
    model.cpu()
    checkpoint = {'arch': arch,
                  'epochs': epochs,
                  'hidden_layers': hidden_layers,
                  'drop_p': drop_p,
                  'class_to_idx': model.class_to_idx,
                  'state_dict': model.state_dict()}

    torch.save(checkpoint, filepath)

# Loads checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model, _ = setup_pretrained_model(checkpoint['arch'],
                                      checkpoint['hidden_layers'],
                                      len(checkpoint['class_to_idx']),
                                      checkpoint['drop_p'])

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def predict(model, image_path, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Load the image and process it
    np_image = utils.process_image(image_path)
    image = torch.from_numpy(np_image)
    image = image.unsqueeze(0).float()
    
    # Predict the class
    model.eval()
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        output = model.forward(image)
        probs, labels = torch.topk(output, topk)
        probs = probs.exp().cpu().numpy()[0]
        labels = labels.cpu().numpy()[0]

    idx_to_class = {model.class_to_idx[x]:x for x in model.class_to_idx}
    classes = [idx_to_class[x] for x in labels]

    return probs, classes
