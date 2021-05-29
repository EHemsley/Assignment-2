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
from get_input_args_predict import get_input_args
from PIL import Image

# Main program function defined below
in_arg = get_input_args()

#Label mapping
cat_to_name = in_arg.category_names
import json
with open(cat_to_name, 'r') as f:
    cat_to_name = json.load(f)
    
# Use GPU if it's available
gpu = in_arg.gpu
if gpu == 'gpu':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
else:
    device = torch.device("cpu")
    
# Function that loads a checkpoint and rebuilds the model
save_dir = in_arg.save_dir
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.vgg16(pretrained=True)
    model = models.densenet121(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model

model = load_checkpoint(save_dir)
print(model)


#Image Preprocessing
def process_image(image):
    ''' Scales, crops, and normalizes a PIL    image for a PyTorch model,      returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image = Image.open(image)
    # Resize
    if image.size[0] > image.size[1]:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))
    # Crop 
    left = ((256 - 224)/2)
    top = ((256 - 224)/2)
    right = ((256 + 224)/2)
    bottom = ((256 + 224)/2)
    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    # Normalize
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225]) 
    image = (image - mean)/std
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose((2, 0, 1))
    
    return torch.from_numpy(image)



#Class prediction
topk =in_arg.topk
def predict(image_path, model, topk=topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    img = process_image(image_path)
    img = img.numpy()
    img = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze_(0)
    img = img.to(device)
    
    with torch.no_grad():

        logps = model.forward(img)
        ps = torch.exp(logps)
        top_probs, top_class = ps.topk(topk , dim = 1)

    #change to lists    
    topk_ps = top_probs.tolist()[0]
    classes =top_class.squeeze(0).tolist()
    
    #iterate
    idx_to_class = {value: key for key , value in model.class_to_idx.items()}
    class_to_idx = model.class_to_idx
    topk_class = [idx_to_class[i] for i in classes]
    return topk_ps, topk_class

#Uses matplotlib to plot the probabilities for the top 5 classes as a bar graph, along with the input image.
image_path = in_arg.image_path
img=Image.open(image_path)
flower_number = image_path.split('/')[2]
flower_name = cat_to_name[flower_number]
top_probs,top_classes = predict(image_path, model)
class_names = [cat_to_name[i] for i in top_classes]

print('Flower name is: {}'.format(flower_name.title()))
print('Top {} classes : {}, Top {} probabilities : {}' .format(topk, class_names, topk, top_probs))

# Call to main function to run the program
def main():
    in_arg

if __name__ == "__main__":
    main()
