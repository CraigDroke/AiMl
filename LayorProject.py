import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms, utils
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
#import scipy.misc
from PIL import Image
import json
import gradio as gr
import os

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=0., std=1.)
])

def generate_feature_maps(im):
    image = Image.fromarray(im, 'RGB')
    plt.imshow(image)

    model = models.resnet18(pretrained=True)
    print(model)

    # we will save the conv layer weights in this list
    model_weights =[]
    #we will save the 49 conv layers in this list
    conv_layers = []
    # get all the model children as list
    model_children = list(model.children())
    #counter to keep count of the conv layers
    counter = 0
    #append all the conv layers and their respective wights to the list
    for i in range(len(model_children)):
        if type(model_children[i]) == nn.Conv2d:
            counter+=1
            model_weights.append(model_children[i].weight)
            conv_layers.append(model_children[i])
        elif type(model_children[i]) == nn.Sequential:
            for j in range(len(model_children[i])):
                for child in model_children[i][j].children():
                    if type(child) == nn.Conv2d:
                        counter+=1
                        model_weights.append(child.weight)
                        conv_layers.append(child)
    print(f"Total convolution layers: {counter}")
    print("conv_layers")

    device = torch.device('cpu')
    model = model.to(device)

    image = transform(image)
    print(f"Image shape before: {image.shape}")
    image = image.unsqueeze(0)
    print(f"Image shape after: {image.shape}")
    image = image.to(device)

    outputs = []
    names = []
    for layer in conv_layers[0:]:
        image = layer(image)
        outputs.append(image)
        names.append(str(layer))
    print(len(outputs))
    #print feature_maps
    for feature_map in outputs:
        print(feature_map.shape)

    processed = []
    for feature_map in outputs:
        feature_map = feature_map.squeeze(0)
        gray_scale = torch.sum(feature_map,0)
        gray_scale = gray_scale / feature_map.shape[0]
        processed.append(gray_scale.data.cpu().numpy())
    for fm in processed:
        print(fm.shape)

    # Plot and save feature maps for each layer
    for i, (fm, name) in enumerate(zip(processed, names)):
        fig = plt.figure(figsize=(10, 10))
        a = fig.add_subplot(1, 1, 1)  # You should adjust the layout as needed
        imgplot = plt.imshow(fm, cmap='viridis')  # Adjust the colormap if needed
        a.axis("off")
        filename = f'layor{i}.jpg'
        plt.savefig('C:\\Users\\cdrok\\Documents\\JuniorClinic\\AiMl\\layors\\' + filename, bbox_inches='tight')
        plt.close(fig)  # Close the figure after saving

# Gradio interface
with gr.Blocks() as demo:
    
    layorNumber = gr.Slider(0, 16, value=4, label="Count", info="Choose between 0 and 16", step=1)

    with gr.Row():
        im = gr.Image()
        im2 = gr.Image(type= 'filepath',)

    def show_feature_maps(im, layorNumber):
        # Future if check for if all layors exist to run faster
        generate_feature_maps(im)
        this_path = 'C:\\Users\\cdrok\\Documents\\JuniorClinic\\AiMl\\layors\\layor' + str(int(layorNumber)) + '.jpg'
        return this_path

    btn = gr.Button(value="Generate Feature Maps")
    btn.click(show_feature_maps, inputs=[im, layorNumber], outputs=[im2])

if __name__ == "__main__":
    demo.launch()
    




