# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import os

model = models.vgg19().features
print(model)

conv_layers = ['0', '5', '10', '19', '28']

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.chosen_features = conv_layers
        self.model = models.vgg19().features[:29]

    def forward(self, x):
        features = []

        for index, layer in enumerate(self.model):
            x = layer(x)
            if str(index) in self.chosen_features:
                features.append(x)

        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


# a bunch of MPS stuff isn't supported :(
device = torch.device("mps" if torch.has_mps else "cpu")
# device = torch.device("cpu")
image_size = 512

loader = transforms.Compose([
    transforms.CenterCrop((image_size, image_size)),
    transforms.ToTensor()
    # transforms.Normalize(mean=[], std=[])
])

model = VGG().to(device).eval()

original_image = load_image("white.png")
style_image = load_image("style.png")

# generated_image = torch.randn(size=original_image.shape, device=device, requires_grad=True)
generated_image = original_image.clone().requires_grad_(True)

# Hyperparameters
steps = 4000
learning_rate = 0.01
alpha = 0.2
beta = 1

optimizer = optim.Adam([generated_image], lr=learning_rate)

epoch = 1

original_features = model(original_image)
style_features = model(style_image)

for step in range(steps):
    generated_features = model(generated_image)

    style_loss = original_loss = 0

    for gen_feature, orig_feature, style_feature in zip(generated_features, original_features, style_features):
        batch_size, channel, height, width = gen_feature.shape
        original_loss += torch.mean((gen_feature - orig_feature) ** 2)

        # Compute Gram Matrix
        G = gen_feature.view(channel, height * width).mm(
            gen_feature.view(channel, height*width).t())

        A = style_feature.view(channel, height*width).mm(
            style_feature.view(channel, height*width).t())

        style_loss += torch.mean((G-A) ** 2)

    total_loss = alpha*original_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(total_loss)
        save_image(generated_image, os.getcwd() + F"/GeneratedImages/generated{epoch}.png")
        epoch += 1

# Press the green button in the gutter to run the script.
# if __name__ == '__main__':
#     print("Done executing")
