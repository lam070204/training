import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import cv2
import threading
import ipywidgets as widgets
from jetcam.csi_camera import CSICamera
from xy_dataset import XYDataset
from utils import preprocess, bgr8_to_jpeg

# Camera setup
camera = CSICamera(width=224, height=224)
camera.running = True

# Transformations
TRANSFORMS = transforms.Compose([
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset setup
dataset = XYDataset("road_following", ['apex'], TRANSFORMS, random_hflip=True)

# Model setup
device = torch.device('cuda')
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(512, 2)  # Output (x, y)
model = model.to(device)
optimizer = optim.Adam(model.parameters())

# Training function
def train(epochs):
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, _, xy in train_loader:
            images, xy = images.to(device), xy.to(device)
            optimizer.zero_grad()
            loss = torch.mean((model(images) - xy) ** 2)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

# Live prediction function
def live():
    while state_widget.value == 'live':
        image = preprocess(camera.value)
        output = model(image).detach().cpu().numpy().flatten()
        x, y = int(224 * (output[0] / 2 + 0.5)), int(224 * (output[1] / 2 + 0.5))
        prediction_widget.value = bgr8_to_jpeg(cv2.circle(camera.value, (x, y), 8, (255, 0, 0), 3))

# UI setup
state_widget = widgets.ToggleButtons(options=['stop', 'live'], description='State')
prediction_widget = widgets.Image(format='jpeg', width=224, height=224)
train_button = widgets.Button(description='Train')
epochs_widget = widgets.IntText(description='Epochs', value=1)
model_path_widget = widgets.Text(value='road_following_model.pth', description='Model Path')

# Button actions
def start_live(change):
    if change['new'] == 'live':
        threading.Thread(target=live).start()

def save_model(_):
    torch.save(model.state_dict(), model_path_widget.value)

def load_model(_):
    model.load_state_dict(torch.load(model_path_widget.value))

def train_model(_):
    train(epochs_widget.value)

# Event bindings
state_widget.observe(start_live, names='value')
train_button.on_click(train_model)
widgets.Button(description='Save').on_click(save_model)
widgets.Button(description='Load').on_click(load_model)

display(widgets.VBox([prediction_widget, state_widget, epochs_widget, train_button, model_path_widget]))
