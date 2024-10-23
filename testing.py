import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Define a Convolutional Neural Network
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) 
        self.fc1 = torch.nn.Linear(128 * 3 * 3, 128) 
        self.fc2 = torch.nn.Linear(128, 10)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2)) 
        x = x.view(-1, 128 * 3 * 3)  
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
model.load_state_dict(torch.load('cnn_model.pth', weights_only=True))
model.eval()

def classify_image(image):
    if isinstance(image, str):
        img = Image.open(image)
        img = img.convert('L')
        img = img.resize((28, 28))
        img = np.array(img)
    else:
        img = image

    img = img / 255.0
    img = (img - 0.5) * 2
    img = img.flatten().reshape(1, 1, 28, 28)
    img = torch.tensor(img, dtype=torch.float32)

    with torch.no_grad():
        logps = model(img)
        ps = torch.exp(logps)
        _, top_class = ps.topk(1, dim=1)
        return top_class.item()
