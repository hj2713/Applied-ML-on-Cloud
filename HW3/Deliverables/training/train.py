import os
import logging
from dotenv import load_dotenv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# Load environment variables from .env file (for local development)
# In Kubernetes, these will be injected via ConfigMaps and Secrets instead
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Convolutional Neural Network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 320)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def train():
    logger.info("Downloading and loading MNIST dataset...")
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Allow batch size to be configured via environment variables
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    # Load dataset into the Deliverables folder (or container path)
    data_dir = os.environ.get('DATA_DIR', 'Deliverables/data')
    train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")
    
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Allow epochs to be configured via environment variables (best practice for K8s Jobs)
    epochs = int(os.environ.get("EPOCHS", "2"))
    logger.info(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 200 == 0:
                logger.info(f"Epoch {epoch}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    model_dir = os.environ.get("MODEL_DIR", "./model_data")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "mnist_model.pt")
    torch.save(model.state_dict(), model_path)
    logger.info(f"Training complete! Model saved successfully to: {model_path}")

if __name__ == "__main__":
    train()
