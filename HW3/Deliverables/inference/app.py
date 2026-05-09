import os
import io
import logging
from functools import wraps
from dotenv import load_dotenv
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, render_template
from PIL import Image
from torchvision import transforms

# Load environment variables from .env file (for local development)
# In Kubernetes, these will be injected via ConfigMaps and Secrets instead
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

# Environment-driven Configuration (Enterprise Best Practice)
API_KEY = os.environ.get("API_KEY", "dev-secret-key")

# Security Decorator for Authentication
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        provided_key = request.headers.get("X-API-Key")
        if provided_key and provided_key == API_KEY:
            return f(*args, **kwargs)
        else:
            logger.warning(f"Unauthorized access attempt. IP: {request.remote_addr}")
            return jsonify({"error": "Unauthorized. Invalid or missing X-API-Key header."}), 401
    return decorated_function

# Declaring the same model architecture to load the weights
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

# Look for the model in the same place training saved it.
# In Kubernetes, both containers mount the same PVC here.
model_dir = os.environ.get("MODEL_DIR", "./model_data")
model_path = os.path.join(model_dir, "mnist_model.pt")

model = SimpleCNN()
if os.path.exists(model_path):
    # Load model mapping to CPU in case it was trained on GPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    logger.info(f"Model loaded successfully from {model_path}.")
else:
    logger.error(f"Model not found at {model_path}. Please run train.py first!")

# Standard MNIST image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
@require_api_key
def predict():
    # Ensure an image was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Read the image file directly from memory
        image = Image.open(io.BytesIO(file.read()))
        # Convert to tensor and add a batch dimension (B, C, H, W)
        tensor = transform(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            output = model(tensor)
            # Find the index of the highest probability
            prediction = output.argmax(dim=1, keepdim=True).item()
            
        return jsonify({
            "message": "Inference successful",
            "prediction": prediction
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    # A real health check: verify the model is actually loaded in memory
    if os.path.exists(model_path):
        return jsonify({
            "status": "healthy",
            "model_loaded": True
        }), 200
    else:
        # If the model isn't found, tell Kubernetes we are not ready for traffic
        return jsonify({
            "status": "unhealthy",
            "error": "Model file not found. Waiting for training job."
        }), 503

if __name__ == '__main__':
    # Listen on all network interfaces to allow K8s to route traffic to us
    port = int(os.environ.get("PORT", "50000"))
    app.run(host='0.0.0.0', port=port)
