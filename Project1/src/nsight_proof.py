import torch
import torchvision.models as models

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        return

    device = torch.device('cuda')
    model = models.resnet50().to(device).half()
    inputs = torch.randn(32, 3, 224, 224, device=device, dtype=torch.float16)
    labels = torch.randint(0, 1000, (32,), device=device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    print("Running 10 simple iterations...")
    for _ in range(10):
        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
    
    torch.cuda.synchronize()
    print("Nsight Proof Run Complete.")

if __name__ == '__main__':
    main()
