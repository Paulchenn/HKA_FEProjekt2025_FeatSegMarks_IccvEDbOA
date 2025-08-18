import torch
from models import generation_imageNet

def debug_forward(model, x, device='cuda'):
    model.eval()  # Eval-Modus
    x = x.to(device)
    
    print(f"Input: {x.shape}, device: {x.device}, mem: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    
    with torch.no_grad():
        for name, module in model.named_children():
            x = module(x)
            mem = torch.cuda.memory_allocated() / 1e6
            print(f"{name:30} -> {x.shape}, device: {x.device}, mem: {mem:.1f} MB")
    
    return x

netG    = generation_imageNet.generator(128, img_size=256).to('cuda')
sample_input = torch.randn(1, 3, 224, 224)  # Beispielinput
output = debug_forward(model, sample_input)
