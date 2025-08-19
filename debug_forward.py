import torch
from models import generation_imageNet
from torch.autograd import Variable

def debug_forward(model, x, y, z, device='cuda'):
    model.eval()  # Eval-Modus
    x = x.to(device)
    y = y.to(device)
    z = z.to(device)
    
    print(f"Input: {x.shape}, device: {x.device}, mem: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    
    with torch.no_grad():
        for name, module in model.named_children():
            x = module(x, y, z)
            mem = torch.cuda.memory_allocated() / 1e6
            print(f"{name:30} -> {x.shape}, device: {x.device}, mem: {mem:.1f} MB")
    
    return x

netG    = generation_imageNet.generator(128, img_size=256).to('cuda')
sample_input_1 = Variable(torch.randn((1, 100)).view(-1, 100, 1, 1))  # Beispielinput 1
sample_input_2 = torch.randn(1, 3, 256, 256)  # Beispielinput 2
sample_input_3 = torch.randn(1, 3, 256, 256)  # Beispielinput 3
output = debug_forward(netG, sample_input_1, sample_input_2, sample_input_3)
