import json
import os

import torch
from torch import optim

from torchvision import transforms
from utils.get_ImageNet import main as get_ImageNet
from models import generation
from models.resnet import ResNet18, BasicBlock

# Path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the config directory
CONFIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'config')

# === Load parameters for training from config.json ===
with open(os.path.join(SCRIPT_DIR, 'config/config.json'), 'r') as f:
    config = json.load(f)
DEVICE = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
BATCH_SIZE = config["batch_size"]
EPOCHS = config["epochs"]
GEN_IN_DIM = config["generator_input_dim"]
PATIENCE = config["patience"]
DEBUG_MODE = config["debug_mode"]
DEBUG_ITERS_START = config["debugIterations_strt"]
DEBUG_ITERS_AMOUNT = config["debugIterations_amount"]
LR_GEN = config["learning_rate"]["generator"]
LR_DISC = config["learning_rate"]["discriminator"]
LR_CLS = config["learning_rate"]["classifier"]


if __name__ == "__main__":
    image_size = 128

    # Transformations for preprocessing training data
    transform_train = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Transformations for preprocessing test data
    transform_test = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # === Load the dataset (ImageNet) ===
    train_dataset, test_dataset = get_ImageNet(BATCH_SIZE)
    
    # Set best accuracy to 0 to be sure that first accuracy is better
    best_acc = 0

    # Resize-Operations
    re12 = transforms.Resize((12, 12))  # scales to 12x12
    re32 = transforms.Resize((32, 32))  # scales to 32x32

    # === Initialize networks (and move to CPU/GPU) ===
    netG    = generation.generator(GEN_IN_DIM).to(DEVICE)       # Generator network with input size GEN_IN_DIM
    netD    = generation.Discriminator().to(DEVICE)             # Discriminator to distinguish real/fake images
    cls     = ResNet18(BasicBlock, num_classes=10).to(DEVICE)   # Classifier (here: ResNet-18)

    # === Initialize optimizers ===
    optimG = optim.Adam(netG.parameters(), lr=LR_GEN, betas=(0., 0.99))     # Optimizer for Generator (GAN-typical Betas)
    optimD = optim.Adam(netD.parameters(), lr=LR_DISC, betas=(0., 0.99))    # Optimizer for Discriminator (GAN-typical Betas)
    optimC = optim.Adam(cls.parameters(), lr=LR_CLS, betas=(0., 0.99))      # Optimizer for Classifier (GAN-typical Betas)

    # === Initialize loss functions ===