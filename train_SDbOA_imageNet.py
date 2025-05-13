import json
import os

import torch
from torch import nn, optim

from torchvision import transforms
from utils.get_ImageNet import main as get_ImageNet
from models import generation
from models.resnet import ResNet18, BasicBlock
from utils.modules  import *

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

    # === Initialize Trainiingclasses ===
    emse    = EMSE()    # Initialize EMSE class
    tsd     = TSD()     # Initialize TSD class
    tsg     = TSG()     # Initialize TSG class

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
    
    # === Initialize networks (and move to CPU/GPU) ===
    netG    = generation.generator(GEN_IN_DIM).to(DEVICE)       # Generator network with input size GEN_IN_DIM
    netD    = generation.Discriminator().to(DEVICE)             # Discriminator to distinguish real/fake images
    cls     = ResNet18(BasicBlock, num_classes=10).to(DEVICE)   # Classifier (here: ResNet-18)

    # === Initialize optimizers ===
    optimG = optim.Adam(netG.parameters(), lr=LR_GEN, betas=(0., 0.99))     # Optimizer for Generator (GAN-typical Betas)
    optimD = optim.Adam(netD.parameters(), lr=LR_DISC, betas=(0., 0.99))    # Optimizer for Discriminator (GAN-typical Betas)
    optimC = optim.Adam(cls.parameters(), lr=LR_CLS, betas=(0., 0.99))      # Optimizer for Classifier (GAN-typical Betas)

    # === Initialize loss functions === 
    L1_loss = nn.L1Loss()                # Absoulte Error (e.g. for Reconstruction)
    CE_loss = nn.CrossEntropyLoss()      # Classification Loss (for multi-class outputs like CIFAR-10 Classes)

    ##### ===== TRAINING ===== #####
    # Initailisation of values for early stop
    no_improvement_counter = 0

    for epoch in range(EPOCHS):
        print(epoch)

        # === Training Phase ===
        netG.train()
        netD.train()
        cls.train()

        # Reduce learning rate after 60 epochs
        if epoch == 60:
            for param_group in optimG.param_groups:
                param_group['lr'] = LR_GEN / 10
            for param_group in optimD.param_groups:
                param_group['lr'] = LR_DISC / 10
            for param_group in optimC.param_groups:
                param_group['lr'] = LR_CLS / 10

        for i, (img, label) in enumerate(train_dataset):
            img     = img.to(DEVICE)
            label   = label.to(DEVICE)

            # === EMSE >>>
            edgeMap = emse.doEMSE(img)
            # <<< EMSE ===

            # === TSD >>>
            deformedImg = tsd.doTSD(img)
            # <<< TSD ===

            # === TSG >>>
            netD, netG, cls, optimD, optimG, optimC, CE_loss, L1_loss, loss_tot = tsg.doTSG(
                img,
                label,
                edgeMap,
                deformedImg,
                netD,
                netG,
                cls,
                optimD,
                optimG,
                optimC,
                CE_loss,
                L1_loss,
            )
            # <<< TSG ===

            # Print loss values after every 5 iterations
            if i % 5 == 0:
                print('Epoch[', epoch + 1, '/', EPOCHS, '][', i + 1, '/', len(train_dataset), ']: TOTAL_LOSS', loss_tot.item())

            # For debugging: Stop after a certain number of iterations
            if DEBUG_MODE and i >= DEBUG_ITERS_START + DEBUG_ITERS_AMOUNT:
                break

        # Save the results and models afer every epochs
        path2save = './Result/cifar_gan/visualization'
        fixed_p = path2save + '/' + str(epoch) + '.png'
        if not os.path.exists(path2save):
            os.makedirs(path2save)
        try:
            show_result(epoch, path=fixed_p, netG=netG)  # Shows the result of actual epoch
        except IndexError as e:
            print(f"IndexError: {e}")
        torch.save(netG.state_dict(), './Result/cifar_gan/tuned_G_' + str(epoch) + '.pth')  # Saves Generator
        torch.save(netD.state_dict(), './Result/cifar_gan/tuned_D_' + str(epoch) + '.pth')  # Saves Diskriminator

        # === Validation Phase ===
        netG.eval()
        netD.eval()
        cls.eval()

        # Counter for correct predictions and total predictions
        correct = torch.zeros(1).squeeze().to(device)
        total   = torch.zeros(1).squeeze().to(device)