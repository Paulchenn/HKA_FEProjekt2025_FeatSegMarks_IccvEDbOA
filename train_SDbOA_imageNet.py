import json
import os
import csv
from datetime import datetime
import time
import torch
from torch import nn, optim

from torchvision import transforms
from utils.get_ImageNet import main as get_ImageNet
from models import generation
from models.resnet import ResNet18, BasicBlock
from utils.modules  import *
from collections import deque

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

IMG_SIZE = config["image_size"]
GEN_IN_DIM = config["generator_input_dim"]
NUM_CLASSES = config["num_classes"]
PATIENCE = config["patience"]
acc_history = deque(maxlen=PATIENCE)
MIN_SLOPE = config["min_slope"]

SHOW_IMAGES = config["show_images"]
SHOW_IMAGES_INTERVAL = config["show_images_interval"]

DEBUG_MODE = config["debug_mode"]
DEBUG_ITERS_START = config["debugIterations_strt"]
DEBUG_ITERS_AMOUNT = config["debugIterations_amount"]
LR_GEN = config["learning_rate"]["generator"]
LR_DISC = config["learning_rate"]["discriminator"]
LR_CLS = config["learning_rate"]["classifier"]



if __name__ == "__main__":

    # === Initialize Trainiingclasses ===
    emse    = EMSE()    # Initialize EMSE class
    tsd     = TSD()     # Initialize TSD class
    tsg     = TSG()     # Initialize TSG class

    # Transformations for preprocessing training data
    transform_train = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Transformations for preprocessing test data
    transform_test = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # === Load the dataset (ImageNet) ===
    train_dataset, test_dataset = get_ImageNet(
        transform_train=transform_train,
        transform_test=transform_test,
        batch_size=BATCH_SIZE
    )
    
    # Set best accuracy to 0 to be sure that first accuracy is better
    best_acc = 0
    
    # === Initialize networks (and move to CPU/GPU) ===
    netG    = generation.generator(GEN_IN_DIM).to(DEVICE)       # Generator network with input size GEN_IN_DIM
    netD    = generation.Discriminator(NUM_CLASSES).to(DEVICE)             # Discriminator to distinguish real/fake images
    cls     = ResNet18(BasicBlock, num_classes=NUM_CLASSES).to(DEVICE)   # Classifier (here: ResNet-18)

    # === Initialize optimizers ===
    optimG = optim.Adam(netG.parameters(), lr=LR_GEN, betas=(0., 0.99))     # Optimizer for Generator (GAN-typical Betas)
    optimD = optim.Adam(netD.parameters(), lr=LR_DISC, betas=(0., 0.99))    # Optimizer for Discriminator (GAN-typical Betas)
    optimC = optim.Adam(cls.parameters(), lr=LR_CLS, betas=(0., 0.99))      # Optimizer for Classifier (GAN-typical Betas)

    # === Initialize loss functions === 
    L1_loss = nn.L1Loss()                # Absoulte Error (e.g. for Reconstruction)
    CE_loss = nn.CrossEntropyLoss()      # Classification Loss (for multi-class outputs like CIFAR-10 Classes)

    # === Create one batch of test data ===
    for i, (img, label) in enumerate(test_dataset):
        img     = img.to(DEVICE)
        label   = label.to(DEVICE)

        # Reset gradients
        netD.zero_grad()
        netG.zero_grad()
        cls.zero_grad()

        # === EMSE >>>
        EDGE_MAP_TEST = emse.doEMSE(img)
        # <<< EMSE ===

        # === TSD >>>
        DEFORMED_IMG_TEST = tsd.doTSD(img)
        # <<< TSD ===

        break
    
    # Liste zum Speichern der Metriken
    epoch_metrics = []

    # Zielordner mit aktuellem Datum anlegen
    run_date = datetime.now().strftime("%Y-%m-%d")
    csv_output_dir = os.path.join("Result", "epoch_metrics", run_date)
    os.makedirs(csv_output_dir, exist_ok=True)

    # Pfad zur CSV-Datei
    csv_output_path = os.path.join(csv_output_dir, "SDbOA_metrics.csv")

    
    ##### ===== TRAINING ===== #####
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
            # For Debugging
            if DEBUG_MODE and i < DEBUG_ITERS_START:
                continue

            img     = img.to(DEVICE)
            label   = label.to(DEVICE)

            # === EMSE >>>
            edgeMap = emse.doEMSE(img)
            # <<< EMSE ===

            # === TSD >>>
            deformedImg = tsd.doTSD(edgeMap)
            # <<< TSD ===

            # === Show images depending on configuration ===
            if SHOW_IMAGES and i % SHOW_IMAGES_INTERVAL == 0:
                show_images(
                    img,
                    deformedImg,
                    edgeMap
                )

            # === TSG >>>
            netD, netG, cls, optimD, optimG, optimC, CE_loss, L1_loss, loss_tot = tsg.doTSG_training(
                emse,
                tsd,
                img,
                label,
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
            show_result(  # Shows the result of actual epoch
                epoch,
                EDGE_MAP_TEST,
                DEFORMED_IMG_TEST,
                path=fixed_p,
                netG=netG
            )
        except IndexError as e:
            print(f"IndexError: {e}")
        torch.save(netG.state_dict(), './Result/cifar_gan/tuned_G_' + str(epoch) + '.pth')  # Saves Generator
        torch.save(netD.state_dict(), './Result/cifar_gan/tuned_D_' + str(epoch) + '.pth')  # Saves Diskriminator

        # === Validation Phase ===
        # Für FPS-Messung: Startzeit erfassen
        val_start_time = time.time()

        netG.eval()
        netD.eval()
        cls.eval()

        # Counter for correct predictions and total predictions
        correct = torch.zeros(1).squeeze().to(device)
        total   = torch.zeros(1).squeeze().to(device)

        for i, (img, label) in enumerate(test_dataset):
            # For Debugging
            if DEBUG_MODE and i < DEBUG_ITERS_START:
                continue

            img     = img.to(DEVICE)
            label   = label.to(DEVICE)

            # === EMSE >>>
            edgeMap = emse.doEMSE(img)
            # <<< EMSE ===

            # === TSD >>>
            # TSD module is not used in the validation phase
            # --> no deformation of the image
            # deformed image is the same as edge map
            deformedImg = edgeMap
            # <<< TSD ===

            # === Generation and Classification >>>
            prediction = tsg.doTSG_testing(
                img,
                deformedImg,
                netG,
                cls
            )
            # <<< Generation and Classification ===

            # Count correct predictions and total predictions
            correct += (prediction == label).sum().float()
            total += len(label)

            # For debuging (Training of even one Epoch takes very long)
            if DEBUG_MODE and i >= DEBUG_ITERS_START+DEBUG_ITERS_AMOUNT:
                break

        # Calculate accuracy, append to history and print
        acc = (correct / total).cpu().detach().data.numpy()
        acc_history.append(acc)
        print('Epoch: ', epoch + 1, ' test accuracy: ', acc)

        if acc > best_acc:
            best_acc = acc
            print('Improvement, best accuracy: ', acc)
            torch.save(cls.state_dict(), './Result/best_cls.pth')
        else:
            print('No improvement, best accuracy: ', best_acc)
        
        val_end_time = time.time()
        val_duration = val_end_time - val_start_time
        fps = float(total.cpu()) / val_duration
        print(f"Validation Inference Speed: {fps:.2f} FPS")

        # === Speichere Metriken dieser Epoche ===
        epoch_metric = {
            "epoch": epoch + 1,
            "accuracy": round(float(acc), 4),
            "loss_total": round(float(loss_tot.item()), 4),
            "FPS": round(fps, 2),
            "AUC@5": None,  # Platzhalter – später in Evaluation ersetzen
            "MMA@3": None,
            "HomographyAcc": None
        }
        epoch_metrics.append(epoch_metric)


        # Early stopping
        if len(acc_history) >= PATIENCE:
            slope = np.polyfit(range(PATIENCE), list(acc_history), 1)[0]
            print(f"Slope of accuracy trend: {slope:.6f}")
            if slope < MIN_SLOPE:
                print(f"Early stopping at epoch {epoch + 1}: accuracy trend too flat.")
                break
    

    # === Schreibe alle Metriken als CSV-Datei ===
    csv_fields = epoch_metrics[0].keys() if epoch_metrics else []

    with open(csv_output_path, mode='w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(epoch_metrics)

    print(f"[INFO] Epoch-Metriken gespeichert unter: {csv_output_path}")
