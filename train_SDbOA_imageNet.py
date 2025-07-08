import pdb
import os
import csv
from datetime import datetime
import time
import torch
from torch import nn, optim

from torchvision import transforms
#from utils.get_ImageNet import main as get_ImageNet
from models import generation
from models.resnet import ResNet18, BasicBlock
from utils.modules  import *
from utils.helpers import *
from collections import deque

def create_saveFolder(save_path):
    """
    Creates a save folder for the results if it does not exist.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        

# Path of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Path to the config directory
CONFIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'config')

# === Load parameters for training from config.json ===
config = get_config(os.path.join(SCRIPT_DIR, 'config/config.json'))

# Clear GPU Memory
torch.cuda.empty_cache()

if __name__ == "__main__":
    # # start debugging
    # pdb.set_trace()

    # === Initialize Trainiingclasses ===
    emse    = EMSE(config=config)    # Initialize EMSE class
    tsd     = TSD(config=config)     # Initialize TSD class
    tsg     = TSG(config=config)     # Initialize TSG class

    # Transformations for preprocessing training data
    transform_train = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Transformations for preprocessing test data
    transform_val = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # === Load the dataset (ImageNet) ===
    cwd = os.getcwd()   # current working directory
    root = os.path.join(
        os.path.dirname(
            os.path.dirname(cwd)
        ),
        "data",
        config.DATASET_NAME
    )
    print(f"Root directory for ImageNet 256: {root}")
    train_loader, val_loader = get_train_val_loaders(
        config,
        data_dir=root,
        transform_train=transform_train,
        transform_val=transform_val,
        pin_memory=True
    )
    
    # Set best accuracy to 0 to be sure that first accuracy is better
    best_acc = 0
    
    # === Initialize networks (and move to CPU/GPU) ===
    netG    = generation.generator(config.GEN_IN_DIM, img_size=config.IMG_SIZE).to(config.DEVICE)       # Generator network with input size GEN_IN_DIM
    netD    = generation.Discriminator(config.NUM_CLASSES, input_size=config.IMG_SIZE).to(config.DEVICE)             # Discriminator to distinguish real/fake images
    cls     = ResNet18(BasicBlock, num_classes=config.NUM_CLASSES).to(config.DEVICE)   # Classifier (here: ResNet-18)
    
    # === Load checkpoints
    if os.path.exists(config.PATH_TUNED_G):
        checkpoint = torch.load(config.PATH_TUNED_G, map_location=config.DEVICE)
        model_dict = netG.state_dict()
        filtered_dict = {
            k: v for k, v in checkpoint.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        skipped = [k for k in checkpoint if k not in filtered_dict]
        if skipped:
            print(f"Not loaded Layer (due to conflicts in name and dimension):")
            for k in skipped:
                print(f"  - {k}  (Checkpoint shape: {checkpoint[k].shape})")
        model_dict.update(filtered_dict)
        netG.load_state_dict(model_dict)
        print(f"Loaded Generator checkpoint from {config.PATH_TUNED_G}")

    if os.path.exists(config.PATH_TUNED_D):
        checkpoint = torch.load(config.PATH_TUNED_D, map_location=config.DEVICE)
        model_dict = netD.state_dict()
        filtered_dict = {
            k: v for k, v in checkpoint.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        skipped = [k for k in checkpoint if k not in filtered_dict]
        if skipped:
            print(f"Not loaded Layer (due to conflicts in name and dimension):")
            for k in skipped:
                print(f"  - {k}  (Checkpoint shape: {checkpoint[k].shape}, Model shape: {model_dict.get(k, 'missing')})")
        model_dict.update(filtered_dict)
        netD.load_state_dict(model_dict)
        print(f"Loaded Discriminator checkpoint from {config.PATH_TUNED_D}")

    # === Initialize optimizers ===
    optimG = optim.Adam(netG.parameters(), lr=config.LR_GEN, betas=(0., 0.99))     # Optimizer for Generator (GAN-typical Betas)
    optimD = optim.Adam(netD.parameters(), lr=config.LR_DISC, betas=(0., 0.99))    # Optimizer for Discriminator (GAN-typical Betas)
    optimC = optim.Adam(cls.parameters(), lr=config.LR_CLS, betas=(0., 0.99))      # Optimizer for Classifier (GAN-typical Betas)

    if os.path.exists(config.PATH_OPTIM_G):
        optimG.load_state_dict(torch.load(config.PATH_OPTIM_G))

    if os.path.exists(config.PATH_OPTIM_D):
        optimD.load_state_dict(torch.load(config.PATH_OPTIM_D))

    # === Initialize loss functions === 
    L1_loss = nn.L1Loss()                # Absoulte Error (e.g. for Reconstruction)
    CE_loss = nn.CrossEntropyLoss()      # Classification Loss (for multi-class outputs like CIFAR-10 Classes)


    # === Create a directory to save the results ===
    folder2save_epochImg = os.path.join(config.SAVE_PATH, config.DATASET_NAME, 'visualization')
    create_saveFolder(folder2save_epochImg)  # Create folder to save epoch images
    num_digits = len(str(config.EPOCHS))
    path2save_epochImg = os.path.join(folder2save_epochImg, f'originalImages.png')  # Path to save the results
    
    # === Create one batch of test data ===
    for i, (img, label) in enumerate(val_loader):
        img     = img.to(config.DEVICE)
        label   = label.to(config.DEVICE)

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

        # === Show images depending on configuration ===
        show_result(  # Shows the result of actual epoch
            -1,  # -1 means no epoch number
            img,
            EDGE_MAP_TEST,
            DEFORMED_IMG_TEST,
            path=path2save_epochImg,
            print_original=True,
            show=False,
            save=True,
            netG=netG,
        )

        break
    
    # Liste zum Speichern der Metriken
    epoch_metrics = []

    # Zielordner mit aktuellem Datum anlegen
    csv_output_dir = os.path.join(config.SAVE_PATH, "epoch_metrics")
    os.makedirs(csv_output_dir, exist_ok=True)

    # Pfad zur CSV-Datei
    csv_output_path = os.path.join(csv_output_dir, "SDbOA_metrics.csv")
    
    
   
    ##### ===== TRAINING ===== #####
    for epoch in range(config.EPOCHS):
        print(epoch)

        # === Training Phase ===
        netG.train()
        netD.train()
        cls.train()

        # Reduce learning rate after 60 epochs
        if epoch == 60:
            for param_group in optimG.param_groups:
                param_group['lr'] = config.LR_GEN / 10
            for param_group in optimD.param_groups:
                param_group['lr'] = config.LR_DISC / 10
            for param_group in optimC.param_groups:
                param_group['lr'] = config.LR_CLS / 10

        # Initialize GradScaler for mixed precision training
        scaler = torch.amp.GradScaler()
        start = time.time()  # Start time for iteration
        for i, (img, label) in enumerate(train_loader):
            # pdb.set_trace()  # Start debugging

            # For Debugging
            if config.DEBUG_MODE and i < config.DEBUG_ITERS_START:
                continue

            if i==10:
                print("Time taken for 10 iterations: ", round(time.time() - start, 2), " s")
                print("Epoch will roughly take: ", round((time.time() - start)/10 * len(train_loader) / 60, 2), " min")

            img = img.to(config.DEVICE, non_blocking=True)
            label = label.to(config.DEVICE, non_blocking=True)
            with torch.amp.autocast(config.DEVICE.type):
                # === EMSE >>>
                edgeMap = emse.doEMSE(img)
                # <<< EMSE ===

                # === TSD >>>
                deformedImg = tsd.doTSD(edgeMap)
                # <<< TSD ===

                # === Show images depending on configuration ===
                if config.SHOW_IMAGES and i % config.SHOW_IMAGES_INTERVAL == 0:
                    show_images(
                        img,
                        deformedImg,
                        edgeMap
                    )

                # === TSG >>>
                netD, netG, cls, optimD, optimG, optimC, CE_loss, L1_loss, loss_tot, scaler = tsg.doTSG_training(
                    emse, tsd, img, label, deformedImg, netD, netG, cls,
                    optimD, optimG, optimC, CE_loss, L1_loss, scaler
                )
                # <<< TSG ===

            # Print loss values after every 5 iterations
            if i % config.LOG_INTERVAL == 0:
                print('Epoch[', epoch + 1, '/', config.EPOCHS, '][', i + 1, '/', len(train_loader), ']: TOTAL_LOSS', loss_tot.item())

            # For debugging: Stop after a certain number of iterations
            if config.DEBUG_MODE and i >= config.DEBUG_ITERS_START + config.DEBUG_ITERS_AMOUNT:
                break

        # Save the results and models afer every epochs
        path2save_epochImg = os.path.join(folder2save_epochImg, f'Epoch_{epoch+1:0{num_digits}d}.png')  # Path to save the results
        try:
            show_result(  # Shows the result of actual epoch
                epoch,
                EDGE_MAP_TEST,
                DEFORMED_IMG_TEST,
                path=path2save_epochImg,
                netG=netG,
                show=False,
                save=True
            )
        except IndexError as e:
            print(f"IndexError: {e}")
        
        # Save the Generator models after every epoch
        folder2save_tunedG = os.path.join(config.SAVE_PATH, config.DATASET_NAME, 'tuned_G')
        create_saveFolder(folder2save_tunedG)  # Create folder to save tuned Generator
        path2save_tunedG = os.path.join(folder2save_tunedG, f'Epoch_{epoch+1:0{num_digits}d}.pth')  # Path to save the results
        torch.save(netG.state_dict(), path2save_tunedG)  # Saves Generator

        # Save the Generator optimizer after every epoch
        folder2save_optimG = os.path.join(config.SAVE_PATH, config.DATASET_NAME, 'optim_G')
        create_saveFolder(folder2save_optimG)  # Create folder to save tuned Generator
        path2save_optimG = os.path.join(folder2save_optimG, f'Epoch_{epoch+1:0{num_digits}d}.pth')  # Path to save the results
        torch.save(optimG.state_dict(), path2save_optimG)


        # Save the Discriminator models after every epoch
        folder2save_tunedD = os.path.join(config.SAVE_PATH, config.DATASET_NAME, 'tuned_D')
        create_saveFolder(folder2save_tunedD)  # Create folder to save tuned Discriminator
        path2save_tunedD = os.path.join(folder2save_tunedD, f'Epoch_{epoch+1:0{num_digits}d}.pth')  # Path to save the results
        torch.save(netD.state_dict(), path2save_tunedD)  # Saves Diskriminator

        # Save the Discriminator optimizer after every epoch
        folder2save_optimD = os.path.join(config.SAVE_PATH, config.DATASET_NAME, 'optim_D')
        create_saveFolder(folder2save_optimD)  # Create folder to save tuned Generator
        path2save_optimD = os.path.join(folder2save_optimD, f'Epoch_{epoch+1:0{num_digits}d}.pth')  # Path to save the results
        torch.save(optimG.state_dict(), path2save_optimD)

        # === Validation Phase ===
        # Fuer FPS-Messung: Startzeit erfassen
        val_start_time = time.time()

        netG.eval()
        netD.eval()
        cls.eval()

        # Counter for correct predictions and total predictions
        correct = torch.zeros(1).squeeze().to(device)
        total   = torch.zeros(1).squeeze().to(device)

        # Only validate every N epochs
        if epoch % 5 == 0:
            print(f"Validating at epoch {epoch + 1}/{config.EPOCHS}...")
            with torch.no_grad():
                for i, (img, label) in enumerate(val_loader):
                    # For Debugging
                    if config.DEBUG_MODE and i < config.DEBUG_ITERS_START:
                        continue

                    img     = img.to(config.DEVICE)
                    label   = label.to(config.DEVICE)

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
                    if config.DEBUG_MODE and i >= config.DEBUG_ITERS_START+config.DEBUG_ITERS_AMOUNT:
                        break

                # Calculate accuracy, append to history and print
                acc = (correct / total).cpu().detach().data.numpy()
                config.acc_history.append(acc)
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
                    "AUC@5": None,  # Platzhalter – spaeter in Evaluation ersetzen
                    "MMA@3": None,
                    "HomographyAcc": None
                }
                epoch_metrics.append(epoch_metric)


                # Early stopping
                if len(config.acc_history) >= config.PATIENCE:
                    slope = np.polyfit(range(config.PATIENCE), list(config.acc_history), 1)[0]
                    print(f"Slope of accuracy trend: {slope:.6f}")
                    if slope < config.MIN_SLOPE:
                        print(f"Early stopping at epoch {epoch + 1}: accuracy trend too flat.")
                        break

                # === Schreibe alle Metriken als CSV-Datei ===
                csv_fields = epoch_metrics[0].keys() if epoch_metrics else []

                with open(csv_output_path, mode='w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
                    writer.writeheader()
                    writer.writerows(epoch_metrics)

                print(f"[INFO] Saved Epoch-Metrics at: {csv_output_path}")


        print('Epoch[', epoch + 1, '/', config.EPOCHS, '][', i + 1, '/', len(train_loader), ']: TOTAL_LOSS', loss_tot.item())
