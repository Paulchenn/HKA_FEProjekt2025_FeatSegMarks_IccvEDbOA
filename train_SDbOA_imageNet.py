import csv
import os
import pdb
import sys
import time
import torch

import torchvision.models as torch_models

from collections import deque
from datetime import datetime
from models import generation
from torch import nn, optim
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from tqdm import tqdm
from utils.modules  import *
from utils.helpers import *


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()  # wichtig für Echtzeit-Ausgabe

    def flush(self):
        for f in self.files:
            f.flush()


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

# create log-directory (if not yet done)
folder2save_log = os.path.join(config.SAVE_PATH)
create_saveFolder(folder2save_log)  # Create folder to save log
path2save_log = os.path.join(folder2save_log, f'log.txt')  # Path to save the log

# Log-Datei öffnen und stdout + stderr umleiten
log_file = open(path2save_log, "w")
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)

# Clear GPU Memory
if not config.DEVICE.type=="cpu":
    torch.cuda.empty_cache()

if __name__ == "__main__":
    # # start debugging
    # pdb.set_trace()

    print("=================================================")
    print(f"===== TRAINING STARTET {datetime.now().strftime('%d.%m.%Y, %H-%M-%S')} =====")
    print("=================================================")

    if config.DEBUG_MODE:
        print("*************************")
        print("!!!!! DEBUG MODE ON !!!!!")
        print("*************************")

    torch.autograd.set_detect_anomaly(True)


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
    root = os.path.join(cwd, "src", config.DATASET_NAME)
    root = config.DATASET_PATH
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
    netD    = generation.Discriminator(config.NUM_CLASSES, input_size=config.IMG_SIZE).to(config.DEVICE) 
    cls     = torch_models.resnet18(weights=ResNet18_Weights.DEFAULT)
    cls.fc  = nn.Linear(cls.fc.in_features, config.NUM_CLASSES) # Passe den letzten Layer an deine num_classes an
    cls     = cls.to(config.DEVICE)
    

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
        print(f"Loaded Generator-Net checkpoint from {config.PATH_TUNED_G}")

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
        print(f"Loaded Discriminator-Net checkpoint from {config.PATH_TUNED_D}")


    # === Initialize optimizers ===
    optimG = optim.Adam(netG.parameters(), lr=config.LR_GEN, betas=(0., 0.99))     # Optimizer for Generator (GAN-typical Betas)
    optimD = optim.Adam(netD.parameters(), lr=config.LR_DISC, betas=(0., 0.99))    # Optimizer for Discriminator (GAN-typical Betas)

    if os.path.exists(config.PATH_OPTIM_G):
        optimG.load_state_dict(torch.load(config.PATH_OPTIM_G))
        print(f"Loaded Generator-Optimizer checkpoint from {config.PATH_TUNED_G}")

    if os.path.exists(config.PATH_OPTIM_D):
        optimD.load_state_dict(torch.load(config.PATH_OPTIM_D))
        print(f"Loaded Discriminator-Opitmizer checkpoint from {config.PATH_TUNED_D}")


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

        IMG = img

        # Reset gradients
        netD.zero_grad()
        netG.zero_grad()
        cls.zero_grad()

        # === EMSE >>>
        EDGE_MAP_TEST = emse.doEMSE(img)
        # <<< EMSE ===

        # === TSD >>>
        DEFORMED_IMG_TEST = tsd.doTSD(EDGE_MAP_TEST)
        # <<< TSD ===

        # === Show images depending on configuration ===
        show_result(  # Shows the result of actual epoch
            config=config,
            num_epoch=-1,
            img=IMG,
            edgeMap=EDGE_MAP_TEST,
            deformedMap=DEFORMED_IMG_TEST,
            path=path2save_epochImg,
            print_original=True,
            netG=netG,
            show=False,
            save=True
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
        print("")
        print("")
        print("===================================")
        print(f"=== Training at Epoch {epoch+1:0{num_digits}d} / {config.EPOCHS} ===")
        print("===================================")
        

        # === Training Phase ===
        netG.train()
        netD.train()
        cls.eval()

        # Reduce learning rate after 60 epochs
        if epoch == 60:
            for param_group in optimG.param_groups:
                param_group['lr'] = config.LR_GEN / 10
            for param_group in optimD.param_groups:
                param_group['lr'] = config.LR_DISC / 10

        # Initialize GradScaler for mixed precision training
        scaler = torch.amp.GradScaler()
        time_Iteration = []  # Start time for iteration
        time_EMSE = []
        time_TSD = []
        time_TSG = type('TimeTSG', (object,), {})()  # Create a simple object to hold TSG times
        time_TSG.time_trainD = []
        time_TSG.time_trainG1 = []
        time_TSG.time_trainCls = []
        time_TSG.time_trainG2 = []
        time_TSG.time_tot = []
        itersForAverageCalc = 10
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True, file=sys.__stderr__, ncols=100)
        for i, (img, label) in pbar:
            time_startIteration = time.time()

            # For Debugging
            if config.DEBUG_MODE and i < config.DEBUG_ITERS_START:
                continue

            img = img.to(config.DEVICE, non_blocking=True)
            label = label.to(config.DEVICE, non_blocking=True)

            # === EMSE >>>
            time_startEMSE = time.time()
            edgeMap = emse.doEMSE(img)
            time_EMSE.append(time.time() - time_startEMSE)
            # <<< EMSE ===

            # === TSD >>>
            time_startTSD = time.time()
            deformedImg = tsd.doTSD(edgeMap)
            time_TSD.append(time.time() - time_startTSD)
            # <<< TSD ===

            # === Show images depending on configuration ===
            if config.SHOW_IMG_EMSE_TSD and i % config.SHOW_IMAGES_INTERVAL == 0:
                show_imgEmseTsd(
                    img,
                    deformedImg,
                    edgeMap
                )

            # === TSG >>>
            time_startTSG = time.time()
            netD, netG, cls, optimD, optimG, CE_loss, L1_loss, loss, time_, scaler = tsg.doTSG_training(
                i, config, emse, img, label, edgeMap, deformedImg, netD, netG, cls,
                optimD, optimG, CE_loss, L1_loss, time_TSG, scaler, downSize=config.DOWN_SIZE
            )
            time_TSG.time_tot.append(time.time() - time_startTSG)
            # <<< TSG ===

            time_Iteration.append(time.time() - time_startIteration)

            # # Calculate times
            # if i==itersForAverageCalc-1:
            #     print(f"Time taken for {i+1} iterations: {round(sum(time_Iteration), 4)} s")
            #     print("Epoch will roughly take: ", round(sum(time_Iteration)/itersForAverageCalc * len(train_loader) / 60, 4), " min")
            #     print('----------')
            #     print(f"Average times after {i+1} iterations:")
            #     print("     Iteration:      ", round(sum(time_Iteration)/(i+1), 4), " s")
            #     print("     EMSE:           ", round(sum(time_EMSE)/(i+1), 4), " s")
            #     print("     TSD:            ", round(sum(time_TSD)/(i+1), 4), " s")
            #     print("     TSG (total):    ", round(sum(time_TSG.time_tot)/(i+1), 4), " s")
            #     print("     TSG (trainD):   ", round(sum(time_TSG.time_trainD)/(i+1), 4), " s")
            #     print("     TSG (trainG1):  ", round(sum(time_TSG.time_trainG1)/(i+1), 4), " s")
            #     print("     TSG (trainCls): ", round(sum(time_TSG.time_trainCls)/(i+1), 4), " s")
            #     print("     TSG (trainG2):  ", round(sum(time_TSG.time_trainG2)/(i+1), 4), " s")
            #     print("----------")
            # elif (i+1) % config.LOG_INTERVAL == 0 and (i+1) > itersForAverageCalc:
            #     print(f"Epoch[{epoch + 1} / {config.EPOCHS}][{i+1} / {len(train_loader)}] LOSS:")
            #     print(f"    Discriminator loss: (-D_result_realImg+D_result_roughImg)+(0.5*D_celoss) = {loss.stage1_D_loss.item():.4f}")
            #     print(f"    Generator loss Stage 1: G_L1_loss_rough-D_result_roughImg+(0.5*G_celoss_rough) = {loss.stage1_G_loss.item():.4f}")
            #     if config.TRAIN_WITH_CLS:
            #         print(f"    Classifier loss: {loss.cls_loss.item():.4f}")
            #         print(f"    Generator loss Stage 2: G_L1_loss_fine-D_result_fineImg+G_celoss_fine+edge_loss+cls_loss = {loss.stage2_G_loss.item():.4f}")
            #     else:
            #         print(f"    Generator loss Stage 2: G_L1_loss_fine-D_result_fineImg+G_celoss_fine+edge_loss = {loss.stage2_G_loss.item():.4f}")
            #     print("----------")

            pbar.set_postfix({
                "Iter": f"{time_Iteration[-1]:.2f}s",
                "EMSE": f"{time_EMSE[-1]:.2f}s",
                "TSD": f"{time_TSD[-1]:.2f}s",
                "TSG": f"{time_TSG.time_tot[-1]:.2f}s",
                "D": f"{loss.stage1_D_loss.item():.2f}",
                "G1": f"{loss.stage1_G_loss.item():.2f}",
                "G2": f"{loss.stage2_G_loss.item():.2f}",
                **({"Cls": f"{loss.cls_loss.item():.2f}"} if config.TRAIN_WITH_CLS else {})
            })

            # For debugging: Stop after a certain number of iterations
            if config.DEBUG_MODE and i >= config.DEBUG_ITERS_START + config.DEBUG_ITERS_AMOUNT:
                break

        # Save the results and models afer every epochs
        path2save_epochImg = os.path.join(folder2save_epochImg, f'Epoch_{epoch+1:0{num_digits}d}.png')  # Path to save the results
        try:
            show_result(  # Shows the result of actual epoch
                config=config,
                num_epoch=epoch,
                img=IMG,
                edgeMap=EDGE_MAP_TEST,
                deformedMap=DEFORMED_IMG_TEST,
                netG=netG,
                show=False,
                save=True,
                path=path2save_epochImg
            )
        except IndexError as e:
            print(f"IndexError: {e}")
        

        # === Save Generator and its Optimizer >>>
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
        # <<< Save Generator and its Optimizer ===


        # === Save Discriminator and its Optimizer >>>
        # Save the Discriminator models after every epoch
        folder2save_tunedD = os.path.join(config.SAVE_PATH, config.DATASET_NAME, 'tuned_D')
        create_saveFolder(folder2save_tunedD)  # Create folder to save tuned Discriminator
        path2save_tunedD = os.path.join(folder2save_tunedD, f'Epoch_{epoch+1:0{num_digits}d}.pth')  # Path to save the results
        torch.save(netD.state_dict(), path2save_tunedD)  # Saves Diskriminator

        # Save the Discriminator optimizer after every epoch
        folder2save_optimD = os.path.join(config.SAVE_PATH, config.DATASET_NAME, 'optim_D')
        create_saveFolder(folder2save_optimD)  # Create folder to save tuned Generator
        path2save_optimD = os.path.join(folder2save_optimD, f'Epoch_{epoch+1:0{num_digits}d}.pth')  # Path to save the results
        torch.save(optimD.state_dict(), path2save_optimD)
        # <<< Save Discriminator and its Optimizer ===


        # === VALIDATION ===
        # Fuer FPS-Messung: Startzeit erfassen
        val_start_time = time.time()

        netG.eval()
        netD.eval()
        cls.eval()

        # Counter for correct predictions and total predictions
        correct = torch.zeros(1).squeeze().to(config.DEVICE, non_blocking=True)
        total   = torch.zeros(1).squeeze().to(config.DEVICE, non_blocking=True)

        # Only validate every N epochs
        if epoch % 1 == 0:
            print("")
            print("=====================================")
            print(f"=== Validating at Epoch {epoch+1:0{num_digits}d} / {config.EPOCHS} ===")
            print("=====================================")
            with torch.no_grad():
                for i, (img, label) in enumerate(tqdm(val_loader, desc="Validating", file=sys.__stderr__, ncols=100)):
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
                    loss, cls_prediction = tsg.doTSG_testing(
                        config=config,
                        emse=emse,
                        img=img,
                        label=label,
                        e_extend=edgeMap,
                        e_deformed=deformedImg,
                        netD=netD,
                        netG=netG,
                        cls=cls,
                        CE_loss=CE_loss,
                        L1_loss=L1_loss,
                        downSize=config.DOWN_SIZE
                    )
                    # <<< Generation and Classification ===

                    # Count correct predictions and total predictions
                    correct += (cls_prediction == label).sum().float()
                    total += len(label)

                    # For debuging (Training of even one Epoch takes very long)
                    if config.DEBUG_MODE and i >= config.DEBUG_ITERS_START+config.DEBUG_ITERS_AMOUNT:
                        break

                # Calculate accuracy, append to history and print
                acc = (correct / total).cpu().detach().data.numpy()
                config.acc_history.append(acc)

                # Print results
                print(f"accuracy: correct/total = {correct}/{total} = {acc}")
                print(f"Discriminator loss: (-D_result_realImg+D_result_roughImg)+(0.5*D_celoss) = {loss.stage1_D_loss.item():.4f}")
                print(f"Generator loss Stage 1: G_L1_loss_rough-D_result_roughImg+(0.5*G_celoss_rough) = {loss.stage1_G_loss.item():.4f}")
                if config.TRAIN_WITH_CLS:
                    print(f"Classifier loss: {loss.cls_loss.item():.4f}")
                    print(f"Generator loss Stage 2: G_L1_loss_fine-D_result_fineImg+G_celoss_fine+edge_loss+cls_loss = {loss.stage2_G_loss.item():.4f}")
                else:
                    print(f"Generator loss Stage 2: G_L1_loss_fine-D_result_fineImg+G_celoss_fine+edge_loss = {loss.stage2_G_loss.item():.4f}")
                
                val_end_time = time.time()
                val_duration = val_end_time - val_start_time
                fps = float(total.cpu()) / val_duration
                print(f"Validation Inference Speed: {fps:.2f} FPS")

                # === Speichere Metriken dieser Epoche ===
                epoch_metric = {
                    "epoch": epoch + 1,
                    "accuracy": float(acc),
                    "discriminator_loss": float(loss.stage1_D_loss.item()),
                    "classifier_loss": float(loss.cls_loss.item()),
                    "generator_loss_stage1": float(loss.stage1_G_loss.item()),
                    "generator_loss_stage2": float(loss.stage2_G_loss.item()),
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

    print("")
    print("")
    print("===============================================")
    print(f"===== TRAINING ENDED {datetime.now().strftime('%d.%m.%Y, %H-%M-%S')} =====")
    print("===============================================")
