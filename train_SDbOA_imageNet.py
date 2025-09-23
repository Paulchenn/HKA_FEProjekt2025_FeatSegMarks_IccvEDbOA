import csv
import os
import pdb
import sys
import time
import torch
import shutil

import torchvision.models as torch_models

# import tkinter as tk
# from tkinter import messagebox

from collections import deque
from datetime import datetime
from models import generation_imageNet_V2_2 as generation_imageNet
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        

def export_weights(
        config,
        training_stage,
        state,
        epoch,
        netD,
        optimD_s1,
        optimD_s2,
        netG,
        optimG_s1,
        optimG_s2
):
    """
    Speichert Netze & Optimizer. Wenn state == 'temp':
      - in jedem Zielordner erst alte Dateien aufräumen (nur die letzten 4 behalten),
      - anschließend den neuen Checkpoint hinzufügen.
    Ergebnis: max. 5 Dateien pro Ordner (4 alte + der neue).
    """
    num_digits = len(str(getattr(config, "epochs", 3)))

    # --- Helper: Ordner anlegen
    def create_saveFolder(folder):
        os.makedirs(folder, exist_ok=True)

    # --- Helper: Alte Dateien aufräumen (nach mtime sortiert)
    def _cleanup_keep_last_k(folder, keep=4, suffix=".pth"):
        try:
            files = [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(suffix)
            ]
        except FileNotFoundError:
            return
        if len(files) <= keep:
            return
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)  # neueste zuerst
        for f in files[keep:]:
            try:
                os.remove(f)
            except OSError:
                pass

    # richtigen Optimizer je Stage
    if training_stage == 1:
        optimD = optimD_s1
        optimG = optimG_s1
        patience = getattr(config, "patience_stage1", 3)
    else:
        optimD = optimD_s2
        optimG = optimG_s2
        patience = getattr(config, "patience_stage2", 3)

    base_dir = os.path.join(config.SAVE_PATH, config.DATASET_NAME)

    # Wenn 'temp': vor dem Speichern entrümpeln (4 alte behalten, neuer kommt gleich dazu)
    if state == "temp":
        for sub in (f"{state}_netD", f"{state}_optimD", f"{state}_netG", f"{state}_optimG"):
            folder = os.path.join(base_dir, sub)
            create_saveFolder(folder)
            _cleanup_keep_last_k(folder, keep=patience, suffix=".pth")

    # === Discriminator + Optimizer speichern ===
    # Discriminator
    folder2save = os.path.join(base_dir, f"{state}_netD")
    create_saveFolder(folder2save)
    path2save = os.path.join(
        folder2save,
        f"netD__stage{training_stage}__Epoch_{epoch+1:0{num_digits}d}.pth"
    )
    torch.save(netD.state_dict(), path2save)

    # Discriminator-Optimizer
    folder2save = os.path.join(base_dir, f"{state}_optimD")
    create_saveFolder(folder2save)
    path2save = os.path.join(
        folder2save,
        f"optimD__stage{training_stage}__Epoch_{epoch+1:0{num_digits}d}.pth"
    )
    torch.save(optimD.state_dict(), path2save)

    # === Generator + Optimizer speichern ===
    # Generator
    folder2save = os.path.join(base_dir, f"{state}_netG")
    create_saveFolder(folder2save)
    path2save = os.path.join(
        folder2save,
        f"netG__stage{training_stage}__Epoch_{epoch+1:0{num_digits}d}.pth"
    )
    torch.save(netG.state_dict(), path2save)

    # Generator-Optimizer
    folder2save = os.path.join(base_dir, f"{state}_optimG")
    create_saveFolder(folder2save)
    path2save = os.path.join(
        folder2save,
        f"optimG__stage{training_stage}__Epoch_{epoch+1:0{num_digits}d}.pth"
    )
    torch.save(optimG.state_dict(), path2save)



def seconds_to_hhmmss(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

        
def myInit():
    # Path of the current script
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Path to the config directory
    CONFIG_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'config')

    # === Load parameters for training from config.json ===
    config = get_config(os.path.join(SCRIPT_DIR, 'config/config.json'))
    config.SCRIPT_DIR = SCRIPT_DIR
    config.CONFIG_DIR = CONFIG_DIR

    # create log-directory (if not yet done)
    folder2save_log = os.path.join(config.SAVE_PATH)
    create_saveFolder(folder2save_log)  # Create folder to save log

    # create log-directory (if not yet done)
    folder2save_log = os.path.join(config.SAVE_PATH)
    create_saveFolder(folder2save_log)  # Create folder to save log

    # aktuelle (aufgelöste) Laufzeit-Config im Log-Ordner speichern
    config_snapshot_path = os.path.join(folder2save_log, "config_runtime.json")
    with open(config_snapshot_path, "w") as f:
        json.dump(vars(config), f, indent=2, default=str)  # default=str wandelt z.B. torch.device in String

    print(f"[Init] Laufzeit-Config gespeichert: {os.path.abspath(config_snapshot_path)}")

    # Log-Datei im selben Ordner
    path2save_log = os.path.join(folder2save_log, "log.txt")

    # Log-Datei öffnen und stdout + stderr umlcsv_epochs_output_patheiten
    log_file = open(path2save_log, "w")
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)

    # Clear GPU Memory
    if not config.DEVICE.type=="cpu":
        torch.cuda.empty_cache()

    return config


if __name__ == "__main__":
    config = myInit()
    # start debugging
    #pdb.set_trace()

    time_trainingStart = time.time()

    print("=================================================")
    print(f"===== TRAINING STARTET {datetime.now().strftime('%d.%m.%Y, %H-%M-%S')} =====")
    print("=================================================")

    if config.DEBUG_MODE:
        print("*************************")
        print("!!!!! DEBUG MODE ON !!!!!")
        print("*************************")

    torch.autograd.set_detect_anomaly(True)


    # get image sizes
    img_h = int(getattr(config, "image_height", 608))
    img_w = int(getattr(config, "image_width", 800))
    if img_h == img_w:
        config.IMG_SIZE = img_h
    else:
        print(f"Image height does not match image width. Setting IMG_SIZE to smaller one.")
        if img_h < img_w:
            config.IMG_SIZE = img_h
        else:
            config.IMG_SIZE = img_w


    # === Initialize Trainiingclasses ===
    ds_emse   = DS_EMSE(config=config)   # Initialize EMSE class
    emse    = EMSE(config=config)    # Initialize EMSE class
    tsd     = TSD(config=config)     # Initialize TSD class
    tsg     = TSG(config=config)     # Initialize TSG class

    # get correct weights for classifier
    cls_weights = ResNet18_Weights.DEFAULT

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

    # Transformations for classifier
    transform_cls = cls_weights.transforms()


    # === Load the dataset (ImageNet) ===
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
    netG    = generation_imageNet.generator(img_size=config.IMG_SIZE, z_dim=config.NOISE_SIZE).to(config.DEVICE)       # Generator network with input size GEN_IN_DIM
    netD    = generation_imageNet.Discriminator(config.NUM_CLASSES, input_size=config.IMG_SIZE).to(config.DEVICE) 
    cls     = torch_models.resnet18(weights=cls_weights)
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
            print(f"Not loaded Layer of NetG (due to conflicts in name and dimension):")
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
            print(f"Not loaded Layer of NetD (due to conflicts in name and dimension):")
            for k in skipped:
                print(f"  - {k}  (Checkpoint shape: {checkpoint[k].shape}, Model shape: {model_dict.get(k, 'missing')})")
        model_dict.update(filtered_dict)
        netD.load_state_dict(model_dict)
        print(f"Loaded Discriminator-Net checkpoint from {config.PATH_TUNED_D}")


    # === Initialize optimizers ===
    optimD_stage1 = optim.Adam(netD.parameters(), lr=config.LR_D_S1, betas=(0., 0.99))    # Optimizer for Discriminator (GAN-typical Betas)
    optimD_stage2 = optim.Adam(netD.parameters(), lr=config.LR_D_S2, betas=(0., 0.99))    # Optimizer for Discriminator (GAN-typical Betas)
    optimG_stage1 = optim.Adam(netG.parameters(), lr=config.LR_G_S1, betas=(0.5, 0.99))     # Optimizer for Generator (GAN-typical Betas)
    optimG_stage2 = optim.Adam(netG.parameters(), lr=config.LR_G_S2, betas=(0.5, 0.99))     # Optimizer for Generator (GAN-typical Betas)

    if os.path.exists(config.PATH_OPTIM_D_S1):
        optimD_stage1.load_state_dict(torch.load(config.PATH_OPTIM_D_S1))
        print(f"Loaded Discriminator-Opitmizer checkpoint from {config.PATH_OPTIM_D_S1}")
    if os.path.exists(config.PATH_OPTIM_D_S2):
        optimD_stage2.load_state_dict(torch.load(config.PATH_OPTIM_D_S2))
        print(f"Loaded Discriminator-Opitmizer checkpoint from {config.PATH_OPTIM_D_S2}")

    if os.path.exists(config.PATH_OPTIM_G_S1):
        optimG_stage1.load_state_dict(torch.load(config.PATH_OPTIM_G_S1))
        print(f"Loaded Generator-Optimizer checkpoint from {config.PATH_OPTIM_G_S1}")
    if os.path.exists(config.PATH_OPTIM_G_S2):
        optimG_stage2.load_state_dict(torch.load(config.PATH_OPTIM_G_S2))
        print(f"Loaded Generator-Optimizer checkpoint from {config.PATH_OPTIM_G_S2}")

    # === Initialize Scheduler ===
    scheduler_mode = 'min'
    scheduler_factor = 0.4
    scheduler_patience = 10
    scheduler_threshold = 1e-3
    scheduler_minLr = 1e-6
    scheduler_D_stage1 = ReduceLROnPlateau(
        optimD_stage1,
        mode=scheduler_mode,
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        min_lr=scheduler_minLr
    )
    scheduler_D_stage2 = ReduceLROnPlateau(
        optimD_stage2,
        mode=scheduler_mode,
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        min_lr=scheduler_minLr
    )
    scheduler_G_stage1 = ReduceLROnPlateau(
        optimG_stage1,
        mode=scheduler_mode,
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        min_lr=scheduler_minLr
    )
    scheduler_G_stage2 = ReduceLROnPlateau(
        optimG_stage2,
        mode=scheduler_mode,
        factor=scheduler_factor,
        patience=scheduler_patience,
        threshold=scheduler_threshold,
        min_lr=scheduler_minLr
    )


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

        # === EMSE >>>
        EDGE_MAP_TEST_EMSE = emse.doEMSE(img)
        EDGE_MAP_TEST_DS_EMSE = ds_emse.diff_edge_map(img)
        # <<< EMSE ===

        # === TSD >>>
        DEFORMED_MAP_TEST, TSD_GRID_TEST = tsd.doTSD(EDGE_MAP_TEST_DS_EMSE, return_grid=True)
        # <<< TSD ===

        # === Show images depending on configuration ===
        show_result(  # Shows the result of actual epoch
            config=config,
            num_epoch=-1,
            img=IMG,
            edgeMap=EDGE_MAP_TEST_DS_EMSE,
            path=path2save_epochImg,
            print_original=True,
            netG=netG,
            show=False,
            save=True
        )

        break
    
    # Liste zum Speichern der Metriken
    epoch_metrics = []
    iteration_metrics = []

    # Zielordner mit aktuellem Datum anlegen
    csv_output_dir = os.path.join(config.SAVE_PATH, "metrics")
    os.makedirs(csv_output_dir, exist_ok=True)

    # Pfad zur CSV-Datei
    csv_iters_output_path = os.path.join(csv_output_dir, "iterations_metrics.csv")
    csv_epochs_output_path = os.path.join(csv_output_dir, "epoch_metrics.csv")
    

   
    ##### ===== TRAINING ===== #####
    # === start training in stage 1 ===
    if config.start_in_stage == 2:
        print("")
        print("")
        print("*******************************")
        print("!!!!! STARTING IN STAGE 2 !!!!!")
        print("*******************************")
        # tk.Tk().withdraw(); messagebox.showinfo("!!!!! STARTING IN STAGE 2 !!!!!", "Go on?");  # optional: tk._default_root.destroy()


    training_stage = config.start_in_stage
    best_val_loss = None
    counter_noBetterValLoss = 0
    # pdb.set_trace()
    for epoch in range(config.EPOCHS):
        print("")
        print("")
        print("===================================")
        print(f"=== Training at Epoch {epoch+1:0{num_digits}d} / {config.EPOCHS} ===")
        print(f"======= currently in STAGE {training_stage} ======")
        print("===================================")
        if training_stage==1:
            current_lr_D = optimD_stage1.param_groups[0]['lr']
            current_lr_G = optimG_stage1.param_groups[0]['lr']
            print(f"Generator LR    : {current_lr_G:.6f}")
            print(f"Discriminator LR: {current_lr_D:.6f}")
        else:
            current_lr_D = optimD_stage2.param_groups[0]['lr']
            current_lr_G = optimG_stage2.param_groups[0]['lr']
            print(f"Generator LR    : {current_lr_G:.6f}")
            print(f"Discriminator LR: {current_lr_D:.6f}")
        

        # === Training Phase ===
        netG.train()
        netD.train()
        cls.eval()

        # === Initializations ===
        # Initialize GradScaler for mixed precision training
        scaler = torch.amp.GradScaler()
        # Initialize time lists
        time_Iteration = []  # Start time for iteration
        time_EMSE = []
        time_DS_EMSE = []
        time_TSD = []
        time_TSG = type('TimeTSG', (object,), {})()  # Create a simple object to hold TSG times
        time_TSG.time_trainD = []
        time_TSG.time_trainG1 = []
        time_TSG.time_trainCls = []
        time_TSG.time_trainG2 = []
        time_TSG.time_tot = []
        
        # === Start Iteration ===
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

            # === DS_EMSE >>>
            time_startDS_EMSE = time.time()
            edgeMap = ds_emse.diff_edge_map(img)
            time_DS_EMSE.append(time.time() - time_startDS_EMSE)
            # <<< DS_EMSE ===

            # === TSD >>>
            time_startTSD = time.time()
            deformedEdge, tsd_grid = tsd.doTSD(edgeMap, return_grid=True)  # random pro Iteration
            time_TSD.append(time.time() - time_startTSD)
            # <<< TSD ===

            # === Show images depending on configuration ===
            if config.SHOW_IMG_EMSE_TSD and i % config.SHOW_IMAGES_INTERVAL == 0:
                show_imgEmseTsd(
                    img,
                    deformedEdge,
                    edgeMap
                )

            # === TSG >>>
            time_startTSG = time.time()
            if training_stage==1:
                netD, netG, optimD_stage1, optimG_stage1, loss, time_TSG = tsg.doTSG_stage1_training(
                    iteration=i,
                    config=config,
                    ds_emse=ds_emse,
                    emse=emse,
                    img=img,
                    label=label,
                    e_extend=edgeMap,
                    netD=netD,
                    netG=netG,
                    optimD=optimD_stage1,
                    optimG=optimG_stage1,
                    CE_loss=CE_loss,
                    L1_loss=L1_loss,
                    time_TSG=time_TSG,
                    scaler=scaler
                )
            else:
                netD, netG, optimD_stage2, optimG_stage2, loss, time_TSG = tsg.doTSG_stage2_training(
                    iteration=i,
                    config=config,
                    ds_emse=ds_emse,
                    emse=emse,
                    tsd=tsd,
                    img=img,
                    label=label,
                    e_deformed=deformedEdge,
                    tsd_grid=tsd_grid,         # <— NEU: das zugehörige Grid
                    netD=netD,
                    netG=netG,
                    cls=cls,
                    transform_cls=transform_cls,
                    optimD=optimD_stage2,
                    optimG=optimG_stage2,
                    CE_loss=CE_loss,
                    L1_loss=L1_loss,
                    time_TSG=time_TSG,
                    scaler=scaler
                )
            time_TSG.time_tot.append(time.time() - time_startTSG)
            # <<< TSG ===

            time_Iteration.append(time.time() - time_startIteration)
            
            pbar.set_postfix({
                "Iter": f"{time_Iteration[-1]:.2f}s",
                "EMSE": f"{time_EMSE[-1]:.2f}s",
                "TSD": f"{time_TSD[-1]:.2f}s",
                "TSG": f"{time_TSG.time_tot[-1]:.2f}s",
                "D": f"{loss.D_loss.item():.2f}",
                f"G{training_stage}": f"{loss.G_loss.item():.2f}",
                **({"Cls": f"{loss.cls_loss.item():.2f}"} if config.TRAIN_WITH_CLS and training_stage==2 else {})
            })


            # === save metrics of this iteration ===
            iteration_metric = {
                "epoch": epoch + 1,
                "iteration": i + 1,
                "stage": training_stage,
                "discriminator_loss": float(loss.D_loss.item()),
                "generator_loss": float(loss.G_loss.item()),
                "classifier_loss": float(loss.cls_loss.item()) if config.TRAIN_WITH_CLS and training_stage==2 else None,
                "AUC@5": None,  # Platzhalter – spaeter in Evaluation ersetzen
                "MMA@3": None,
                "HomographyAcc": None
            }
            iteration_metrics.append(iteration_metric)


            # === Write all metrics as csv-file ===
            csv_fields = iteration_metrics[0].keys() if iteration_metrics else []

            with open(csv_iters_output_path, mode='w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
                writer.writeheader()
                writer.writerows(iteration_metrics)


            # For debugging: Stop after a certain number of iterations
            if config.DEBUG_MODE and i >= config.DEBUG_ITERS_START + config.DEBUG_ITERS_AMOUNT:
                break

        ### === END OF ITERATION === ###

        # Save the results and models afer every epochs
        path2save_epochImg = os.path.join(folder2save_epochImg, f'Stage{training_stage}_Epoch_{epoch+1:0{num_digits}d}.png')  # Path to save the results
        try:
            show_result(  # Shows the result of actual epoch
                config=config,
                num_epoch=epoch,
                img=IMG,
                edgeMap=EDGE_MAP_TEST_DS_EMSE if training_stage==1 else DEFORMED_MAP_TEST,
                netG=netG,
                show=False,
                save=True,
                path=path2save_epochImg,
                training_stage=training_stage
            )
        except IndexError as e:
            print(f"IndexError: {e}")

        if getattr(config, "save_all_epochs", False):
            export_weights(
                config=config,
                training_stage=training_stage,
                state='tuned',
                epoch=epoch,
                netD=netD,
                optimD_s1=optimD_stage1,
                optimD_s2=optimD_stage2,
                netG=netG,
                optimG_s1=optimG_stage1,
                optimG_s2=optimG_stage2
            )
        else:
            export_weights(
                config=config,
                training_stage=training_stage,
                state='temp',
                epoch=epoch,
                netD=netD,
                optimD_s1=optimD_stage1,
                optimD_s2=optimD_stage2,
                netG=netG,
                optimG_s1=optimG_stage1,
                optimG_s2=optimG_stage2
            )


        # === VALIDATION ===
        # Fuer FPS-Messung: Startzeit erfassen
        val_start_time = time.time()

        # Counter for correct predictions and total predictions
        correct = torch.zeros(1).squeeze().to(config.DEVICE, non_blocking=True)

        # initialize accumulators
        total_D_loss = 0.0
        total_G_loss = 0.0
        total_cls_loss = 0.0
        acc = 0.0

        # Only validate every N epochs
        print("")
        print("=====================================")
        print(f"=== Validating at Epoch {epoch+1:0{num_digits}d} / {config.EPOCHS} ===")
        print(f"======== currently in STAGE {training_stage} =======")
        print("=====================================")
        
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Epoch {epoch+1}", dynamic_ncols=True, file=sys.__stderr__, ncols=100)
        for i, (img, label) in pbar:
            time_startValIteration = time.time()

            # For Debugging
            if config.DEBUG_MODE and i < config.DEBUG_ITERS_START:
                continue

            img     = img.to(config.DEVICE)
            label   = label.to(config.DEVICE)

            # === EMSE >>>
            time_startEmse = time.time()
            edgeMap = emse.doEMSE(img)
            time_emse = time.time()-time_startEmse
            # <<< EMSE ===

            # === DS_EMSE >>>
            time_startDS_EMSE = time.time()
            edgeMap = ds_emse.diff_edge_map(img)
            time_DS_EMSE = time.time() - time_startDS_EMSE
            # <<< DS_EMSE ===

            # === TSD >>>
            time_startTsd = time.time()
            deformedEdge, tsd_grid = tsd.doTSD(edgeMap, return_grid=True)  # random pro Iteration
            # NO TSD IN VALIDATION --> PAPER (testing with and without, currently with deform)
            time_tsd = time.time() - time_startTsd
            # <<< TSD ===

            # === Generation and Classification >>>
            time_startTsg = time.time()
            if training_stage==1:
                loss = tsg.doTSG_stage1_testing(
                    config=config,
                    ds_emse=ds_emse,
                    emse=emse,
                    img=img,
                    label=label,
                    e_extend=edgeMap,
                    netD=netD,
                    netG=netG,
                    CE_loss=CE_loss,
                    L1_loss=L1_loss
                )
            else:
                loss, cls_prediction = tsg.doTSG_stage2_testing(
                    config=config,
                    ds_emse=ds_emse,
                    emse=emse,
                    tsd=tsd,
                    img=img,
                    label=label,
                    e_extend=deformedEdge,
                    tsd_grid=tsd_grid,         # <— NEU: das zugehörige Grid
                    netD=netD,
                    netG=netG,
                    cls=cls,
                    transform_cls=transform_cls,
                    CE_loss=CE_loss,
                    L1_loss=L1_loss
                )
                # Count correct predictions and total predictions to calculate accuracy
                if config.TRAIN_WITH_CLS:
                    correct += (cls_prediction == label).sum().float()
            time_tsg = time.time() - time_startTsg
            # <<< Generation and Classification ===

            total_D_loss += loss.D_loss.item()
            total_G_loss += loss.G_loss.item()
            if config.TRAIN_WITH_CLS and training_stage==2:
                total_cls_loss += loss.cls_loss.item()

            time_valIteration = time.time() - time_startValIteration

            pbar.set_postfix({
                "Iter": f"{time_valIteration:.2f}s",
                "EMSE": f"{time_emse:.2f}s",
                "TSD": f"{time_tsd:.2f}s",
                "TSG": f"{time_tsg:.2f}s",
                "D": f"{loss.D_loss.item():.2f}",
                f"G{training_stage}": f"{loss.G_loss.item():.2f}",
                **({"Cls": f"{loss.cls_loss.item():.2f}"} if config.TRAIN_WITH_CLS and training_stage==2 else {})
            })

            # For debuging (Training of even one Epoch takes very long)
            if config.DEBUG_MODE and i >= config.DEBUG_ITERS_START+config.DEBUG_ITERS_AMOUNT:
                break

        #TODO: call diagnostics.py

        # calculate average losses
        total_batches = max(len(val_loader), 1)
        total_items = total_batches*config.BATCH_SIZE
        avg_D_loss = total_D_loss / total_batches
        avg_G_loss = total_G_loss / total_batches
        current_val_loss = avg_G_loss
        if config.TRAIN_WITH_CLS and training_stage==2:
            avg_cls_loss = total_cls_loss / total_batches
            acc = (correct / total_items).cpu().detach().data.numpy()

        # Print results
        lambda_L1 = config.LAMBDA_S2_L1 # original: 1.0
        lamda_GAN = config.LAMBDA_S2_GAN # original: 1.0
        lambda_CE = config.LAMBDA_S2_CE # original: 0.5
        lambda_cls = config.LAMBDA_S2_CLS # original: 1.0
        lambda_edge = config.LAMBDA_S2_EDGE # original: 1.0
        if training_stage==1:
            print(f"Discriminator loss: (-D_result_realImg+D_result_roughImg)+0.5*D_celoss = {avg_D_loss:.4f}")
            print(f"Generator loss:     {config.LAMBDA_S1_L1}*G_L1_loss_rough-{config.LAMBDA_S1_GAN}*D_result_roughImg+{config.LAMBDA_S1_CE}*G_celoss_rough = {avg_G_loss:.4f}")
        else:
            print(f"Discriminator loss: (-D_result_realImg+D_result_roughImg)+0.5*D_celoss = {avg_D_loss:.4f}")
            if config.TRAIN_WITH_CLS and training_stage==2:
                print(f"Generator loss: {config.LAMBDA_S2_L1}*G_L1_loss_fine-{config.LAMBDA_S2_GAN}*D_result_fineImg+{config.LAMBDA_S2_CE}*G_celoss_fine+{config.LAMBDA_S2_CLS}*cls_loss+{config.LAMBDA_S2_EDGE}*edge_loss = {avg_G_loss:.4f}")
                print(f"Classifier loss: {avg_cls_loss:.4f}")
                print(f"Accuracy: correct/total = {correct}/{total_items} = {acc}")
            else:
                print(f"Generator loss: {config.LAMBDA_S2_L1}*G_L1_loss_fine-{config.LAMBDA_S2_GAN}*D_result_fineImg+{config.LAMBDA_S2_CE}*G_celoss_fine+{config.LAMBDA_S2_EDGE}*edge_loss = {avg_G_loss:.4f}")
        
        val_end_time = time.time()
        val_duration = val_end_time - val_start_time
        fps = float(total_items) / val_duration
        print(f"Validation Inference Speed: {fps:.0f} FPS")


        # === save metrics of this epoch ===
        epoch_metric = {
            "epoch": epoch + 1,
            "stage": training_stage,
            "accuracy": float(acc) if config.TRAIN_WITH_CLS and training_stage==2 else None,
            "discriminator_loss": float(avg_D_loss),
            "generator_loss": float(avg_G_loss),
            "classifier_loss": float(avg_cls_loss) if config.TRAIN_WITH_CLS and training_stage==2 else None,
            "FPS": round(fps, 2),
            "AUC@5": None,  # Platzhalter – spaeter in Evaluation ersetzen
            "MMA@3": None,
            "HomographyAcc": None
        }
        epoch_metrics.append(epoch_metric)


        # === Write all metrics as csv-file ===
        csv_fields = epoch_metrics[0].keys() if epoch_metrics else []

        with open(csv_epochs_output_path, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(epoch_metrics)


        # === Early stopping / Early switching to stage 2 ===
        # Calculate current validiation loss
        print(f"Total batches: {total_batches}")
        print(f"Total items: {total_items}")
        if best_val_loss != None:
            if current_val_loss <= best_val_loss:
                best_val_loss = current_val_loss
                netD_best = netD
                netG_best = netG
                if training_stage==1:
                    optimD_best = optimD_stage1
                    optimG_best = optimG_stage1
                else:
                    optimD_best = optimD_stage2
                    optimG_best = optimG_stage2
                epoch_best = epoch
                counter_noBetterValLoss = 0
            else:
                counter_noBetterValLoss += 1
                if training_stage==1:
                    if counter_noBetterValLoss >= config.PATIENCE_S1:
                        # Save the best models
                        export_weights(
                            config=config,
                            training_stage=training_stage,
                            state='best',
                            epoch=epoch_best,
                            netD=netD_best,
                            optimD_s1=optimD_best,  # giving both times best, because function will save it to the correct place due to training_stage
                            optimD_s2=optimD_best,  # giving both times best, because function will save it to the correct place due to training_stage
                            netG=netG_best,
                            optimG_s1=optimG_best,
                            optimG_s2=optimG_best
                        )
                        # switching to training stage 2
                        training_stage = 2
                        netD = netD_best
                        netG = netG_best
                        best_val_loss = None

                else:
                    if counter_noBetterValLoss >= config.PATIENCE_S2:
                        # Save the best models
                        export_weights(
                            config=config,
                            training_stage=training_stage,
                            state='best',
                            epoch=epoch_best,
                            netD=netD_best,
                            optimD_s1=optimD_best,  # giving both times best, because function will save it to the correct place due to training_stage
                            optimD_s2=optimD_best,  # giving both times best, because function will save it to the correct place due to training_stage
                            netG=netG_best,
                            optimG_s1=optimG_best,
                            optimG_s2=optimG_best
                        )
                        print(f"counter for no better val loss = {counter_noBetterValLoss}")
                        if current_val_loss:
                            print(f"last current val loss = avg_G_loss = {current_val_loss:.4f}")
                        if best_val_loss:
                            print(f"last best val loss = avg_G_loss = {best_val_loss:.4f}")
                        break
        else:
            best_val_loss = current_val_loss
            netD_best = netD
            netG_best = netG
            if training_stage==1:
                optimD_best = optimD_stage1
                optimG_best = optimG_stage1
            else:
                optimD_best = optimD_stage2
                optimG_best = optimG_stage2
            counter_noBetterValLoss = 0

        print(f"counter for no better val loss = {counter_noBetterValLoss}")
        if current_val_loss:
            print(f"current val loss = avg_G_loss = {current_val_loss:.4f}")
        if best_val_loss:
            print(f"best val loss = avg_G_loss = {best_val_loss:.4f}")

        ### === END OF VALIDATION === ###

        # adapt learning-rate
        if training_stage==1:
            scheduler_D_stage1.step(avg_D_loss)
            scheduler_G_stage1.step(avg_G_loss)
        else:
            scheduler_D_stage2.step(avg_D_loss)
            scheduler_G_stage2.step(avg_G_loss)

    ### END OF EPOCH === ###

    time_forTraining = time.time()-time_trainingStart

    print("")
    print("")
    print("===============================================")
    print(f"===== TRAINING ENDED {datetime.now().strftime('%d.%m.%Y, %H-%M-%S')} =====")
    print(f"====== Complete training time: {seconds_to_hhmmss(time_forTraining)} ======")
    print("===============================================")
