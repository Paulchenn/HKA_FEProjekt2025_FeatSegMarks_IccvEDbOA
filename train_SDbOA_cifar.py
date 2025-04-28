import numpy as np
import os
import torch
from torch import nn, optim
from torchvision import datasets, transforms
import torch.nn.functional as F

from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from models import resnet, generation
from models.resnet import ResNet18, BasicBlock
from utils.canny import canny
from skimage.color import rgb2gray
from tps_grid_gen import TPSGridGen
from torch.autograd import Variable
import itertools

torch.autograd.set_detect_anomaly(True)

# Modes for Debugging
debug_mode = False
debugIterations_strt = 386  # If Debug Mode is on start at this Iteration
debugIterations_amount = 6     # If Debug Mode is on only do this amount of Iterations

def get_edge(images, sigma=1.0, high_threshold=0.3, low_threshold=0.2, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    # median = kornia.filters.MedianBlur((3,3))
    # for i in range(3):
    #     images = median(images)

    images = images.cpu().numpy()
    edges = []
    for i in range(images.shape[0]):
        img = images[i]

        img = img * 0.5 + 0.5
        img_gray = rgb2gray(np.transpose(img, (1, 2, 0)))
        edge = canny(np.array(img_gray), sigma=sigma, high_threshold=high_threshold,
                     low_threshold=low_threshold).astype(float)
        # edge = Image.fromarray((edge * 255.).astype(np.int8), mode='L')
        edge = (edge - 0.5) / 0.5
        edges.append([edge])
    edges = np.array(edges).astype('float32')
    edges = torch.from_numpy(edges).to(device)
    return edges


def rgb2Gray_batch(input):
    R = input[:, 0]
    G = input[:, 1]
    B = input[:, 2]
    input[:, 0] = 0.299 * R + 0.587 * G + 0.114 * B
    input = input[:, 0]

    input = input.view(input.shape[0], 1, 32, 32)
    return input


def grid_sample(input, grid, canvas=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    output = F.grid_sample(input, grid, align_corners=True).to(device)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1).to(device))
        output_mask = F.grid_sample(input_mask, grid, align_corners=True)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


def TPS_Batch(imgs, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    height, width = imgs.shape[3], imgs.shape[2]
    tps_img = []
    for i in range(imgs.shape[0]):
        img = imgs[i, :, :, :]
        img = img.unsqueeze(0)
        target_control_points = torch.Tensor(list(itertools.product(
            torch.arange(-1.0, 1.00001, 2.0 / 4),
            torch.arange(-1.0, 1.00001, 2.0 / 4),
        ))).to(device)
        source_control_points = target_control_points + torch.Tensor(target_control_points.size()).uniform_(-0.1,
                                                                                                            0.1).to(device)
        # source_control_points = target_control_points + 0.01*torch.ones(target_control_points.size()).to(device)
        tps = TPSGridGen(height, width, target_control_points)
        source_coordinate = tps(Variable(torch.unsqueeze(source_control_points, 0)))

        grid = source_coordinate.view(1, height, width, 2)
        canvas = Variable(torch.Tensor(1, 3, height, width).fill_(1.0)).to(device)
        target_image = grid_sample(img, grid, canvas)
        tps_img.append(target_image)
    tps_img = torch.cat(tps_img, dim=0)
    return tps_img


def get_info(input, batch_size, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    gray = rgb2Gray_batch(input)
    # gray = input
    mat1 = torch.cat([gray[:, :, 0, :].unsqueeze(2), gray], 2)[:, :, :gray.shape[2], :]
    mat2 = torch.cat([gray, gray[:, :, gray.shape[2] - 1, :].unsqueeze(2)], 2)[:, :, 1:, :]
    mat3 = torch.cat([gray[:, :, :, 0].unsqueeze(3), gray], 3)[:, :, :, :gray.shape[3]]
    mat4 = torch.cat([gray, gray[:, :, :, gray.shape[3] - 1].unsqueeze(3)], 3)[:, :, :, 1:]
    info_rec = (gray - mat1) ** 2 + (gray - mat2) ** 2 + (gray - mat3) ** 2 + (gray - mat4) ** 2
    info_rec_ave = info_rec.view(batch_size, -1)
    ave = torch.mean(info_rec_ave, dim=1)
    # info = torch.zeros(gray.shape, dtype=torch.float32)
    tmp = torch.zeros(gray.shape).to(device)
    for b in range(input.shape[0]):
        tmp[b] = ave[b]
    info = torch.where(info_rec > tmp, 1.0, 0.0)

    return info


def show_result(num_epoch, show=False, save=False, path='result.png'):
    zz = torch.randn(64, 100, 1, 1).to(device)
    netG.eval()
    test_images = netG(zz, show_x, bx)
    netG.train()

    size_figure_grid = 8
    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)

    for k in range(64):
        i = k // 8
        j = k % 8
        ax[i, j].cla()
        ax[i, j].imshow(np.transpose(test_images[k].cpu().data.numpy(), (1, 2, 0)))

    label = 'Epoch {0}'.format(num_epoch)
    fig.text(0.5, 0.04, label, ha='center')
    plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


# --- Alles, was ausgeführt werden soll, wenn du train_SDbOA_cifar.py direkt laufen lässt ---
if __name__ == "__main__":

    # --- Zusammenfassung ---
    # - Daten werden vorbereitet mit normalen Augmentierungen.
    # - Es handelt sich um ein GAN-Setup: Generator (netG), Diskriminator (netD) und einen Klassifikator (cls), basierend auf ResNet18.
    # - Drei Optimizer und drei Loss-Funktionen sind definiert – Ziel: Kombination aus GAN-Training und Klassifikationsaufgabe.
    # - Resize-Objekte (re12, re32) könnten für Custom-Image-Verarbeitung oder Encoder/Decoder verwendet werden.

    # --- Datenvorverarbeitung (Trainingsdaten) ---
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Zufälliges horizontales Spiegeln – hilft beim Data Augmentation
        transforms.ToTensor(),              # Wandelt ein PIL-Bild oder NumPy-Array in ein Tensor um
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalisiert RGB-Werte auf [-1, 1]
    ])

    # --- Datenvorverarbeitung (Testdaten) ---
    transform_test = transforms.Compose([
        transforms.ToTensor(),              # Nur Umwandlung in Tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Und Normalisierung (gleich wie beim Training)
    ])

    # --- CIFAR-10 Datensätze laden ---
    # Trainingsdatensatz: wird beim ersten Mal heruntergeladen und transformiert
    cifar10_train = datasets.CIFAR10(
        root='./src/cifar10', train=True, download=True, transform=transform_train
    )

    # Testdatensatz: ebenfalls mit den passenden Transformationen
    cifar10_test = datasets.CIFAR10(
        root='./src/cifar10', train=False, download=True, transform=transform_test
    )

    # Take cuda if availabe (NVIDIA) if not then cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set best accuracy to 0 to be sure that first accuracy is better
    best_acc = 0

    # --- Trainingsparameter ---
    batch_size = 128         # Anzahl der Bilder pro Trainingsbatch
    lr = 1e-4                # Lernrate für Generator und Diskriminator
    epochs = 120             # Anzahl der Trainingsepochen

    # --- Resize-Operationen (vermutlich für Upsampling/Downsampling) ---
    re12 = transforms.Resize((12, 12))  # Verkleinert Bilder auf 12x12
    re32 = transforms.Resize((32, 32))  # Skaliert Bilder auf 32x32 (Originalgröße CIFAR-10)

    # --- Netzwerke initialisieren ---
    netG = generation.generator(128)   # Generator-Netzwerk mit Eingabegröße 128 (vermutlich latenter Vektor)
    netD = generation.Discriminator()  # Diskriminator zur Unterscheidung von real/fake Bildern
    cls = ResNet18(BasicBlock, num_classes=10)            # Klassifikator (hier: ResNet-18)

    # Netzwerke auf GPU/CPU verschieben
    netG = netG.to(device)
    netD = netD.to(device)
    cls = cls.to(device)

    # --- Optimizer definieren ---
    optimizerD = optim.Adam(
        netD.parameters(), lr=lr, betas=(0., 0.99)
    )  # Optimizer für Diskriminator (GAN-typische Betas)
    optimizerG = optim.Adam(
        netG.parameters(), lr=lr, betas=(0., 0.99)
    )  # Optimizer für Generator
    optimizerC = optim.Adam(
        cls.parameters(), lr=1e-3, betas=(0., 0.99)
    )  # Optimizer für Klassifikator (mit höherer Lernrate)

    # --- DataLoader für Batch-Training ---
    train_loader = DataLoader(
        dataset=cifar10_train, batch_size=batch_size, shuffle=True#, drop_last=True
    )  # Trainingsdaten gemischt in Batches
    test_loader = DataLoader(
        dataset=cifar10_test, batch_size=64
    )  # Testdaten (kleinerer Batch reicht hier oft)

    # --- Verlustfunktionen ---
    L1_loss = nn.L1Loss()                # Absoluter Fehler (z. B. für Rekonstruktion)
    MSE_loss = nn.MSELoss()              # Quadratischer Fehler (z. B. für Bildvergleich)
    CE_loss = nn.CrossEntropyLoss()      # Klassifikationsverlust (für multi-class outputs wie CIFAR-10 Klassen)

    for i, (img, label) in enumerate(test_loader):
        img = img.to(device)  # Übertragen der Bilder auf die GPU
        label = label.to(device)  # Übertragen der Labels auf die GPU

        netD.zero_grad()  # Setzt die Gradienten des Diskriminators (netD) auf null, um die Gradienten im aktuellen Durchgang zu löschen

        mn_batch = img.shape[0]  # Bestimmt die Batch-Größe, indem es die Anzahl der Bilder im Batch abruft

        # --- Bildverarbeitung: Generierung von Features ---
        # `get_edge()` erstellt ein Kantenbild des Eingabebildes unter Verwendung eines Kantenerkennungsfilters
        generated1 = get_edge(img, sigma=1.0, high_threshold=0.3, low_threshold=0.2)

        # `get_info()` erstellt ein weiteres Feature-Bild, das auf den Eingabebildern basiert
        generated2 = get_info(img, mn_batch)

        # --- Bildveränderung ---
        # Werte von `generated1` werden in [0, 1] normalisiert (negative Werte werden zu 0, positive zu 1)
        generated1 = torch.where(generated1 < 0, 0., 1.)

        # `generated2` wird invertiert (Multiplikation mit -1)
        generated2 = generated2 * -1

        # Werte von `generated2` werden ebenfalls in [0, 1] normalisiert
        generated2 = torch.where(generated2 < 0, 0., 1.)

        # Kombinieren der beiden generierten Bilder (durch Addition)
        combined = generated2 + generated1
        combined = torch.cat([combined, combined, combined], 1)  # Kombinieren in den 3 Farbkanälen (RGB)

        # --- Bildgröße ändern (vermutlich für zusätzliche Verarbeitung) ---
        blur_img = re12(img)  # Verkleinert das Bild auf 12x12
        blur_img = re32(blur_img)  # Skaliert das Bild auf 32x32 zurück (wieder die CIFAR-10 Originalgröße)

        # --- Schleifensteuerung ---
        if i > 0:  # Hier wird die Schleife nach dem ersten Batch gestoppt
            break

    # --- Zuordnung der bearbeiteten Bilder zu Variablen für spätere Nutzung ---
    show_x = combined  # Kombiniertes Bild (mit Kanten und Info)
    bx = blur_img  # Das unscharf veränderte Bild (nach Resize)

    # Schleife für das Training über die angegebene Anzahl an Epochen
    for epoch in range(epochs):
        print(epoch)  # Gibt die aktuelle Epoche aus
        netG.train()  # Setzt den Generator in den Trainingsmodus
        netD.train()  # Setzt den Diskriminator in den Trainingsmodus
        cls.train()  # Setzt das Klassifizierungsnetzwerk in den Trainingsmodus

        # Nach der 60. Epoche wird die Lernrate des Diskriminators und des Generators verringert
        if epoch == 60:
            optimizerD = optim.Adam(netD.parameters(), lr=lr * 0.1, betas=(0., 0.99))
            optimizerG = optim.Adam(netG.parameters(), lr=lr * 0.1, betas=(0., 0.99))

        # Schleife für das Training über den gesamten Datensatz
        for i, (img, label) in enumerate(train_loader):
            # For Debugging
            if debug_mode and i < debugIterations_strt:
                continue

            img = img.to(device)  # Überträgt das Bild auf die GPU
            label = label.to(device)  # Überträgt das Label auf die GPU

            netD.zero_grad()  # Setzt die Gradienten des Diskriminators auf Null

            mn_batch = img.shape[0]  # Bestimmt die Batch-Größe

            # Bildverarbeitung, um Kanten und zusätzliche Informationen zu extrahieren
            generated1 = get_edge(img, sigma=1.0, high_threshold=0.3, low_threshold=0.2)
            generated2 = get_info(img, img.shape[0])

            generated1 = torch.where(generated1 < 0, 0., 1.)  # Kantenbild normalisieren
            generated2 *= -1  # Invertiere das Informationsbild
            generated2 = torch.where(generated2 < 0, 0., 1.)  # Normalisieren
            combined = generated2 + generated1  # Kombiniere beide Ergebnisse
            combined = torch.cat([combined, combined, combined],
                                1).detach().to(device)  # Zu einem 3-Kanal-Bild kombinieren und auf GPU verschieben

            combined = TPS_Batch(combined)  # Anwendung einer Transformation auf das kombinierte Bild

            # Bild unscharf machen und auf neue Größe ändern
            blur_img = re12(img)
            blur_img = re32(blur_img)

            # Diskriminator-Bewertung des echten Bildes
            D_result, aux_output = netD(img)
            D_result = D_result.squeeze()  # Entfernen der überflüssigen Dimensionen

            D_result_1 = -D_result.mean()  # Berechne den Verlust für den Diskriminator (Teil 1)

            # Erzeugung eines zufälligen Eingabeverteilers für den Generator
            z_ = Variable(torch.randn((mn_batch, 100)).view(-1, 100, 1, 1).to(device))

            # Generiere ein Bild mit dem Generator
            G_result = netG(z_, combined, blur_img)

            # Diskriminator-Bewertung des generierten Bildes
            D_result, _ = netD(G_result)
            D_result = D_result.squeeze()
            D_result_2 = D_result.mean()  # Berechne den Verlust für den Diskriminator (Teil 2)

            # Berechne den Klassifikationsverlust für den Diskriminator
            D_celoss = CE_loss(aux_output, label)

            # Gesamter Diskriminator-Verlust
            D_loss = D_result_1 + D_result_2 + 0.5 * D_celoss
            D_loss.backward()  # Gradienten berechnen
            optimizerD.step()  # Optimierer für den Diskriminator Schritt ausführen

            # Jetzt wird der Generator trainiert (nachdem der Diskriminator trainiert wurde)
            netG.zero_grad()  # Setzt die Gradienten des Generators auf Null

            z_ = Variable(torch.randn((mn_batch, 100)).view(-1, 100, 1, 1).to(device))  # Neuer Zufallswert für den Generator

            # Erzeuge das Bild mit dem Generator
            G_result = netG(z_, combined, blur_img)

            # Berechne den Verlust für den Diskriminator und den Generator
            D_result, aux_output = netD(G_result)
            D_result = D_result.squeeze()

            # Falls img nur 1 Kanal hat, auf 3 duplizieren
            img_for_loss = img.repeat(1, 3, 1, 1) if img.shape[1] == 1 else img
            G_L1_loss = L1_loss(G_result, img_for_loss)  # L1-Verlust zwischen generiertem Bild und Originalbild

            D_result = D_result.mean()  # Berechne den Durchschnitt der Diskriminator-Ergebnisse

            G_celoss = CE_loss(aux_output, label)  # Kreuzentropie-Verlust für den Generator
            G_celoss = G_celoss.sum()  # Summiere alle Kreuzentropie-Verluste

            # Klassifikationsverlust für das Generatorausgabe-Bild
            cls_output = cls(G_result)
            cls_loss = CE_loss(cls_output, label)

            # Verlust der Kanteninformationen
            combined_gray = combined[:, 0:1, :, :]  # Nimm nur den ersten Kanal (z. B. R)
            edge_loss = L1_loss(get_info(G_result.clone(), img.shape[0]), combined_gray)

            # Gesamter Verlust des Generators
            total_loss = G_L1_loss - D_result + 0.5 * G_celoss + cls_loss + edge_loss
            total_loss.backward()  # Gradienten berechnen
            optimizerG.step()  # Optimierer für den Generator Schritt ausführen

            # Alle x Iterationen den aktuellen Verlust ausgeben
            if i % 5 == 0:
                print('Epoch[', epoch + 1, '/', epochs, '][', i + 1, '/', len(train_loader), ']: TOTAL_LOSS', total_loss.item())

            # For debuging (Training of even one Epoch takes very long)
            if debug_mode and i >= debugIterations_strt+debugIterations_amount:
                break

        # Speichern von Ergebnissen und Modellen nach jeder Epoche
        path2save = './Result/cifar_gan/visualization'
        fixed_p = path2save + '/' + str(epoch) + '.png'
        if not os.path.exists(path2save):
            os.makedirs(path2save)
        try:
            show_result(epoch, path=fixed_p)  # Zeigt das Ergebnis der aktuellen Epoche an
        except IndexError as e:
            print(f"IndexError: {e}")
        torch.save(netG.state_dict(), './Result/cifar_gan/tuned_G_' + str(epoch) + '.pth')  # Speichert den Generator
        torch.save(netD.state_dict(), './Result/cifar_gan/tuned_D_' + str(epoch) + '.pth')  # Speichert den Diskriminator

        # Wechsel in den Evaluierungsmodus (für Testphase)
        cls.eval()
        netG.eval()
        netD.eval()

        correct = torch.zeros(1).squeeze().to(device)  # Zähler für korrekte Vorhersagen
        total = torch.zeros(1).squeeze().to(device)  # Gesamtzahl der Bilder

        # Testphase
        for i, (img, label) in enumerate(test_loader):
            # For Debugging
            if debug_mode and i < debugIterations_strt:
                continue

            img = img.to(device)  # Bild auf GPU verschieben
            label = label.to(device)  # Label auf GPU verschieben

            # Verarbeitung der Testbilder ähnlich wie im Training
            generated1 = get_edge(img, sigma=1.0, high_threshold=0.3, low_threshold=0.2)
            #generated2 = get_info(img)
            generated2 = get_info(img, img.shape[0])
            generated1 = torch.where(generated1 < 0, 0., 1.)
            generated2 *= -1
            generated2 = torch.where(generated2 < 0, 0., 1.)
            combined = generated2 + generated1
            combined = torch.cat([combined, combined, combined], 1).detach().to(device)

            z_ = torch.randn((img.shape[0], 100)).view(-1, 100, 1, 1).to(device)  # Zufällige Eingabeverteilung

            # Bild unscharf machen und skalieren
            blur_img = re12(img)
            blur_img = re32(blur_img)

            # Generator erzeugt Testbilder
            G_result = netG(z_, combined, blur_img)

            # Klassifikationsnetzwerk bewertet das generierte Bild
            output = cls(G_result)
            prediction = torch.argmax(output, 1)
            correct += (prediction == label).sum().float()  # Zählt die korrekten Vorhersagen
            total += len(label)  # Zählt die Gesamtanzahl der Testbilder#

            # For debuging (Training of even one Epoch takes very long)
            if debug_mode and i >= debugIterations_strt+debugIterations_amount:
                break

        acc = (correct / total).cpu().detach().data.numpy()  # Berechnet die Genauigkeit
        print('Epoch: ', epoch + 1, ' test accuracy: ', acc)

        # Speichert das Modell, wenn die beste Genauigkeit erreicht wurde
        if acc > best_acc:
            best_acc = acc
            print('Best accuracy: ', acc)
            torch.save(cls.state_dict(), './Result/best_cls.pth')  # Speichert das beste Klassifizierungsmodell




