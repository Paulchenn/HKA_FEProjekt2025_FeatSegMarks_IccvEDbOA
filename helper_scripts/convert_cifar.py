import pickle
import numpy as np
from PIL import Image
import os

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

# Klassen-Namen
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Erstelle die Ordnerstruktur
base_dir = './src/cifar10_batches_jpg'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)

for name in class_names:
    class_dir = os.path.join(base_dir, name)
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

# Passe diesen Pfad an, wenn deine Datendateien woanders liegen
original_data_path = './src/cifar10_batches_py/' 

# Verarbeite die Datendateien
for i in range(1, 6):
    data_batch_file = f'{original_data_path}data_batch_{i}'
    data_batch = unpickle(data_batch_file)
    data = data_batch[b'data']
    labels = data_batch[b'labels']

    for idx, (img_data, label) in enumerate(zip(data, labels)):
        # Konvertiere das Array in ein Bild
        img_reshaped = img_data.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img_reshaped)
        
        # Speichere das Bild
        class_name = class_names[label]
        file_path = os.path.join(base_dir, class_name, f'img_{idx+1}_batch_{i}.jpg')
        img.save(file_path, 'JPEG')

print("Datensatz erfolgreich in die gewünschte Ordnerstruktur konvertiert und als JPG gespeichert.")