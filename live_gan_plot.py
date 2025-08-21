import csv
import matplotlib.pyplot as plt
import os
import pdb
#from pynput.keyboard import Controller

# === CONFIG ===
#print(os.getcwd())
result_dir = "./Result"

# === Funktionen ===
def get_correct_path(result_dir):
    available_results = [d for d in os.listdir(result_dir) if os.path.isdir(os.path.join(result_dir, d)) and d.startswith("2025")]
    available_results = sorted(available_results, reverse=True)
    train_csv_path = os.path.join(result_dir, available_results[0], "metrics", "iterations_metrics.csv") if available_results else None
    val_csv_path = os.path.join(result_dir, available_results[0], "metrics", "epoch_metrics.csv") if available_results else None

    return train_csv_path, val_csv_path

def moving_average(values, window_size):
    if len(values) < window_size:
        return values
    smoothed = []
    for i in range(len(values)):
        window = values[max(0, i - window_size + 1):i + 1]
        smoothed.append(sum(window) / len(window))
    return smoothed

def safe_read_csv_raw(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t' if path.endswith('.tsv') else ',')
            return list(reader)
    except Exception as e:
        print(f"[WARN] Could not read {path}: {e}")
        return None

def parse_float_list(data, key):
    try:
        return [float(row[key]) for row in data]
    except Exception:
        return []

def parse_int_list(data, key):
    try:
        return [int(row[key]) for row in data]
    except Exception:
        return []

def plot_gan_logs(axs, result_dir):
    train_csv_path, val_csv_path = get_correct_path(result_dir)

    train_data = safe_read_csv_raw(train_csv_path)
    val_data = safe_read_csv_raw(val_csv_path)

    if train_data:
        epoch = parse_int_list(train_data, 'epoch')
        iteration = parse_int_list(train_data, 'iteration')
        d_loss_raw = parse_float_list(train_data, 'discriminator_loss')
        g_loss_raw = parse_float_list(train_data, 'generator_loss')

        N = max(iteration) if iteration else 1000
        smooth_window = 50
        global_iter = [(e-1) * N + i for e, i in zip(epoch, iteration)]
        d_loss_smooth = moving_average(d_loss_raw, smooth_window)
        g_loss_smooth = moving_average(g_loss_raw, smooth_window)

        axs[0, 0].clear()
        axs[0, 0].plot(global_iter, d_loss_raw, alpha=0.3, label='Disc Loss (raw)')
        axs[0, 0].plot(global_iter, d_loss_smooth, color='blue', label='Disc Loss (smoothed)')
        axs[0, 0].set_title("Train: Discriminator Loss")
        axs[0, 0].set_xlabel("Global Iteration")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        axs[0, 1].clear()
        axs[0, 1].plot(global_iter, g_loss_raw, alpha=0.3, label='Gen Loss (raw)')
        axs[0, 1].plot(global_iter, g_loss_smooth, color='orange', label='Gen Loss (smoothed)')
        axs[0, 1].set_title("Train: Generator Loss")
        axs[0, 1].set_xlabel("Global Iteration")
        axs[0, 1].set_ylabel("Loss")
        axs[0, 1].legend()
        axs[0, 1].grid(True)

    if val_data:
        epoch_val = parse_int_list(val_data, 'epoch')
        d_loss_val = parse_float_list(val_data, 'discriminator_loss')
        g_loss_val = parse_float_list(val_data, 'generator_loss')

        axs[1, 0].clear()
        axs[1, 0].plot(epoch_val, d_loss_val, color='green', label='Disc Loss (Val)')
        axs[1, 0].set_title("Validation: Discriminator Loss")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("Loss")
        axs[1, 0].legend()
        axs[1, 0].grid(True)

        axs[1, 1].clear()
        axs[1, 1].plot(epoch_val, g_loss_val, color='red', label='Gen Loss (Val)')
        axs[1, 1].set_title("Validation: Generator Loss")
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Loss")
        axs[1, 1].legend()
        axs[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.draw()

def on_key(event):
    if event.key == 'r':
        print("Reloading plots...")
        plot_gan_logs(axs, result_dir)
        print("... finished.")
    elif event.key == 'c':
        print("Clearing plots...")
        for ax in axs.flat:
            ax.clear()
        fig.canvas.draw()
        print("... finished.")
    elif event.key == 'q':
        print("Exiting.")
        plt.close()

# === Setup ===
fig, axs = plt.subplots(2, 2, figsize=(8.5, 6))
fig.suptitle("GAN Training & Validation Losses (press 'r' to reload, 'q' to quit)")

plot_gan_logs(axs, result_dir)
fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

while(1):
    kb = Controller()
    kb.press('r'); kb.release('r')