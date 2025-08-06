import os
import argparse

from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
from pytorch_lightning import Trainer

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from matplotlib import pyplot as plt
import seaborn as sns

from generic_neuromotor_interface.cler import GestureType, compute_cler
from generic_neuromotor_interface.data import make_dataset

def merge_overlapping_logits_by_time(logits_list, num_classes, output_seq_len, output_step):
    """
    Merges a list of 2D logits from overlapping windows into a single 2D tensor,
    assuming the input tensor shape is (num_classes, output_seq_len).

    Args:
        logits_list (list): List of 2D tensors, each shaped (num_classes, output_seq_len).
        num_classes (int): The number of classes (e.g., 9).
        output_seq_len (int): The length of the time sequence from a single window.
        output_step (int): The step size (stride) in the output space.
    """
    # Calculate the total length of the final merged time sequence
    total_time_steps = output_seq_len + (len(logits_list) - 1) * output_step

    # Initialize tensors with the correct final shape: (num_classes, total_time_steps)
    merged_logits = torch.zeros((num_classes, total_time_steps))
    counts = torch.zeros((num_classes, total_time_steps))

    for i, logits in enumerate(logits_list):
        start_index = i * output_step
        end_index = start_index + output_seq_len
        # Add the logits from the current window, slicing along the time dimension (dim 1)
        merged_logits[:, start_index:end_index] += logits
        # Increment the counts in the same region
        counts[:, start_index:end_index] += 1

    # Average the logits in the overlapping regions
    counts[counts == 0] = 1
    return merged_logits / counts

def get_logits(model, emg_sample, freq=2_000, device="cuda"):
    """
    Run forward pass on a single EMG sample using sliding window inference.
    """
    W = 8 * freq  # window length in samples
    S = W // 4 # 25% overlap

    logits_list = []
    C, T = emg_sample.shape
    with torch.no_grad():
        for i in range(0, T - W + 1, S):
            emg_window = emg_sample[:, i:i+W]
            logits = model(emg_window.unsqueeze(0).contiguous().to(device))
            logits_list.append(logits[0].cpu())
    
    # Now we have several logits tensors, each of shape (num_classes, output_seq_len)
    # they need to be merged into a single tensor which in theory should be close to the
    # one obtained with a single forward pass on the entire sequence.
    # Since running the full sequence is not feasible for Transformer models,
    # we will merge the logits by averaging them over the overlapping regions.
    num_classes, output_seq_len = logits_list[0].shape

    output_step = S // model.network.stride

    merged_logits = merge_overlapping_logits_by_time(
        logits_list, num_classes, output_seq_len, output_step
    )
    return merged_logits


if __name__ == "__main__":
    #torch.backends.cudnn.enabled = False

    parser = argparse.ArgumentParser(description="Evaluate discrete gestures model.")
    parser.add_argument("--data_dir", type=str, default="/capstor/scratch/cscs/mfasulo/datasets/generic-neuromotor-interface/discrete_gestures/", help="Path to EMG data directory.")
    parser.add_argument("--model_ckpt", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to model configuration file.")
    parser.add_argument("--test_one_sample", action="store_true", help="If set, test one sample from the dataset.")
    args = parser.parse_args()

    TASK_NAME = "discrete_gestures"
    EMG_DATA_DIR = args.data_dir
    MODEL_CKPT = args.model_ckpt
    CONFIG_FILE = args.config_path
    DEVICE = "cuda"

    print(f"EMG Data Directory: {EMG_DATA_DIR}")
    print(f"Model Checkpoint: {MODEL_CKPT}")
    print(f"Configuration File: {CONFIG_FILE}")

    if not os.path.exists(os.path.expanduser(EMG_DATA_DIR)):
        raise FileNotFoundError(f"The EMG data path does not exist: {EMG_DATA_DIR}")
    if not os.path.exists(os.path.expanduser(MODEL_CKPT)):
        raise FileNotFoundError(f"The model checkpoint does not exist: {MODEL_CKPT}")
    if not os.path.exists(os.path.expanduser(CONFIG_FILE)):
        raise FileNotFoundError(f"The configuration file does not exist: {CONFIG_FILE}")

    config = OmegaConf.load(CONFIG_FILE)

    model = instantiate(config.lightning_module)
    model = model.load_from_checkpoint(
        MODEL_CKPT,
        map_location=torch.device("cpu")
    )
    
    # Update DataModule config with data path
    config["data_module"]["data_location"] = os.path.expanduser(EMG_DATA_DIR)
    if "from_csv" in config["data_module"]["data_split"]["_target_"]:
        config["data_module"]["data_split"]["csv_filename"] = os.path.join(
            os.path.expanduser(EMG_DATA_DIR),
            f"{TASK_NAME}_corpus.csv"
        )

    datamodule = instantiate(config["data_module"])

    if args.test_one_sample:
        test_dataset = make_dataset(
            datamodule.data_location,
            partition_dict={"discrete_gestures_user_002_dataset_000": None},
            transform=datamodule.transform,
            emg_augmentation=None,
            window_length=None,
            stride=None,
            jitter=False,
        )

        print("Testing on one sample from the dataset with sliding window inference...")

        sample = test_dataset[0]
        model.eval()
        model.to(DEVICE)

        # Unpack sample
        emg = sample["emg"]
        emg_times = sample["timestamps"]
        labels = sample["prompts"]

        logits = get_logits(model, emg, device=DEVICE)

        probs = torch.nn.Sigmoid()(logits)

        prob_times = emg_times[model.network.left_context::model.network.stride]
        # The length of prob_times must match the number of time steps in probs (dim 1)
        prob_times = prob_times[:probs.shape[1]]

        cler = compute_cler(probs, prob_times, labels)
        print("CLER on this dataset:", cler)

        fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, sharey=False)

        t0 = emg_times[0]

        # plot EMG
        ax = axes[0]
        spacing = 200
        for channel_index, channel_data in enumerate(emg):
            ax.plot(
                emg_times - t0,
                channel_data + channel_index * spacing,
                linewidth=1,
                color="0.7",
            )

        ax.set_ylim([-spacing, len(emg) * spacing])
        ax.set_yticks([])

        sns.despine(ax=ax, left=True)

        # labels
        ax = axes[1]
        for gesture in GestureType:
            prob_index = gesture.value
            ax.plot(
                prob_times - t0,
                probs[prob_index] + prob_index,
                linewidth=1,
                label=gesture.name
            )
        ax.set_yticks([])

        sns.despine(ax=ax, left=True)

        legends, handles = ax.get_legend_handles_labels()
        ax.legend(
            legends[::-1],
            handles[::-1],
            loc="upper left",
            ncols=1,
            bbox_to_anchor=(1.0, 1.0),
            frameon=False
        )
        ax.set_xlim([352, 357])

        axes[0].set_ylabel("EMG\n(normalized)")
        axes[1].set_ylabel("predicted gesture\nprobability")
        axes[1].set_xlabel("time\n(sec)")

        tmin, tmax = ax.get_xlim()
        _, ymax = axes[0].get_ylim()

        labels_in_window = False

        for label in labels.to_dict(orient="records"):
            gesture_name = label["name"]
            t = label["time"] - t0
            if (t > tmin) and (t < tmax):
                labels_in_window = True
                lines = axes[0].axvline(t, color="k")
                axes[0].text(
                    t - 0.075,
                    ymax + 200,
                    gesture_name,
                    rotation="vertical",
                    va="top",
                    ha="left"
                )

        if labels_in_window:
            axes[0].legend(
                [lines],
                ["ground truth labels"],
                loc="upper left",
                ncols=1,
                bbox_to_anchor=(1.0, 1.0),
                frameon=False,
            )

        # save the figure
        plt.tight_layout()
        plt.savefig("images/discrete_gestures_evaluation_transformer.png", dpi=300)
        print("Figure saved!")

    else:
        print("Testing on all samples from the dataset with sliding window inference...")

        model.eval()
        model.to(DEVICE)

        # Use the test dataloader from the DataModule
        datamodule.setup("test")
        test_dloader = datamodule.test_dataloader()

        all_preds = []
        all_times = []
        all_prompts = []
        for sample in tqdm(test_dloader, desc="Testing all samples"):
            emg = sample["emg"][0]
            times = sample["timestamps"][0]
            prompts = sample["prompts"][0]

            logits = get_logits(model, emg, device=DEVICE)

            preds = torch.nn.Sigmoid()(logits)
            preds = preds.squeeze(0).detach().cpu().numpy()
            times = times[model.network.left_context :: model.network.stride]
            times = times[:preds.shape[1]]

            all_preds.append(preds)
            all_times.append(times)
            all_prompts.append(prompts)

        # Concatenate all predictions, times, and prompts
        final_preds = np.concatenate(all_preds, axis=1)
        final_times = np.concatenate(all_times)
        final_prompts = pd.concat(all_prompts, ignore_index=True)

        cler = compute_cler(final_preds, final_times, final_prompts)
        print(f"CLER on the entire test set: {cler}")


# Meta (LSTM) - full sequence
# 0.1819 CLER

# Meta (LSTM) - sliding window
# 0.1596 CLER
# python notebooks/eval_discrete_gestures.py --model_ckpt /capstor/scratch/cscs/mfasulo/checkpoints/META/model_checkpoint.ckpt --config_path /capstor/scratch/cscs/mfasulo/checkpoints/META/model_config.yaml

# EMG-Transformer - sliding window
# 0.1599 CLER
# python notebooks/eval_discrete_gestures.py --model_ckpt logs/2025-08-04/11-39-22/lightning_logs/version_1386059/checkpoints/epoch\=111-step\=6160.ckpt --config_path logs/2025-08-04/11-39-22/hydra_configs/config.yaml
