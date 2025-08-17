# Discrete Gestures Evaluation

This script provides the commands to evaluate **discrete gesture recognition models** trained on EMG (electromyography) data.
Evaluation is performed using **sliding window inference** and the **CLER** metric as in the original paper.

## Sliding Window Inference

Unlike RNNs (e.g., LSTMs), Transformer-based models cannot always process arbitrarily long input sequences due to **quadratic complexity in sequence length** and GPU memory constraints.

In the original paper, the authors evaluate the LSTM giving the full sequence as input, but this is not feasible with Transformers.

To handle long EMG recordings efficiently, we use a **sliding window inference approach**:

* A **fixed-length window** of EMG samples (e.g., 8 seconds at 2000 Hz) is extracted.
* Windows are processed with **25% overlap** so that each EMG time point is covered multiple times.
* The model outputs **logits per time step** for each window.
* Logits from overlapping windows are **averaged** in the overlapping regions, producing a **continuous sequence of predictions** across the entire recording.

To produce a fair comparison with the original inference method on the LSTM model, we have also reported the CLER using the sliding window approach on the LSTM model. In this way, the EMG Transformer can be evaluated more fairly against the LSTM baseline under the same inference settings.

## Usage

The main script for evaluation is:

```bash
python notebooks/eval_discrete_gestures.py \
    --model_ckpt PATH_TO_CHECKPOINT \
    --config_path PATH_TO_CONFIG \
    [--data_dir PATH_TO_DATASET] \
    [--test_one_sample]
```

### Arguments

* `--model_ckpt` (**required**)
  Path to the trained model checkpoint (`.ckpt`).

* `--config_path` (**required**)
  Path to the Hydra config YAML file used for training.

* `--data_dir` (default: `.../discrete_gestures/`)
  Path to the EMG dataset directory.

* `--test_one_sample` (optional flag)
  If set, runs evaluation on a **single dataset sample** with visualization.
  Otherwise, evaluation runs on the **entire test set**.

---

## Examples

### 1. Test one sample

Run evaluation on a single dataset sample with **sliding window inference** and plot EMG signals, predicted probabilities, and ground-truth labels:

```bash
python notebooks/eval_discrete_gestures.py \
    --model_ckpt notebooks/META/model_checkpoint.ckpt \
    --config_path notebooks/META/model_config.yaml \
    --test_one_sample
```

This saves a figure under:

```
images/discrete_gestures_evaluation_transformer.png
```

---

### 2. Evaluate on full test set

#### Meta (LSTM) – full sequence

```bash
python notebooks/eval_discrete_gestures.py \
    --model_ckpt notebooks/META/model_checkpoint.ckpt \
    --config_path notebooks/META/model_config.yaml
```

* CLER (full sequence): **0.1819** . This is also reported in the `tests` folder
* CLER (sliding window): **0.1596**

---

#### EMG-Transformer (pretrained) – sliding window

```bash
python notebooks/eval_discrete_gestures.py \
    --model_ckpt notebooks/pretrained/epoch=100-step=5555.ckpt \
    --config_path notebooks/pretrained/config.yaml
```

* CLER: **0.1553**

---

#### EMG-Transformer (scratch / supervised) – sliding window

```bash
python notebooks/eval_discrete_gestures.py \
    --model_ckpt notebooks/supervised/epoch=121-step=6710.ckpt \
    --config_path notebooks/supervised/config.yaml
```

* CLER: **0.1594**

---

## Results

| Model                                      | Evaluation Mode | CLER   | Parameters |
| ------------------------------------------ | --------------- | ------ | ---------- |
| **Meta (LSTM)**                            | Full sequence   | 0.1819 | 6.4M       |
| **Meta (LSTM)**                            | Sliding window  | 0.1596 | 6.4M       |
| **EMG-Transformer (pretrained)**           | Sliding window  | 0.1553 | 3.6M       |
| **EMG-Transformer (scratch / supervised)** | Sliding window  | 0.1594 | 3.6M       |

