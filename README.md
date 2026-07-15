## FIT Senior-Data-Scientist(CV, ML) Mail Assignment - Task2
## To run the program for Task 2, please follow the requirements and procedures outlined below.

### Windows PowerShell

Please open PowerShell or Command Prompt (cmd.exe) in the project directory.

Install the required packages:

```powershell
pip install torch numpy pandas
```

### Verify the Installation

Run:

```bash
python -c "import torch, numpy, pandas; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

A valid installation should print the installed PyTorch version and whether a CUDA GPU is available.

---

## Required Folder Structure

The scripts use paths relative to the current working directory. Therefore, the code files should be placed in a separate code folder, while the training and testing data folders should be located one level above it.

Recommended structure:

```text
MyTask2/
├── Data_Challenge_PHM2022_training_data/
│   ├── data_pin1.csv
│   ├── data_pdmp1.csv
│   ├── data_po1.csv
│   ├── data_pin2.csv
│   ├── data_pdmp2.csv
│   ├── data_po2.csv
│   ├── ...
│   ├── data_pin6.csv
│   ├── data_pdmp6.csv
│   └── data_po6.csv
│
├── Data_Challenge_PHM2022_testing_data/
│   ├── data_pin3.csv
│   ├── data_pdmp3.csv
│   └── data_po3.csv
│
├── results/
│
├── model_weights/
│
├── models.py
├── train-valid-source_domain.py
└── train-pred-source-target-domain.py
```

Before running either script, move into the code directory:

```bash
cd PHM2022_Project/code
```

This step is important because the scripts construct the dataset paths using `os.getcwd()`.

---

## Input Data Format

Each individual must contain three CSV files:

```text
data_pin<individual_id>.csv
data_pdmp<individual_id>.csv
data_po<individual_id>.csv
```

For example, Individual 1 requires:

```text
data_pin1.csv
data_pdmp1.csv
data_po1.csv
```

Each CSV row is expected to follow this format:

```text
label, signal_value_1, signal_value_2, ..., signal_value_n
```

The first column is read as an integer class label. All remaining columns are read as the sensor sequence.

The three sensor files belonging to the same individual must contain the same number of rows.

---

## Dataset Configuration

The source and target individuals are defined directly in the training scripts.

### Validation Script

```python
source_domain_ls = [1, 2, 4, 5, 6]
```

The current validation script uses Individuals 1, 2, 4, 5, and 6.

### Final Training and Prediction Script

```python
source_domain_ls = [1, 2, 4, 5, 6]
target_domain_ls = [3]
```

This configuration uses:

- Individuals 1, 2, 4, 5, and 6 as labeled source domains
- Individual 3 as the target domain

These lists can be changed if a different source-target configuration is required.

---

## How to Run the Program

## Step 1: Run Source-Domain Validation

From the code directory, run:

```bash
python train-valid-source_domain.py
```

### Validation Console Output

The script first prints the number of samples and class distribution for each source individual:

```text
Source individual 1: N=..., label_counts={...}
Source individual 2: N=..., label_counts={...}
...
```

During training, it prints one line for each epoch:

```text
Epoch 001/100 | Loss=... Task-loss=... domain-loss=... mmd-loss=... acc=...
```

The reported values are:

- `Loss`: Sum of the task, domain, and MMD losses
- `Task-loss`: Source-domain classification loss
- `domain-loss`: Domain-classification loss
- `mmd-loss`: Feature-distribution alignment loss
- `acc`: Classification accuracy calculated on the selected validation batch

This script is intended for checking model behavior and selecting the training configuration before final training.

---

## Step 2: Prepare the Results Directory

Before running the final script, create the `results` directory.

### Windows PowerShell

```powershell
New-Item -ItemType Directory -Force results
```

### macOS or Linux

```bash
mkdir -p results
```

The `model_weights` directory is created automatically by the script.

---

## Step 3: Run Final Training and Prediction

Run:

```bash
python train-pred-source-target-domain.py
```

During final training, the program prints:

```text
Epoch 001/100 | Loss=... Task-loss=... domain-loss=... mmd-loss=...
```

The script trains for 100 epochs using:

```text
Batch size:       384 per domain
Learning rate:    0.001
Weight decay:     0.0
Number of classes: 11
Crop length:      300
Random seed:      1234
```

The program automatically selects the execution device:

```text
CUDA GPU, when available
CPU, otherwise
```

---

## Generated Results

After the final script completes, the following outputs are generated.

### 1. Trained Model Weights

```text
model_weights/model_weights.pth
```

This file contains the trained PyTorch `state_dict`.

The model can be loaded using:

```python
import torch
from models import DANN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DANN(
    crop_len=300,
    num_classes=11,
    num_domains=6
).to(device)

model.load_state_dict(
    torch.load(
        "model_weights/model_weights.pth",
        map_location=device
    )
)

model.eval()
```

The number of domains must match the number used during final training.

### 2. Prediction CSV

The intended prediction output is:

```text
results/predictions_test.csv
```

The CSV contains:

| Column | Description |
|---|---|
| `cycle_id` | Sequential sample ID beginning from 1 |
| `predicted_label` | Predicted class label from 1 to 11 |

Example:

```csv
cycle_id,predicted_label
1,3
2,3
3,7
```

### 3. Preprocessed Data

The final training script also calls `numpy.savez_compressed()` during preprocessing.

With the current implementation, a file named approximately:

```text
preprocessed_sensor_sequences.npz
```

may be generated in the code directory and overwritten whenever another individual is processed.

The file contains:

```text
X: preprocessed sensor samples
y: labels or placeholder labels
```

---

## Model Input Shape

After preprocessing, each sample has the following shape:

```text
[3, 3, 300]
```

The dimensions represent:

```text
3 temporal segments:
- Beginning
- Center
- End

3 sensor channels:
- pin
- pdmp
- po

300 time steps per segment
```

The DANN model processes the three temporal segments through three separate CNN encoders and concatenates their latent features before classification and domain discrimination.

---

## Important Implementation Notes

### Validation Design

In the current validation script, the final source-domain loader is also used as the validation/reference domain, while its data remains part of the source training batches.

Therefore, the reported accuracy is useful for monitoring the program, but it is not a strict held-out individual-level validation result.

For strict validation, one individual should be removed from the source training list and loaded separately as the validation domain.

### Final Prediction Block

The current final script retrieves prediction samples using:

```python
next(source_domain_loader_iteration[-1])
```

This reads one batch from the last source domain rather than the complete target domain.

For official target prediction, the inference block should iterate through:

```python
target_domain_dataloaders
```

and concatenate predictions from all target batches.

### Prediction Filename

The current source code contains an extra symbol and tab character in the output filename.

It is recommended to use:

```python
pred_res.to_csv(
    os.path.join(
        os.getcwd(),
        "results",
        "predictions_test.csv"
    ),
    index=False
)
```

### Batch Size and `drop_last`

The DataLoaders currently use:

```python
batch_size = 384
drop_last = True
```

Each domain must contain at least 384 samples. Otherwise, its DataLoader may contain zero complete batches.

When a domain contains fewer than 384 samples, reduce the batch size or set:

```python
drop_last = False
```

### GPU Memory

One batch is loaded from every source domain and the target domain during each update. The effective amount of data processed in one update can therefore be substantially larger than 384 samples.

If a CUDA out-of-memory error occurs, reduce:

```python
batch_size = 128 * 3
```

to a smaller value, such as:

```python
batch_size = 128
```

or:

```python
batch_size = 64
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'models'`

Confirm that `models.py` and both training scripts are located in the same directory, and run the command from that directory.

### `FileNotFoundError`

Confirm that:

- The current terminal location is the code directory
- The training data folder is named exactly `Data_Challenge_PHM2022_training_data`
- The testing data folder is named exactly `Data_Challenge_PHM2022_testing_data`
- Both data folders are located one level above the code directory
- The required `pin`, `pdmp`, and `po` files exist for every configured individual

### Row-Count Mismatch

An error such as:

```text
Row-count mismatch for individual ...
```

means that the `pin`, `pdmp`, and `po` CSV files for the same individual contain different numbers of rows.

### CUDA Out-of-Memory Error

Reduce the batch size in both training scripts.

### Empty DataLoader or `max()` Error

This may occur when `drop_last=True` and one domain contains fewer samples than the configured batch size.

Reduce the batch size or set `drop_last=False`.

### Missing `results` Directory

Create the directory before running the final script:

```bash
mkdir results
```

On Windows PowerShell:

```powershell
New-Item -ItemType Directory -Force results
```

---

## Recommended Execution Summary

```text
1. Place the code and datasets in the required folder structure.
2. Create and activate a Python virtual environment.
3. Install PyTorch, NumPy, and pandas.
4. Move into the code directory.
5. Run train-valid-source_domain.py.
6. Review the training losses and validation accuracy.
7. Create the results directory.
8. Run train-pred-source-target-domain.py.
9. Check model_weights/model_weights.pth.
10. Check results/predictions_test.csv after correcting the target inference block.
```
