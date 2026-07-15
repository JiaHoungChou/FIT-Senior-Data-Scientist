## FIT Senior-Data-Scientist(CV, ML) Mail Assignment - Task2
## To run the program for Task 2, please follow the requirements and procedures outlined below.

### Windows PowerShell

Please open PowerShell or Command Prompt (cmd.exe) in the project directory.

Create a virtual environment for the project and activate it. Before proceeding, please install Anaconda to ensure that your environment is consistent with mine:

```powershell
conda create -n myenv python=3.10
conda activate myenv
```

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
cd MyTask2/
```

This step is important because the scripts construct the dataset paths using `os.getcwd()`.

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

This script is intended to check model behavior and select the training configuration before final training.

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

10. Check results/predictions_test.csv after correcting the target inference block.
```
