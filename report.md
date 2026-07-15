# Report: Domain Adaptation-Based Fault Diagnosis for the 2022 PHM Conference Data Challenge

## 1. Overview

This report describes the implemented approach for the 2022 PHM Conference Data Challenge on hydraulic rock drill fault diagnosis. The goal is to classify the health condition of each impact cycle using three pressure signals collected from the hydraulic rock drill system.

The implementation was developed by referring to the second-place paper in the 2022 PHM Conference Data Challenge: **"Domain Adaptation based Fault Diagnosis under Variable Operating Conditions of a Rock Drill"** by Kim et al. The referenced paper proposed a domain-adaptation-based fault diagnosis framework using three-crop data preprocessing, a convolutional neural network, domain-adversarial learning, maximum mean discrepancy (MMD), and soft-voting ensemble. The main idea is to reduce the domain shift caused by different rock drill individuals and operating conditions.

In this project, the problem is formulated as a multi-source domain adaptation task. Several labeled training individuals are used as source domains, while another individual can be used either as an internal validation domain or as an unlabeled target domain. When the target data are unlabeled, they are not used for classification loss. Instead, they are used for domain alignment through the domain discriminator and MMD loss.

## 2. Reference Paper

This implementation follows the main concept of the second-place solution:

> Kim, Y. C., Kim, T., Ko, J. U., Lee, J., & Kim, K. (2023). *Domain Adaptation based Fault Diagnosis under Variable Operating Conditions of a Rock Drill*. International Journal of Prognostics and Health Management.

The referenced paper introduces three important ideas that were adopted in this implementation:

1. **Three-crop preprocessing**: each impact cycle is represented by the first, center, and last parts of the signal.
2. **Domain-adversarial neural network (DANN)**: a domain discriminator and gradient reversal layer are used to learn domain-invariant features.
3. **MMD-based feature alignment**: the feature distribution between source domains and target domain is reduced using maximum mean discrepancy.

The paper also reports that the method ranked second in the 2022 PHM Conference Data Challenge and achieved 99.77% accuracy on the official final validation data. In this implementation, the main architecture and training logic were reproduced based on the paper, but the code was written independently.

## 3. Dataset Description

Each row in the dataset represents one impact cycle. The first column is the label, and the remaining columns are time-series pressure measurements. Since different rows may have different sequence lengths, the dataset is read using Python `csv.reader` instead of `pandas.read_csv()`.

Three pressure signals are used:

| Signal | Description in this implementation |
|---|---|
| `pin` | Percussion pressure signal |
| `pdmp` | Damper pressure signal |
| `po` | Pressure signal behind the piston |

The original labels are from 1 to 11. Label 1 represents the no-fault condition, and labels 2 to 11 represent different fault conditions. For unlabeled target data, the first column may contain 0. This value is treated as a dummy label and is not used to compute classification loss.

## 4. Approach

### 4.1 Data Loading

For each individual, the three sensor files are loaded separately:

```text
data_pin{individual}.csv
data_pdmp{individual}.csv
data_po{individual}.csv
```

The same row index across the three sensor files is assumed to represent the same impact cycle. Therefore, the three sensor sequences are merged by row index to form one multi-channel sample.

### 4.2 Normalization

Each one-dimensional sensor sequence is normalized independently using z-score normalization:

```python
x_norm = (x - mean(x)) / (std(x) + eps)
```

This normalization reduces the effect of pressure-scale differences across individuals and encourages the model to learn waveform shape and fault-related dynamic patterns.

### 4.3 Three-Crop Preprocessing

Following the second-place paper, each impact cycle is converted into three fixed-length segments:

| Crop | Definition | Length |
|---|---|---:|
| First crop | First 300 samples | 300 |
| Center crop | Center 300 samples | 300 |
| Last crop | Last 300 samples | 300 |

The final input shape of each sample is:

```text
[3 crops, 3 sensors, 300 time points]
```

This is not a standard sliding-window method. A standard sliding window would generate a variable number of windows depending on the sequence length and step size. In contrast, this method always generates exactly three crops for each impact cycle. The reason is to preserve the physical interpretation of the beginning, middle, and ending stages of the impact cycle.

### 4.4 Source and Target Domain Design

The source domains are the labeled individuals. In the current experiment, the following training individuals were loaded:

```python
source_domain_ls = [1, 2, 4, 5, 6]
```

Each source individual is assigned a source domain ID:

| Individual | Domain ID |
|---:|---:|
| 1 | 0 |
| 2 | 1 |
| 4 | 2 |
| 5 | 3 |
| 6 | 4 |

When an additional unlabeled target individual is used, it should be loaded outside the source-domain loop. To avoid confusion, the target domain ID is assigned as:

```python
target_domain_id = len(source_domain_ls)
```

Therefore, if five source domains are used, the target domain ID is 5. In that case, the domain discriminator output dimension must be:

```python
num_domains = len(source_domain_ls) + 1
```

This design ensures that the domain discriminator can distinguish all source domains and the target domain.

## 5. Model Architecture

The model is based on a domain-adversarial neural network. The input contains three crops, and each crop is processed by a CNN feature extractor:

```text
First crop  -> CNN feature extractor
Center crop -> CNN feature extractor
Last crop   -> CNN feature extractor
```

The extracted features are concatenated into a single latent representation. This latent representation is then sent to two branches:

1. **Fault classifier**: predicts one of the 11 fault classes.
2. **Domain discriminator**: predicts the domain ID.

The fault classifier is optimized only with labeled source data. The domain discriminator is optimized with both source and target data.

## 6. Loss Function

The total loss is defined as:

```text
Total Loss = Classification Loss + Domain Loss + MMD Loss
```

### 6.1 Classification Loss

Classification loss is computed only on source samples:

```python
task_loss = cross_entropy(class_logits_source, y_source - 1)
```

The subtraction is required because the original labels are from 1 to 11, while PyTorch `cross_entropy` requires class indices from 0 to 10.

### 6.2 Domain-Adversarial Loss

The domain loss is calculated using both source and target samples. A gradient reversal layer is applied before the domain discriminator. Therefore, the domain discriminator learns to identify the source or target domain, while the feature extractor learns to produce domain-invariant features.

The gradient reversal coefficient is gradually increased during training:

```python
lambda = 2 / (1 + exp(-10 * progress)) - 1
```

This prevents the adversarial loss from dominating the model at the beginning of training.

### 6.3 MMD Loss

The MMD loss is used to align the feature distribution between each source domain and the target domain. For each training step, the MMD loss is calculated between each source feature batch and the target feature batch:

```text
MMD(source 1, target)
MMD(source 2, target)
MMD(source 4, target)
MMD(source 5, target)
MMD(source 6, target)
```

The final MMD term is the average of all source-target MMD values.

## 7. Reasoning

### 7.1 Why Use Three Crops?

The pressure waveform of a rock drill impact cycle contains different physical stages. Fault-related information may appear in the beginning, middle, or ending part of the cycle. Using only one segment may miss important fault features. Therefore, the three-crop strategy provides a more complete representation while maintaining a fixed input size.

### 7.2 Why Use Domain Adaptation?

Different individuals may have different waveform baselines, pressure amplitudes, signal delays, and operating characteristics. If a model is trained only on labeled source individuals, it may learn individual-specific features instead of fault-specific features. Domain adaptation helps reduce this problem by encouraging the feature extractor to learn domain-invariant and fault-discriminative representations.

### 7.3 Why Use MMD Together with DANN?

DANN aligns the domains adversarially by making it difficult for the domain discriminator to distinguish domains. MMD provides an additional statistical alignment by directly reducing the distance between source and target feature distributions. Combining both losses improves feature-level domain adaptation.

### 7.4 Why Not Use Target Labels During Training?

When the target individual is unlabeled, its labels are unknown. Therefore, target labels are not used in the classification loss. Target samples are used only for:

- domain-adversarial loss;
- MMD feature alignment;
- final prediction.

If the target data contain only dummy labels such as 0, these values should not be used as true class labels.

## 8. Internal Validation Experiment

Before applying the model to fully unlabeled target data, an internal validation experiment was conducted using one individual from the available training data as a validation check. This setting is useful because the training data contain known labels, so the predicted labels can be compared with the true labels to verify whether the implementation can learn meaningful fault-discriminative features.

The loaded source individuals and their sample counts were:

| Individual | Number of samples |
|---:|---:|
| 1 | 7,311 |
| 2 | 7,867 |
| 4 | 7,597 |
| 5 | 7,977 |
| 6 | 3,293 |
| **Total** | **34,045** |

The class distribution was generally balanced across the 11 classes. This is helpful for supervised source training because the model does not strongly favor only a few dominant classes.

### 8.1 Validation Result

The model was trained for 100 epochs. The validation accuracy improved rapidly during the first few epochs:

| Epoch | Total loss | Task loss | Domain loss | MMD loss | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 45.1154 | 13.6512 | 28.9292 | 2.5350 | 0.0911 |
| 2 | 33.8891 | 2.4402 | 30.6894 | 0.7595 | 0.8958 |
| 3 | 33.0874 | 0.9104 | 31.6657 | 0.5114 | 1.0000 |
| 100 | 31.2704 | 0.0018 | 31.2226 | 0.0459 | 1.0000 |

The result shows that the model quickly learned the source fault classification task. The validation accuracy increased from 9.11% at epoch 1 to 89.58% at epoch 2 and reached 100% at epoch 3. The final epoch maintained 100% accuracy.

The task loss also decreased from 13.6512 to 0.0018, which indicates that the classifier learned highly discriminative features for the labeled training-domain data. The MMD loss decreased from 2.5350 to 0.0459, showing that the feature alignment term became more stable during training.

### 8.2 Interpretation of the Validation Result

The strong internal validation performance confirms that the preprocessing pipeline, three-crop input representation, CNN classifier, domain discriminator, and MMD implementation are functioning correctly. It also shows that the model can learn the fault labels from the available training individuals.

However, if the validation individual is also included in the source-domain training set, the result should be interpreted as an implementation sanity check rather than a strict leave-one-individual-out validation result. For a stricter generalization test, the validation individual should be removed from the source list and used only for evaluation. For example:

```python
source_domain_ls = [1, 2, 4, 5]
validation_individual = 6
```

This strict setting better evaluates whether the model can generalize to a new individual.

## 9. Unlabeled Target Prediction

After confirming the implementation with internal validation, the model can be trained using labeled source individuals and an unlabeled target individual. In this case, the target label column is treated as dummy information only.

For unlabeled target data, accuracy cannot be calculated because the ground-truth labels are not available. The model should instead output predicted labels:

```python
prob = torch.softmax(class_logits, dim=1)
pred_label = prob.argmax(dim=1) + 1
```

The final output should be saved as a CSV file:

| Column | Description |
|---|---|
| `sample_index` | Index of the target impact cycle |
| `pred_label` | Predicted class label from 1 to 11 |

If probability outputs are needed, the softmax probabilities for all 11 classes can also be saved.

## 10. Model Saving

The trained model weights were saved using:

```python
torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pth"))
```

This saves only the model parameters. When loading the model weights later, the same architecture settings must be used, including:

- `crop_len = 300`
- `num_classes = 11`
- `num_domains = number of source domains + number of target domains`

For example, if five source domains and one target domain are used:

```python
num_domains = len(source_domain_ls) + 1
```

## 11. Conclusion

This project implemented a domain-adaptation-based fault diagnosis model for the 2022 PHM Conference Data Challenge. The implementation was based on the second-place paper and reproduced its core concepts: three-crop preprocessing, CNN-based feature extraction, domain-adversarial learning, MMD feature alignment, and source-target domain adaptation.

The internal validation experiment showed strong performance. The validation accuracy reached 89.58% at epoch 2 and 100% from epoch 3 onward. This result indicates that the implemented model can effectively learn fault-discriminative patterns from the available training individuals.

For fully unlabeled target data, the model should not calculate target accuracy. Instead, the target samples should be used for domain alignment during training, and the final output should be the predicted fault labels for each target impact cycle.

## References

1. Kim, Y. C., Kim, T., Ko, J. U., Lee, J., & Kim, K. (2023). *Domain Adaptation based Fault Diagnosis under Variable Operating Conditions of a Rock Drill*. International Journal of Prognostics and Health Management. https://papers.phmsociety.org/index.php/ijphm/article/view/3425
2. 2022 PHM Conference Data Challenge Rock Drill Dataset. PHM Society Data Challenge. https://data.phmsociety.org/2022-phm-conference-data-challenge/
