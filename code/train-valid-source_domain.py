import os, csv, random, torch, math
import pandas as pd
import numpy as np
from itertools import cycle
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from models import DANN
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
seed= 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def read_csv(path: str):
    label_ls= []; sequence_ls= []
    with open(path, mode= "r", newline= "", encoding= "utf-8-sig") as file:
        data_reader= csv.reader(file)
        for line_i, row in enumerate(data_reader, start= 1):
            row= [value.strip() for value in row if value.strip() != ""]
            if not row: continue
            label= int(float(row[0]))
            sequence= np.asarray(row[1 :], dtype= np.float32)
            label_ls.append(label); sequence_ls.append(sequence)
    return label_ls, sequence_ls

def load_individual(data_dir: str, individual: str):
    sensor_labels= {}; sensor_sequences= {}
    for sensor in ["pin", "pdmp", "po"]:
        labels, sequences= read_csv(os.path.join(data_dir, "data_%s%s.csv"%(sensor, individual)))
        sensor_labels[sensor], sensor_sequences [sensor]= labels, sequences
    n= len(sensor_labels["pin"])
    for sensor in ["pdmp", "po"]:
        if len(sensor_labels[sensor]) != n:
            raise ValueError("Row-count mismatch for individual %s: pin has %2d, %s has %d"%(individual, n, sensor, len(sensor_labels[sensor])))
    return sensor_labels, sensor_sequences

def normalization(sequence: np.array):
    return ((sequence - sequence.mean()) / (sequence.std())).astype(np.float32)

### All of the concepts are from the paper got the second place.
def pad_or_crop_process(x: np.array, crop_len: int, mode= "edge"):
    if len(x)== crop_len: return x.astype(np.float32)
    if len(x)> crop_len: return x[: crop_len].astype(np.float32)
    pad_width= crop_len - len(x)
    if len(x) == 0: return np.zeros(crop_len, dtype= np.float32)
    if mode== "zero": return np.pad(x, (0, pad_width), mode= "constant").astype(np.float32)
    return np.pad(x, (0, pad_width), mode="edge").astype(np.float32)

def crop_first_center_last(sensor_sequences: np.array, crop_len= 300):
    n_observations= sensor_sequences.shape[1]
    croped_sensor_sequence= []
    for s in range(3):
        croped_sensor_sequence.append(pad_or_crop_process(sensor_sequences[s, :], crop_len))
    croped_sensor_sequence= np.stack(croped_sensor_sequence, axis= 0)
    if n_observations>= crop_len:
        center_s= max((n_observations- crop_len)// 2, 0)
        center= sensor_sequences[: , center_s: center_s+ crop_len]
        croped_center_sensor_sequence= []
        for s in range(3):
            croped_center_sensor_sequence.append(pad_or_crop_process(center[s, :], crop_len))
        croped_center_sensor_sequence= np.stack(croped_center_sensor_sequence, axis= 0)
        last_sensor_sequence= sensor_sequences[: , - crop_len:]
        croped_last_sensor_sequence= []
        for s in range(3):
            croped_last_sensor_sequence.append(pad_or_crop_process(last_sensor_sequence[s, :], crop_len))
        croped_last_sensor_sequence= np.stack(croped_last_sensor_sequence, axis= 0)
    else:
        croped_center_sensor_sequence= []
        for s in range(3):
            croped_center_sensor_sequence.append(pad_or_crop_process(center[s, :], crop_len))
        croped_center_sensor_sequence= np.stack(croped_center_sensor_sequence, axis= 0)
        croped_last_sensor_sequence= center.copy()
    return np.stack([croped_sensor_sequence,croped_center_sensor_sequence , croped_last_sensor_sequence], axis=0).astype(np.float32)

def preprocessing(sensor_labels, sensor_sequences, normalization_process= True, crop_len= 300):
    x_ls= []; n= len(sensor_labels["pin"])
    for i in range(n):
        ### each label's sensors data
        rows= [sensor_sequences[sensor][i] for sensor in ["pin", "pdmp", "po"]]
        min_len= min(len(r) for r in rows)
        if min_len <= 0: continue
        processed= []
        for row_data in rows:
            sub_sequence= row_data[: min_len].astype(np.float32)
            if normalization_process== True:
                sub_sequence= normalization(sub_sequence)
            processed.append(sub_sequence)
        sensor_sequence= np.stack(processed, axis= 0)
        x_ls.append(crop_first_center_last(sensor_sequence, crop_len= crop_len))
    x= np.stack(x_ls, axis= 0).astype(np.float32)
    y= np.array(sensor_labels["pin"]).astype(np.int64).copy()
    if len(x) != len(y):
        y= y[: len(x)]
    # if not os.path.exists(os.path.join(os.getcwd(), "preprocessed_sensor_sequences")):
        # os.mkdir(os.path.join(os.getcwd(), "preprocessed_sensor_sequences"))
    return x, y.astype(np.int64)

class TrainDataLoader(Dataset):
    def __init__(self, x: np.array, y: np.array, domain_id: int):
        self.x= torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64).astype(np.int64))
        self.domain= torch.full((len(self.x), ), int(domain_id), dtype= torch.long)
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.domain[idx]

def dann_lambda(progress: float, gamma= 10.0):
    progress = min(max(progress, 0.0), 1.0)
    return float(2.0 / (1.0+ math.exp(-gamma * progress)) - 1.0)

def gaussian_kernel_matrix(x, y, kernel_mul= 2.0, num_kernels= 5, fix_sigma= None):
    total= torch.cat([x, y], dim=0)
    dist= torch.cdist(total, total, p=2).pow(2)
    if fix_sigma is None:
        bandwidth= dist.detach().sum() / max(total.shape[0] ** 2 - total.shape[0], 1)
        bandwidth= torch.clamp(bandwidth, min=1e-6)
    else:
        bandwidth= torch.tensor(float(fix_sigma), device=x.device, dtype=x.dtype)
    bandwidth = bandwidth / (kernel_mul ** (num_kernels// 2))
    bandwidths = [bandwidth * (kernel_mul ** i) for i in range(num_kernels)]
    kernels = [torch.exp(-dist / bw) for bw in bandwidths]
    return sum(kernels)

def mmd_loss(source_feat: torch.Tensor, target_feat: torch.Tensor, kernel_mul= 2.0, num_kernels= 5, fix_sigma= None):
    kernels = gaussian_kernel_matrix(source_feat, target_feat, kernel_mul, num_kernels, fix_sigma)
    n_s= source_feat.shape[0]
    n_t= target_feat.shape[0]
    k_xx = kernels[:n_s, :n_s]
    k_yy = kernels[n_s:, n_s:]
    k_xy = kernels[:n_s, n_s:]
    return k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()

def dann_lambda(progress: float, gamma= 10.0):
    progress= min(max(progress, 0.0), 1.0)
    return float(2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)

if __name__== "__main__":
    device= torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_domain_ls= [1, 2, 4, 5, 6]
    ### source domain datasets
    source_domain_dataloaders= []
    for domain_idx, individual in enumerate(source_domain_ls):
        sensor_labels, sensor_sequences= load_individual(data_dir= os.path.join(os.getcwd(), "../", "Data_Challenge_PHM2022_training_data"), individual= str(individual))
        train_x, train_y= preprocessing(sensor_labels= sensor_labels, sensor_sequences= sensor_sequences)
        train_dataloader= TrainDataLoader(train_x, train_y, domain_id= domain_idx)
        source_domain_dataloaders.append(train_dataloader)
    for source_domain_idx, source_domain_dataloader in zip(source_domain_ls, source_domain_dataloaders):
        labels, counts= np.unique(source_domain_dataloader.y, return_counts= True)
        label_info= dict(zip(labels.astype(int).tolist(), counts.astype(int).tolist()))
        ### TODO - visualization
        print(f"Source individual {source_domain_idx}: N={len(source_domain_dataloader)}, label_counts={label_info}")
    loader_kwargs = dict(batch_size= 128* 3, drop_last= True)
    source_domain_dataloaders= [DataLoader(ds, shuffle=True, **loader_kwargs) for ds in source_domain_dataloaders]
    source_domain_loader_iteration= [cycle(loader) for loader in source_domain_dataloaders]
    ### TRAINING
    model= DANN(crop_len= 300, num_classes= 11, num_domains= len(source_domain_ls)).to(device= device)
    optimizer= torch.optim.Adam(model.parameters(), lr= 0.001, weight_decay= 0.0)
    model_update_freqeuncy= max(len(loader) for loader in source_domain_dataloaders)
    epoch= 100; update_time= 0; total_update_times= max(1, epoch* model_update_freqeuncy)
    
    for e in range(1, epoch+ 1):
        model.train()
        train_batch_loss= {"loss": 0.0, "task": 0.0, "domain": 0.0, "mmd": 0.0}
        for step in range(model_update_freqeuncy):
            progress = update_time/ total_update_times
            grl_lambd= dann_lambda(progress, gamma= 10.0)
            x_sources, y_sources, i_sources= [], [], []
            for src_iter in source_domain_loader_iteration:
                xs, ys, ds= next(src_iter)
                x_sources.append(xs.to(device, non_blocking=True))
                y_sources.append(ys.to(device, non_blocking=True))
                i_sources.append(ds.to(device, non_blocking=True))
            xt, yt, dt = next(source_domain_loader_iteration[-1])
            xt, dt= xt.to(device, non_blocking= True), dt.to(device, non_blocking= True)
            x_all, i_all= torch.cat(x_sources+ [xt], dim=0), torch.cat(i_sources+ [dt], dim=0)
            class_logits_all, domain_logits_all, feat_all= model(x_all, grl_lambda= grl_lambd)
            n_source_total = sum(x.shape[0] for x in x_sources)
            class_logits_src = class_logits_all[: n_source_total]
            y_src= torch.cat(y_sources, dim=0)- 1
            task_loss= nn.functional.cross_entropy(class_logits_src, y_src)
            domain_loss= nn.functional.cross_entropy(domain_logits_all, i_all)
            mmd_terms= []; cursor = 0
            feat_target= feat_all[n_source_total: ]
            for xs in x_sources:
                n_i = xs.shape[0]
                feat_i = feat_all[cursor:cursor + n_i]
                mmd_terms.append(mmd_loss(feat_i, feat_target, kernel_mul= 2.0, num_kernels= 5, fix_sigma= None))
                cursor+= n_i
            multi_source_mmd = torch.stack(mmd_terms).mean()
            loss= task_loss + 1.0* domain_loss + 1.0* multi_source_mmd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_batch_loss["loss"]+= float(loss.detach().cpu())
            train_batch_loss["task"]+= float(task_loss.detach().cpu())
            train_batch_loss["domain"]+= float(domain_loss.detach().cpu())
            train_batch_loss["mmd"]+= float(multi_source_mmd.detach().cpu())
            update_time+= 1
        model.eval()
        with torch.no_grad():
            xt, yt, dt = next(source_domain_loader_iteration[-1])
            xt, dt= xt.to(device, non_blocking= True), dt.to(device, non_blocking= True)
            class_logits, _, _= model(xt, grl_lambda= grl_lambd)
            pred_label= torch.nn.functional.log_softmax(class_logits, -1).argmax(dim= 1).detach().cpu()+ 1
            acc = (pred_label== yt).float().mean()
        print(f"Epoch {e:03d}/{100} | " f"Loss={train_batch_loss['loss']:02.4f} Task-loss={train_batch_loss['task']:02.4f} " f"domain-loss={train_batch_loss['domain']:02.4f} mmd-loss={train_batch_loss['mmd']:02.4f} " f"acc= {acc:.4f}")