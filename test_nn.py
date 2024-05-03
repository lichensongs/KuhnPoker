from Model import TorchModelH
from KuhnPoker import SelfPlayData

import pandas as pd
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm import tqdm


train_filepath = 'training_data/'
model_path = 'temp/'
# device = torch.device('mps')

def train(h_model: TorchModelH, optim, gen: int, num_batches=256, batch_size=256):
    dfs = []
    for i in range(gen + 1):
        dfs.append(pd.read_csv(f'{train_filepath}df-{i}.csv'))
    df = pd.concat(dfs, axis=0)

    dataset = SelfPlayData(df)
    loader = DataLoader(dataset, batch_size, shuffle=True)

    batch_num = 0
    for data in loader:
        num_batches -= 1
        if num_batches == 0:
            break
        
        batch_num += 1

        optim.zero_grad()
        data_p, target_h = data
        data_p = data_p
        target_h = target_h

        h_hat = h_model(data_p)
        lossF = nn.BCELoss()
        loss = lossF(h_hat, target_h)
        loss.backward()
        optim.step()

        if batch_num % 25 == 0:
            print(f'Gen: {gen:3d} Batch: {batch_num:4d} loss: {loss:.5f}')


    filepath = f'{model_path}model-{gen}.pt'
    torch.save(h_model, filepath)

if __name__ == '__main__':
    dfs = []
    for i in range(0, 128):
        dfs.append(pd.read_csv(f'training_data/df-{i}.csv'))
    df = pd.concat(dfs, axis=0)

    h_model = TorchModelH()
    # h_model = torch.load('temp/model-127.pt')

    learning_rate = 1e-2
    momentum = 0.9
    weight_decay = 1e-6

    opt = optim.SGD(h_model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    for _ in tqdm(range(128)):
        train(h_model, opt, 127, num_batches=256, batch_size=1024)