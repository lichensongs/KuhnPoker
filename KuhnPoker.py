from MCTS import ISMCTS, InfoSet, Action, Card, Model, ActionNode, ConstModel
from Model import ModelH, TorchModelH

import numpy as np
import random
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch import optim
import torch.nn as nn

from dataclasses import dataclass
from typing import List, Tuple

dtype = torch.float32

Policy = List[float]
Value = float

train_filepath = 'training_data/'
model_path='temp/'

class SelfPlayData(Dataset):
    def __init__(self, df):
        ix = (df['facing_action'] == 1) & (df['card'] == 1)
        self.df = df[ix]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        input_p = row['input_p']
        isKing = (row['opponent_card'] == 2).astype(int)
        x = torch.tensor([input_p], dtype=dtype)
        target_h = torch.tensor([isKing], dtype=dtype)
        return (x, target_h)


class KuhnPoker:
    def __init__(self, model, n_games_per_gen=64, n_mcts_iters=100, fixed_eps=False):
        self.n_games_per_gen = n_games_per_gen
        self.n_mcts_iters = n_mcts_iters
        self.fixed_eps = fixed_eps

        self.model = model
        self.opt = None  

    def run(self, gamma=0.9, n_gen=20):
        p_list = [self.model.p]
        q_list = [self.model.q]
        h_list = [self.model.h]
        
        for g in range(n_gen):
            print(f'Running generation {g}')
            self.run_generation(g)
            # p, q, h = self.update_model(g)
            p, q, h = self.update_h(g, gamma=gamma)
            p_list.append(p)
            q_list.append(q)
            h_list.append(h)
        
            if g % 64 == 0:
                df = pd.DataFrame({'p': p_list,
                           'q': q_list,
                           'h': h_list})
                df.to_csv(f'{train_filepath}df-stats.csv', index=False)
                
        df = pd.DataFrame({'p': p_list,
                           'q': q_list,
                           'h': h_list})
        
        df.to_csv(f'{train_filepath}df-stats.csv', index=False)

    def run_gen_train_loop(self, n_gen=64):
        p_list = [self.model.p]
        q_list = [self.model.q]

        for g in range(n_gen):
            print(f'Running generation {g}')
            self.run_generation(g)
            self.train(g)
            self.update_model(g)

            p_list.append(self.model.p)
            q_list.append(self.model.q)

            if g % 64 == 0:
                df = pd.DataFrame({'p': p_list,
                                   'q': q_list})
                df.to_csv(f'{train_filepath}df-stats-{g}.csv', index=False)

    def update_model(self, gen, gamma=0.80):
        df = pd.read_csv(f'{train_filepath}df-{gen}.csv')
        p_old = self.model.p
        q_old = self.model.q

        p = df[(df['facing_action'] == 0) & (df['card'] == 0)]['prob'].mean()
        q = df[(df['facing_action'] == 1) & (df['card'] == 1)]['prob'].mean()
        
        update_p = gamma * p_old + (1 - gamma) * p
        update_q = gamma * q_old + (1 - gamma) * q
              
        self.model = ModelH(update_p, update_q, h_model=self.model.h_model, eps=self.model.eps)
        print(update_p, update_q)
    
    def update_h(self, gen, gamma=0.8, batch=4096):

        # df = pd.read_csv(f'{train_filepath}df-{gen}.csv')

        dfs = []
        for i in range(gen + 1):
            dfs.append(pd.read_csv(f'{train_filepath}df-{i}.csv'))
        df = pd.concat(dfs, axis=0)[-batch:]
                
        p = self.model.p
        q = self.model.q
        h = self.model.h
        eps = self.model.eps
        
        p_freq = df[(df['facing_action'] == 0) & (df['card'] == 0)]['prob'].mean()
        q_freq = df[(df['facing_action'] == 1) & (df['card'] == 1)]['prob'].mean()
       
        ix = (df['facing_action'] == 1) & (df['card'] == 1)
        count = df[ix]['opponent_card'].count()
        h_freq = (df[ix]['opponent_card'].value_counts() / count).loc[2]

        update_p = gamma * p + (1 - gamma) * p_freq
        update_q = gamma * q + (1 - gamma) * q_freq
        update_h = gamma * h + (1 - gamma) * h_freq

        if self.fixed_eps:
            std = eps
        else:
            std = np.sqrt(update_h * (1 - update_h) / count)

        self.model = ConstModel(update_p, update_q, h=update_h, eps=std)
        print(f'model: {p:.3f}, {q:.3f}, {h:.3f}, {eps:.3f}, \nfreq: {p_freq:.3f}, {q_freq:.3f}, {h_freq:.4f}, \nupdate: {update_p:.3f}, {update_q:.3f}, {update_h:.3f}, {std:.3f}')
        return (update_p, update_q, update_h)
    
    def train(self, gen: int, num_batches=256, batch_size=256):
        if gen == 0:
            learning_rate = 1e-2
            momentum = 0.9
            weight_decay = 6e-5
            self.opt = optim.SGD(self.model.h_model.parameters(), 
                                 lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        
        dfs = []
        for i in range(gen + 1):
            dfs.append(pd.read_csv(f'{train_filepath}df-{i}.csv'))
        df = pd.concat(dfs, axis=0)

        h_model = self.model.h_model
        dataset = SelfPlayData(df)
        loader = DataLoader(dataset, batch_size, shuffle=True)

        batch_num = 0
        for data in loader:
            num_batches -= 1
            if num_batches == 0:
                break
            
            batch_num += 1

            self.opt.zero_grad()
            data_p, target_h = data
            h_hat = h_model(data_p)
            lossF = nn.BCELoss()
            loss = lossF(h_hat, target_h)
            loss.backward()
            self.opt.step()

            if batch_num % 100 == 0:
                print(f'Gen: {gen:3d} Batch: {batch_num:4d} loss: {loss:.5f}')


        filepath = f'{model_path}model-{gen}.pt'
        torch.save(h_model, filepath)

    def run_generation(self, gen):
        model = self.model
        num_games = self.n_games_per_gen
        num_iters = self.n_mcts_iters

        # print(f'Running gen-{gen} self-play')
        
        df_list = []

        # for i in tqdm(range(num_games)):
        for i in range(num_games):
            info_set = InfoSet([Action.PASS])
            deck  = list(Card)
            random.shuffle(deck)
            cards = deck[:2]

            # if cards[1] == Card.JACK:
            #     print('caught')

            data: List[Tuple[InfoSet, Policy]] = []
            while True:

                cp = info_set.get_current_player()
                info_set = info_set.clone()
                info_set.cards = [None, None]
                info_set.cards[cp.value] = cards[cp.value]
                node = ActionNode(info_set, model)

                mcts = ISMCTS(model, node)
                dist = mcts.get_visit_distribution(num_iters)
                data.append(((info_set.action_history[-1].value, cards[cp.value].value, cards[1 - cp.value].value, cp.value), dist[Action.ADD_CHIP]))

                actions, probs = zip(*list(dist.items()))
                move = np.random.choice(actions, p=probs)
                
                info_set = info_set.apply_move(move)
                info_set.cards = cards
                outcome = info_set.get_game_outcome()

                if outcome is None:
                    continue
                break
            
            facing_action = [info[0] for info, p in data] # facing action
            holding_card = [info[1] for info, p in data] # own card
            opp_card = [info[2] for info, p in data] # opponent card
            probs = [p for info, p in data] # action probability
            current_player = [info[3] for info, p in data]

            df = pd.DataFrame({'facing_action': facing_action,
                               'card': holding_card,
                               'opponent_card': opp_card,
                               'current_player': current_player,
                               'prob': probs})

            df['game#'] = i
            df['outcome'] = df['current_player'].apply(lambda x: outcome[x])
            df['input_p'] = self.model.p
            df['input_q'] = self.model.q
            df['input_h'] = self.model.h
            df_list.append(df)

        df = pd.concat(df_list, axis=0, ignore_index=True)
        df['gen'] = gen
        df.to_csv(f'{train_filepath}df-{gen}.csv', index=False)

def test_const_model():
    model = ConstModel(0.99, 0.99, h=0.75, eps=0.05)
    poker = KuhnPoker(model, n_games_per_gen=64, n_mcts_iters=100, fixed_eps=True)
    poker.run(n_gen=1025, gamma=0.0)

def test_h_model():
    model = ModelH(0.99, 0.99, TorchModelH(), eps=0.05)
    poker = KuhnPoker(model, n_games_per_gen=64, n_mcts_iters=100, fixed_eps=True)
    poker.run_gen_train_loop(n_gen=1025)

if __name__ == '__main__':
    test_h_model()
