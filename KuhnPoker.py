from MCTS import ISMCTS, InfoSet, Action, Card, Model, ActionNode, ConstModel

# third-party (like torch)
import numpy as np
import random
from tqdm import tqdm
import pandas as pd

# python 
from dataclasses import dataclass
from typing import List, Tuple


Policy = List[float]
Value = float

train_filepath = 'training_data/'


class KuhnPoker:
    def __init__(self, model, n_games_per_gen=64, n_mcts_iters=1000):
        self.n_games_per_gen = n_games_per_gen
        self.n_mcts_iters = n_mcts_iters

        self.model = model
        self.opt = None  

    def run(self, n_gen=20):
        p_list = [self.model.p]
        q_list = [self.model.q]
        h_list = [self.model.h]
        
        for g in range(n_gen):
            self.run_generation(g)
            # p, q, h = self.update_model(g)
            p, q, h = self.update_h(g)
            p_list.append(p)
            q_list.append(q)
            h_list.append(h)

        df = pd.DataFrame({'p': p_list,
                           'q': q_list,
                           'h': h_list})
        df.to_csv(f'{train_filepath}df-stats.csv', index=False)

    def update_model(self, gen, gamma=0.80):
        df = pd.read_csv(f'{train_filepath}df-{gen}.csv')
        p_old = self.model.p
        q_old = self.model.q
        h_old = self.model.h
        p = df[(df['facing_action'] == 0) & (df['card'] == 0)]['prob'].mean()
        q = df[(df['facing_action'] == 1) & (df['card'] == 1)]['prob'].mean()
        
        update_p = gamma * p_old + (1 - gamma) * p
        update_q = gamma * q_old + (1 - gamma) * q
        
        ix = (df['facing_action'] == 1) & (df['card'] == 1)
        count = df[ix]['opponent_card'].count()
        h = (df[ix]['opponent_card'].value_counts() / count).loc[2]
        
        print(p, q, h)
        self.model = ConstModel(update_p, update_q)
        return (p, q, h)
    
    def update_h(self, gen, gamma=0.90):
        df = pd.read_csv(f'{train_filepath}df-{gen}.csv')
        p = self.model.p
        q = self.model.q
        h = self.model.h
        eps = self.model.eps
        
        p_freq = df[(df['facing_action'] == 0) & (df['card'] == 0)]['prob'].mean()
        q_freq = df[(df['facing_action'] == 1) & (df['card'] == 1)]['prob'].mean()
        
        ix = (df['facing_action'] == 1) & (df['card'] == 1)
        count = df[ix]['opponent_card'].count()
        h_freq = (df[ix]['opponent_card'].value_counts() / count).loc[2]

        # if h_freq > h - eps and h_freq < h + eps:
        #     std = np.sqrt(h * (1 - h) / count)
        #     update_p = gamma * p + (1 - gamma) * p_freq
        #     update_q = gamma * q + (1 - gamma) * q_freq
        #     self.model = ConstModel(update_p, update_q, h = h, eps = std)
        #     print(f'model: {p:.3f}, {q:.3f}, {h:.3f}, {eps:.3f}, \nfreq: {p_freq:.3f}, {q_freq:.3f}, {h_freq:.4f}, \nupdate: {update_p:.3f}, {update_q:.3f}, {h:.3f}, {std:.3f}')
        #     return (p_freq, q_freq, h)
        
        update_p = gamma * p + (1 - gamma) * p_freq
        update_q = gamma * q + (1 - gamma) * q_freq

        update_h = gamma * h + (1 - gamma) * h_freq
        # if h_freq > h + eps or h_freq < h - eps:
        #     update_h = gamma * h + (1 - gamma) * h_freq
        # else:
        #     update_h = h

        std = np.sqrt(update_h * (1 - update_h) / count)
        self.model = ConstModel(update_p, update_q, h=update_h, eps=std)
        print(f'model: {p:.3f}, {q:.3f}, {h:.3f}, {eps:.3f}, \nfreq: {p_freq:.3f}, {q_freq:.3f}, {h_freq:.4f}, \nupdate: {p:.3f}, {q:.3f}, {update_h:.3f}, {std:.3f}')
        return (p_freq, q_freq, h)
    
    def run_generation(self, gen):
        model = self.model
        num_games = self.n_games_per_gen
        num_iters = self.n_mcts_iters

        print(f'Running gen-{gen} self-play')
        
        df_list = []

        # for i in tqdm(range(num_games)):
        for i in range(num_games):
            info_set = InfoSet([Action.PASS])
            deck  = list(Card)
            random.shuffle(deck)
            cards = deck[:2]

            data: List[Tuple[InfoSet, Policy]] = []
            while True:
                cp = info_set.get_current_player()
                info_set = info_set.clone()
                info_set.cards = [None, None]
                info_set.cards[cp.value] = cards[cp.value]
                node = ActionNode(info_set)

                mcts = ISMCTS(model, node)
                dist = mcts.get_visit_distribution(num_iters)
                data.append(((info_set.action_history[-1].value, cards[cp.value].value, cards[1 - cp.value].value), dist[Action.ADD_CHIP]))

                actions, probs = zip(*list(dist.items()))
                move = np.random.choice(actions, p=probs)
                
                info_set = info_set.apply_move(move)
                info_set.cards = cards
                outcome = info_set.get_game_outcome()

                if outcome is None:
                    continue
                break
            
            facing_action = [info[0] for info, p in data]
            holding_card = [info[1] for info, p in data]
            opp_card = [info[2] for info, p in data]
            probs = [p for info, p in data]

            df = pd.DataFrame({'facing_action': facing_action,
                               'card': holding_card,
                               'opponent_card': opp_card,
                               'prob': probs})

            df['game#'] = i
            df_list.append(df)

        df = pd.concat(df_list, axis=0, ignore_index=True) 
        df.to_csv(f'{train_filepath}df-{gen}.csv', index=False)


if __name__ == '__main__':
    model = ConstModel(0.5, 0.5, h=0.75, eps=0.25)
    poker = KuhnPoker(model, n_games_per_gen=64)
    # poker.run_generation(0)
    poker.run(n_gen=100)
