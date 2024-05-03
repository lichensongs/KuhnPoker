from InfoSet import Card, InfoSet, Value, Action

from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.init as init

DEBUG = False
Policy = np.ndarray
CardDistribution = np.ndarray

class TorchModelH(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_layer = 16

        self.fc1 = nn.Linear(1, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 1)

        init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')

        self.relu = nn.ReLU()


        # self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = self.fc1(x)  # (N, 16)
        x = self.relu(x)  # (N, 16)
        x = self.fc2(x)
        p = F.sigmoid(x)
       
        return p


class ConstModel:
    def __init__(self, p, q, h=None, eps=0):
        self.p = p
        self.q = q
        self.h = h
        self.eps = eps

        self._P_tensor = np.zeros((2, 3, 2))
        self._V_tensor = np.zeros((2, 3, 2))

        J = Card.JACK.value
        Q = Card.QUEEN.value
        K = Card.KING.value

        self._P_tensor[:, K, 1] = 1  # always add chip with King
        self._P_tensor[0, Q, 0] = 1  # never bet with a Queen
        self._P_tensor[1, J, 0] = 1  # never call with a Jack

        self._P_tensor[0, J] = np.array([1-p, p])  # bluff with a Jack with prob p
        self._P_tensor[1, Q] = np.array([1-q, q])  # call with a Queen with prob q

        self._V_tensor[0, J, 1] = -1 + p*(1-3*q)/2
        self._V_tensor[0, Q, 1] = 0
        self._V_tensor[0, K, 1] = 1 - q/2
        self._V_tensor[0, :, 0] = -self._V_tensor[0, :, 1]

        self._V_tensor[1, J, 0] = -1
        self._V_tensor[1, Q, 0] = -1 + q*(3*p-1)/(1+p)
        self._V_tensor[1, K, 0] = +2
        self._V_tensor[1, :, 1] = -self._V_tensor[1, :, 0]

    def __call__(self, info_set: InfoSet) -> Tuple[Policy, Value]:
        if len(info_set.action_history) == 0:
            return (np.array([1.0, 0]), self._V_tensor[0].sum(axis=0) / 3)

        cp = info_set.get_current_player()
        card = info_set.cards[cp.value]
        assert card is not None

        x = info_set.action_history[-1].value
        y = card.value

        P = self._P_tensor[x][y]
        if info_set.action_history == [Action.PASS, Action.ADD_CHIP] and info_set.cards == [Card.QUEEN, Card.JACK]:
            V = np.zeros(2)
        else:
            V = self._V_tensor[x][y]
            
        if DEBUG:
            print(f'  model({info_set}) -> P={P}, V={V}')
        return (P, V)

    def bayes_prob(self, info_set: InfoSet) -> CardDistribution:
        return self._bayes_prob(info_set, self.h)

    def _bayes_prob(self, info_set: InfoSet, h: int) -> CardDistribution:
        cp = info_set.get_current_player().value
        if h is not None:
            if (info_set.action_history[-1] == Action.ADD_CHIP) and (info_set.cards[1-cp] == Card.QUEEN):
                H = np.array([1 - h, 0, h])
                return (H, self.eps)
        
        H = np.ones(3)
        for k in range(len(info_set.action_history) - 1):
            if k % 2 == cp:
                continue
            x = info_set.action_history[k].value
            y = info_set.action_history[k+1].value
            H *= self._P_tensor[x, :, y]

        card = info_set.cards[1 - cp]
        assert card is not None

        H[card.value] = 0
        return (H / sum(H), 0.0)

class ModelH(ConstModel):
    def __init__(self, p, q, h_model: TorchModelH = None, eps=0):
        super().__init__(p, q, eps=eps)
        self.h_model = h_model
        self.h_model.eval()
    
    def bayes_prob(self, info_set: InfoSet):
        x = torch.tensor([self.p], dtype=torch.float32)
        h  = self.h_model(x).cpu().detach().numpy()[0, 0]
        return self._bayes_prob(info_set, h)
    
class Model:
    def __init__(self, p, q):
        self.p = p
        self.q = q

        self._P_tensor = np.zeros((2, 3, 2))
        self._V_tensor = np.zeros((2, 3, 2))

        J = Card.JACK.value
        Q = Card.QUEEN.value
        K = Card.KING.value

        self._P_tensor[:, K, 1] = 1  # always add chip with King
        self._P_tensor[0, Q, 0] = 1  # never bet with a Queen
        self._P_tensor[1, J, 0] = 1  # never call with a Jack

        self._P_tensor[0, J] = np.array([1-p, p])  # bluff with a Jack with prob p
        self._P_tensor[1, Q] = np.array([1-q, q])  # call with a Queen with prob q

        self._V_tensor[0, J, 1] = -1 + p*(1-3*q)/2
        self._V_tensor[0, Q, 1] = 0
        self._V_tensor[0, K, 1] = 1 - q/2
        self._V_tensor[0, :, 0] = -self._V_tensor[0, :, 1]

        self._V_tensor[1, J, 0] = -1
        self._V_tensor[1, Q, 0] = -1 + q*(3*p-1)/(1+p)
        self._V_tensor[1, K, 0] = +2
        self._V_tensor[1, :, 1] = -self._V_tensor[1, :, 0]

    def __call__(self, info_set: InfoSet) -> Tuple[Policy, Value]:
        if len(info_set.action_history) == 0:
            return (np.array([1.0, 0]), self._V_tensor[0].sum(axis=0) / 3)

        cp = info_set.get_current_player()
        card = info_set.cards[cp.value]
        assert card is not None

        x = info_set.action_history[-1].value
        y = card.value

        P = self._P_tensor[x][y]
        V = self._V_tensor[x][y]
        if DEBUG:
            print(f'  model({info_set}) -> P={P}, V={V}')
        return (P, V)

    def bayes_prob(self, info_set: InfoSet) -> CardDistribution:
        cp = info_set.get_current_player().value
        H = np.ones(3)
        for k in range(len(info_set.action_history) - 1):
            if k % 2 == cp:
                continue
            x = info_set.action_history[k].value
            y = info_set.action_history[k+1].value
            H *= self._P_tensor[x, :, y]

        card = info_set.cards[1 - cp]
        assert card is not None

        H[card.value] = 0
        return (H / sum(H), 0.0)
    

if __name__ == '__main__':

    info_set = InfoSet([Action.PASS, Action.ADD_CHIP, Action.ADD_CHIP])
    info_set.cards = [Card.QUEEN, None]
    h_model = torch.load('temp/model-127.pt')
    model = ModelH(0.99, 0.99, h_model=h_model, eps=0.10)
    print(model.bayes_prob(info_set))