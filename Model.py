from InfoSet import Card, InfoSet, Value, Action

from typing import Tuple
import numpy as np

DEBUG = False
Policy = np.ndarray
CardDistribution = np.ndarray

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

        # self._V_tensor[0, J, 1] = -1 + p*(1-3*q)/2
        # self._V_tensor[0, Q, 1] = 0
        # self._V_tensor[0, K, 1] = 1 - q/2
        # self._V_tensor[0, :, 0] = -self._V_tensor[0, :, 1]

        # self._V_tensor[1, J, 0] = -1
        # self._V_tensor[1, Q, 0] = -1 + q*(3*p-1)/(1+p)
        # self._V_tensor[1, K, 0] = +2
        # self._V_tensor[1, :, 1] = -self._V_tensor[1, :, 0]

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
        if self.h is not None:
            if (info_set.action_history[-1] == Action.ADD_CHIP) and (info_set.cards[1-cp] == Card.QUEEN):
                H = np.array([1 - self.h, 0, self.h])
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
        return H / sum(H)