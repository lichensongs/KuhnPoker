import numpy as np

from enum import Enum
from typing import Dict, List, Optional, Tuple


np.set_printoptions(suppress=True)  # avoid scientific notation

EPSILON = 1e-6
c_FPU = 0.2
c_PUCT = 1.0


class Action(Enum):
    PASS = 0
    ADD_CHIP = 1


class Player(Enum):
    ALICE = 0
    BOB = 1


class Card(Enum):
    JACK = 0
    QUEEN = 1
    KING = 2

Policy = np.ndarray
Value = np.ndarray
CardDistribution = np.ndarray

class InfoSet:
    def __init__(self, action_history: List[Action], cards=None):
        self.action_history = action_history
        if cards is None:
            self.cards: List[Optional[Card]] = [None, None]  # indexed by player
        else:
            self.cards = cards

    def __str__(self):
        hist_str = ''.join(map(str, [a.value for a in self.action_history]))
        card_str = ''.join(['?' if c is None else c.name[0] for c in self.cards])
        return f'history=[{hist_str}], cards={card_str}'

    def __repr__(self):
        return str(self)

    def clone(self) -> 'InfoSet':
        return InfoSet(list(self.action_history), list(self.cards))

    def get_current_player(self) -> Player:
        return Player(len(self.action_history) % 2)

    def get_game_outcome(self) -> Optional[Value]:
        """
        None if game not terminal.
        """
        if None in self.cards:
            return None

        if len(self.action_history) < 2:
            return None

        if tuple(self.action_history[-2:]) == (Action.PASS, Action.ADD_CHIP):
            return None

        if tuple(self.action_history[-2:]) == (Action.ADD_CHIP, Action.PASS):
            winner = self.get_current_player()
        elif self.cards[Player.ALICE.value].value > self.cards[Player.BOB.value].value:
            winner = Player.ALICE
        else:
            winner = Player.BOB

        loser = Player(1 - winner.value)
        loser_pot_contribution = self._get_pot_contribution(loser)

        outcome = np.zeros(2)
        outcome[winner.value] = loser_pot_contribution
        outcome[loser.value] = -loser_pot_contribution
        return outcome

    def get_valid_actions(self) -> List[Action]:
        if len(self.action_history) == 0:
            # Force check for now to simplify game
            return [Action.PASS]
        return [Action.PASS, Action.ADD_CHIP]

    def _get_pot_contribution(self, player: Player):
        action_values = [a.value for i, a in enumerate(self.action_history) if i % 2 == player.value]
        return 1 + sum(action_values)

    def apply_move(self, action: Action) -> 'InfoSet':
        action_history = self.action_history + [action]
        cards = list(self.cards)
        return InfoSet(action_history, cards)


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


class BaseNode:
    def __init__(self, info_set: InfoSet):
        self.info_set = info_set

        self.game_outcome = info_set.get_game_outcome()
        self.current_player = info_set.get_current_player()
        self.valid_actions = info_set.get_valid_actions()
        self.valid_action_mask = np.zeros(2)
        for a in self.valid_actions:
            self.valid_action_mask[a.value] = 1

        self.P = None
        self.V = None
        self.Q = np.zeros(2)
        self.N = 0

        if self.is_terminal():
            self.Q = self.game_outcome.copy()

    def is_terminal(self) -> bool:
        return self.game_outcome is not None

    def getQ(self, cp: int, default=None):
        return default if self.Q is None else self.Q[cp]


class ActionNode(BaseNode):
    def __init__(self, info_set: InfoSet):
        super().__init__(info_set)
        self.children_by_action: Dict[Action, BaseNode] = {}
        self.spawned_tree: Optional[ISMCTS] = None

    def __str__(self):
        return f'Action({self.info_set}, N={self.N}, Q={self.Q})'

    def expand_leaf(self, model: Model):
        self.N = 1
        assert self.game_outcome is None
        self.expand_children()
        self.eval_model_and_normalize(model)
        self.Q = self.V

    def expand_children(self):
        assert len(self.children_by_action) == 0

        for action in self.valid_actions:
            info_set = self.info_set.apply_move(action)
            child = HiddenStateSamplingNode(info_set)
            assert action not in self.children_by_action
            self.children_by_action[action] = child

    def eval_model_and_normalize(self, model: Model):
        assert self.P is None
        self.P, self.V = model(self.info_set)
        self.P *= self.valid_action_mask

        s = np.sum(self.P)
        if s < EPSILON:
            # just act uniformly random
            self.P = self.valid_action_mask / np.sum(self.valid_action_mask)
        else:
            self.P /= s


class HiddenStateSamplingNode(BaseNode):
    def __init__(self, info_set: InfoSet):
        super().__init__(info_set)
        self.children_by_card: Dict[Card, ActionNode] = {}

    def __str__(self):
        return f'Hidden({self.info_set}, N={self.N}, Q={self.Q})'

    def sample(self, model: Model) -> ActionNode:
        card_distr = model.bayes_prob(self.info_set)
        card = Card(np.random.choice(3, p=card_distr))
        node = self.children_by_card.get(card, None)
        if node is None:
            info_set = self.info_set.clone()
            cp = info_set.get_current_player().value
            info_set.cards[cp] = card
            assert None not in info_set.cards
            node = ActionNode(info_set)
            self.children_by_card[card] = node

            if not node.is_terminal():
                info_set2 = info_set.clone()
                info_set2.cards[1-cp] = None
                spawned_node = ActionNode(info_set2)
                node.spawned_tree = ISMCTS(model, spawned_node)
        return node


class ISMCTS:
    def __init__(self, model: Model, root: BaseNode):
        self.model = model
        self.root = root

    def get_visit_distribution(self, n: int) -> Dict[Action, float]:
        while self.root.N <= n:
            self.visit(self.root)
            if self.root.N == 1:
                continue

        n_total = self.root.N - 1
        return {action: node.N / n_total for action, node in self.root.children_by_action.items()}

    def choose_best_child(self, node: ActionNode, chosen_action: Optional[List[Action]]=None):
        actions = node.valid_actions

        P = np.array([node.P[a.value] for a in actions])
        N = np.array([node.children_by_action[a].N for a in actions])

        P_explored = np.sum(P * (N > 0))

        cp = node.info_set.get_current_player().value
        children = [node.children_by_action[a] for a in actions]
        Q = np.array([c.getQ(cp, default=0) for c in children])
        Q_FPU = node.Q[cp] - c_FPU * np.sqrt(P_explored)
        Q = Q * (N > 0) + Q_FPU * (N < 1)

        PUCT = Q + c_PUCT * P * np.sqrt(np.sum(N)) / np.maximum(0.5, N)
        best_index = np.argmax(PUCT)

        print('  PUCT calc:')

        act = np.array([a.value for a in actions])
        best_arr = np.zeros(len(P))
        best_arr[best_index] = 1
        full_arr = np.vstack([act, P, N, Q, PUCT, best_arr])
        lines = str(full_arr).splitlines()
        descrs = ['act', 'P', 'N', 'Q', 'PUCT', 'best']
        for descr, line in zip(descrs, lines):
            print('  %6s %s' % (descr, line))

        best_action = actions[best_index]
        if chosen_action is not None:
            chosen_action.append(best_action)
        return node.children_by_action[best_action]

    def visit(self, node: BaseNode, chosen_action: Optional[List[Action]]=None, indent=0):
        print(f'{" "*indent}visit {id(self)} {node}')
        node.N += 1
        if node.is_terminal():
            print(f'{" "*indent}end visit (terminal) {id(self)} {node}')
            return node.Q

        if isinstance(node, ActionNode):
            if node.N == 1:
                node.expand_leaf(self.model)
                print(f'{" "*indent}end visit (leaf) {id(self)} {node}')
                return node.Q

            if node.spawned_tree is not None:
                chosen_action = []
                node.spawned_tree.visit(node.spawned_tree.root, chosen_action=chosen_action, indent=indent+1)
                if not chosen_action:
                    return node.Q
                assert len(chosen_action) == 1
                action = chosen_action[0]
                child = node.children_by_action[action]
            else:
                child = self.choose_best_child(node, chosen_action=chosen_action)

            leaf_Q = self.visit(child, indent=indent+1)
            node.Q = (node.Q * (node.N - 1) + leaf_Q) / node.N
            print(f'{" "*indent}end visit {id(self)} {node}')
            return leaf_Q

        assert isinstance(node, HiddenStateSamplingNode)
        child = node.sample(self.model)
        leaf_Q = self.visit(child, indent=indent+1)
        node.Q = (node.Q * (node.N - 1) + leaf_Q) / node.N
        print(f'{" "*indent}end visit {id(self)} {node}')
        return leaf_Q


def main():
    nash_model = Model(1/3, 1/3)

    history = [Action.PASS]
    info_set = InfoSet(history)
    info_set.cards[Player.BOB.value] = Card.JACK
    node = ActionNode(info_set)

    ismcts = ISMCTS(nash_model, node)
    distr = ismcts.get_visit_distribution(1000)
    print(distr)


if __name__ == '__main__':
    main()
