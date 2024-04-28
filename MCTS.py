from helper_functions import perturb_probs, find_midpoint_overlap, intervals_overlap
from InfoSet import Card, Action, InfoSet, Value, Player
from Model import ConstModel, Model

import numpy as np
from typing import Dict, List, Optional, Tuple


np.set_printoptions(suppress=True)  # avoid scientific notation


DEBUG = True
EPSILON = 1e-6
c_FPU = 0.2
c_PUCT = 4

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
    def __init__(self, info_set: InfoSet, model):
        super().__init__(info_set)
        self.children_by_action: Dict[Action, BaseNode] = {}
        self.spawned_tree: Optional[ISMCTS] = None
        self.model = model
        
        if not self.is_terminal():
            self.eval_model_and_normalize(self.model)
            self.Q = self.V

    def __str__(self):
        return f'Action({self.info_set}, N={self.N}, Q={self.Q}), V={self.V}'

    def expand_leaf(self):
        model = self.model
        self.N = 1
        assert self.game_outcome is None
        self.expand_children(model)
        self.eval_model_and_normalize(model)
        self.Q = self.V

    def expand_children(self, model: Model):
        assert len(self.children_by_action) == 0

        for action in self.valid_actions:
            info_set = self.info_set.apply_move(action)
            child = HiddenStateSamplingNode(model, info_set)
            assert action not in self.children_by_action
            self.children_by_action[action] = child

    def eval_model_and_normalize(self, model: Model):
        # assert self.P is None
        self.P, self.V = model(self.info_set)
        self.P *= self.valid_action_mask

        s = np.sum(self.P)
        if s < EPSILON:
            # just act uniformly random
            self.P = self.valid_action_mask / np.sum(self.valid_action_mask)
        else:
            self.P /= s


class HiddenStateSamplingNode(BaseNode):
    def __init__(self, model: Model, info_set: InfoSet):
        super().__init__(info_set)
        self.model = model
        # self.card_distr, self.eps = model.bayes_prob(self.info_set)
        self.children_by_card: Dict[Card, ActionNode] = {}
        self.Q_range = None

    def __str__(self):
        return f'Hidden({self.info_set}, N={self.N}, Q={self.Q})'

    def sample(self, model: Model) -> ActionNode:
        card_distr, eps = model.bayes_prob(self.info_set)
        card = Card(np.random.choice(3, p=card_distr))
        if DEBUG:
            print(f'  sample {self.info_set} -> {card_distr} -> card={card}')
        node = self.children_by_card.get(card, None)
        return node
    
    def create_children(self, indent=0):
        if DEBUG:
            print(f'{" "*indent}create children for hidden node: {self}')

        model = self.model
        card_dist, eps = model.bayes_prob(self.info_set)
        cp = self.info_set.get_current_player().value

        for c in Card:
            if card_dist[c.value] > 0:
                info_set = self.info_set.clone()
                info_set.cards[cp] = c
                node = ActionNode(info_set, self.model)
                self.children_by_card[c] = node

                if DEBUG:
                    print(f'{" "*indent}created child: {node}')

                if not node.is_terminal():
                    info_set2 = info_set.clone()
                    info_set2.cards[1 - cp] = None
                    spawned_node = ActionNode(info_set2, self.model)
                    node.spawned_tree = ISMCTS(model, spawned_node)
                    if DEBUG:
                        print(f'{" "*indent}created spawned tree at: {spawned_node}')

    def recalcQ(self, indent=0):
        card_dist, eps = self.model.bayes_prob(self.info_set)
        lower, upper = perturb_probs(card_dist, eps)
        self.Q = np.zeros(2)
        Q_lower = np.zeros(2)
        Q_upper = np.zeros(2)
        for card, child in self.children_by_card.items():
            if DEBUG:
                print(f'{" "*indent}child: {child}, Q: {child.Q}, V: {child.V}')
            self.Q += child.Q * card_dist[card.value]
            Q_lower += child.Q * lower[card.value]
            Q_upper += child.Q * upper[card.value]
        self.Q_range = np.stack([Q_lower, Q_upper], axis=0)
        if DEBUG:
            print(f'{" "*indent} recalcQ --> Q: {self.Q}, Q_lower: {Q_lower}, Q_upper: {Q_upper}')

class ISMCTS:
    def __init__(self, model: Model, root: BaseNode):
        self.model = model
        self.root = root

    def get_visit_distribution(self, n: int) -> Dict[Action, float]:
        while self.root.N <= n:
            if DEBUG:
                print(f'------------------- # visit: {self.root.N}   ---------------------')
            self.visit(self.root)
            if self.root.N == 1:
                continue

        n_total = self.root.N
        return {action: node.N / n_total for action, node in self.root.children_by_action.items()}

    def choose_best_child(self, node: ActionNode, chosen_action: Optional[List[Action]]=None):

        actions = node.valid_actions

        P = np.array([node.P[a.value] for a in actions])
        N = np.array([node.children_by_action[a].N for a in actions])

        P_explored = np.sum(P * (N > 0))

        cp = node.info_set.get_current_player().value
        children = [node.children_by_action[a] for a in actions]
        Q = np.array([c.getQ(cp, default=0) for c in children])
        Q_range = [c.Q_range for c in children]

        PUCT_factor = c_PUCT * P * np.sqrt(np.sum(N)) / np.maximum(0.5, N)

        # eps-condition
        if not any([item is None for item in Q_range]):
            # assert False, 'TODO: use cp to decide the side of Q_range'
            Q_int1 = Q_range[0][:, cp]
            Q_int2 = Q_range[1][:, cp]
            is_overlap = intervals_overlap(Q_int1 + PUCT_factor[0], Q_int2 + PUCT_factor[1])
            
            if DEBUG:
                print(f'--PUCT overlap: {is_overlap} Q1: {Q_int1}, Q2: {Q_int2}, PUCT1: {PUCT_factor[0]}, PUCT2: {PUCT_factor[1]}')
                
            if is_overlap:
                PUCT = c_PUCT * P * np.sqrt(np.sum(N)) / np.maximum(0.5, N)
                best_index = np.argmax(PUCT)
                best_action = actions[best_index]

                if chosen_action is not None:
                    chosen_action.append(best_action)
                return node.children_by_action[best_action]

        Q_FPU = node.Q[cp] - c_FPU * np.sqrt(P_explored)
        Q = Q * (N > 0) + Q_FPU * (N < 1)

        PUCT = Q + c_PUCT * P * np.sqrt(np.sum(N)) / np.maximum(0.5, N)
        best_index = np.argmax(PUCT)

        if DEBUG:
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
        if DEBUG:
            print(f'{" "*indent}visit {id(self)} {node}')
        node.N += 1
        if node.is_terminal():
            if DEBUG:
                print(f'{" "*indent}end visit (terminal) {id(self)} {node}')
            return node.Q

        if isinstance(node, ActionNode):
            if node.N == 1:
                node.expand_leaf()
                if DEBUG:
                    print(f'{" "*indent}expanded leaves {id(self)} {node}')
                # return node.Q

            if node.spawned_tree is not None:
                chosen_action = []
                node.spawned_tree.visit(node.spawned_tree.root, chosen_action=chosen_action, indent=indent+1)
                if not chosen_action:
                    return node.Q
                assert len(chosen_action) == 1
                action = chosen_action[0]
                child = node.children_by_action[action]

                if DEBUG:
                    print(f'{"*"*indent}spawned tree action: {action}')

            else:
                child = self.choose_best_child(node, chosen_action=chosen_action)

            leaf_Q = self.visit(child, indent=indent+1)
            node.Q = (node.Q * (node.N - 1) + leaf_Q) / node.N
            if DEBUG:
                print(f'{" "*indent}end visit {id(self)} {node}, return leaf_Q: {leaf_Q}')
            return leaf_Q

        assert isinstance(node, HiddenStateSamplingNode)
        
        if not node.children_by_card:
            node.create_children(indent=indent)
        child = node.sample(self.model)

        leaf_Q = self.visit(child, indent=indent+1)
        node.recalcQ(indent)
        if DEBUG:
            print(f'{" "*indent}end visit {id(self)} {node}')

        return leaf_Q


def main():
    # nash_model = Model(1/3, 1/3)
    model = ConstModel(1/3, 1/3, h=0.75, eps=0.05)

    # history = [Action.PASS, Action.ADD_CHIP]
    # info_set = InfoSet(history)
    # info_set.cards[Player.ALICE.value] = Card.QUEEN
    # node = ActionNode(info_set)


    history = [Action.PASS]
    info_set = InfoSet(history)
    info_set.cards[Player.BOB.value] = Card.JACK
    node = ActionNode(info_set, model)

    ismcts = ISMCTS(model, node)
    distr = ismcts.get_visit_distribution(200)
    print(distr)


if __name__ == '__main__':
    main()
