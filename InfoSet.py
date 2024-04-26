from enum import Enum
from typing import List, Optional
import numpy as np

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

Value = np.ndarray

class InfoSet:
    def __init__(self, action_history: List[Action]=[], cards=None):
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