import copy
import math
import random
import numpy as np

# -------------------- TREE NODE -------------------- #
class TreeNode:
    """Nó da árvore MCTS com PUCT e heurísticas."""
    def __init__(self, parent=None, prior=1.0, move=None, state=None):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior
        self.move = move
        self.state = state

    def is_leaf(self):
        return self._children == {}

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                next_state = copy.deepcopy(self.state)
                next_state.do_move(action)
                self._children[action] = TreeNode(
                    parent=self,
                    prior=prob,
                    move=action,
                    state=next_state
                )

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self._n_visits += 1
        self._Q += (leaf_value - self._Q) / self._n_visits

    def get_value(self, c_puct):
        self._u = (c_puct * self._P * math.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def select(self, c_puct):
        # fallback seguro se não houver filhos
        if not self._children:
            return None, self
        return max(self._children.items(), key=lambda item: item[1].get_value(c_puct))


# -------------------- MCTS -------------------- #
class MCTSGomoku:
    def __init__(self, n_playout=100, c_puct=1.4):
        self._root = None
        self.n_playout = n_playout
        self.c_puct = c_puct

    # -------------------- HEURÍSTICAS -------------------- #
    def _heuristic_policy(self, state):
        """Retorna lista de (ação, probabilidade) baseada em heurística."""
        moves = state.get_legal_moves()
        if not moves:
            return []

        action_scores = []
        for move in moves:
            r, c = move
            attack = self._threat_score(state, r, c, state.current_player)
            defense = self._threat_score(state, r, c, 3 - state.current_player)
            dist_center = -abs(r - state.size // 2) - abs(c - state.size // 2)
            score = 2.0 * attack + 1.5 * defense + 0.1 * dist_center
            action_scores.append(score)

        # fallback se todos scores forem zero
        if sum(action_scores) == 0:
            action_scores = [1.0] * len(moves)

        probs = np.array(action_scores)
        probs = np.exp(probs - np.max(probs))
        probs /= probs.sum()
        return list(zip(moves, probs))

    def _threat_score(self, state, r, c, player):
        board = state.board
        size = state.size
        score = 0
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for dr, dc in directions:
            count = 1
            open_ends = 0
            rr, cc = r + dr, c + dc
            while 0 <= rr < size and 0 <= cc < size and board[rr,cc] == player:
                count += 1
                rr += dr
                cc += dc
            if 0 <= rr < size and 0 <= cc < size and board[rr,cc] == 0:
                open_ends += 1
            rr, cc = r - dr, c - dc
            while 0 <= rr < size and 0 <= cc < size and board[rr,cc] == player:
                count += 1
                rr -= dr
                cc -= dc
            if 0 <= rr < size and 0 <= cc < size and board[rr,cc] == 0:
                open_ends += 1
            if count >= 5:
                score += 100
            elif count == 4 and open_ends == 2:
                score += 50
            elif count == 4 and open_ends == 1:
                score += 25
            elif count == 3 and open_ends == 2:
                score += 10
            elif count == 3 and open_ends == 1:
                score += 4
            elif count == 2 and open_ends == 2:
                score += 2
        return score

    # -------------------- MCTS PLAYOUT -------------------- #
    def _playout(self, state):
        node = self._root
        while not node.is_leaf():
            move, node = node.select(self.c_puct)
            if move is None:
                break

        action_probs = self._heuristic_policy(node.state)
        if action_probs:
            node.expand(action_probs)
        leaf_value = self._simulate(node.state)
        node.update_recursive(-leaf_value)

    def _simulate(self, state, max_depth=3):
        depth = 0
        while not state.is_game_over() and depth < max_depth:
            moves = state.get_legal_moves()
            if not moves:
                break

            # Vitória imediata
            for move in moves:
                test = copy.deepcopy(state)
                test.do_move(move)
                if test.check_winner() == state.current_player:
                    state.do_move(move)
                    return 1

            # Bloqueio imediato
            for move in moves:
                test = copy.deepcopy(state)
                test.current_player = 3 - state.current_player
                test.do_move(move)
                if test.check_winner() == (3 - state.current_player):
                    state.do_move(move)
                    return 0.8
                
            else:
                # fallback seguro
                action_probs = self._heuristic_policy(state)
                if action_probs:
                    moves_list, probs = zip(*action_probs)
                    move = random.choices(moves_list, weights=probs, k=1)[0]
                else:
                    move = random.choice(moves)
                state.do_move(move)

            depth += 1

        winner = state.check_winner()
        if winner == 0:
            return 0.5
        elif winner == state.current_player:
            return 1
        else:
            return 0

    # -------------------- GET MOVE -------------------- #
    def get_move(self, state):
        self._root = TreeNode(state=copy.deepcopy(state))
        for _ in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # Fallback seguro caso não haja filhos
        if not self._root._children:
            moves = state.get_legal_moves()
            if moves:
                return random.choice(moves)
            else:
                return None

        return max(self._root._children.items(), key=lambda x: x[1]._n_visits)[0]    


class MCTSPente:
    def __init__(self, n_playout=120, c_puct=1.4):
        self._root = None
        self.n_playout = n_playout
        self.c_puct = c_puct

    # -------------------- HEURÍSTICAS -------------------- #
    def _heuristic_policy(self, state):
        """Retorna lista de (ação, probabilidade) baseada em heurística adaptada ao Pente."""
        moves = state.get_legal_moves()
        if not moves:
            return []

        action_scores = []
        for move in moves:
            r, c = move
            # ameaça e defesa (como no Gomoku)
            attack = self._threat_score(state, r, c, state.current_player)
            defense = self._threat_score(state, r, c, 3 - state.current_player)
            # potencial de captura
            capture_potential = self._capture_potential(state, r, c, state.current_player)
            # centralização
            dist_center = -abs(r - state.size // 2) - abs(c - state.size // 2)

            # ponderação
            score = (
                2.0 * attack +
                1.5 * defense +
                3.0 * capture_potential +
                0.1 * dist_center
            )
            action_scores.append(score)

        if sum(action_scores) == 0:
            action_scores = [1.0] * len(moves)

        probs = np.array(action_scores)
        probs = np.exp(probs - np.max(probs))
        probs /= probs.sum()
        return list(zip(moves, probs))

    def _threat_score(self, state, r, c, player):
        """Avalia ameaças ofensivas/defensivas — 5 em linha."""
        board = state.board
        size = state.size
        score = 0
        directions = [(1,0),(0,1),(1,1),(1,-1)]
        for dr, dc in directions:
            count = 1
            open_ends = 0
            rr, cc = r + dr, c + dc
            while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                count += 1
                rr += dr
                cc += dc
            if 0 <= rr < size and 0 <= cc < size and board[rr, cc] == 0:
                open_ends += 1

            rr, cc = r - dr, c - dc
            while 0 <= rr < size and 0 <= cc < size and board[rr, cc] == player:
                count += 1
                rr -= dr
                cc -= dc
            if 0 <= rr < size and 0 <= cc < size and board[rr, cc] == 0:
                open_ends += 1

            if count >= 5:
                score += 120
            elif count == 4 and open_ends >= 1:
                score += 60
            elif count == 3 and open_ends >= 1:
                score += 15
            elif count == 2 and open_ends >= 1:
                score += 4
        return score

    def _capture_potential(self, state, r, c, player):
        """Calcula o potencial de capturar pares inimigos."""
        board = state.board
        size = state.size
        opp = 3 - player
        captures = 0
        directions = [(1,0), (0,1), (1,1), (1,-1)]
        for dr, dc in directions:
            r1, c1 = r + dr, c + dc
            r2, c2 = r + 2*dr, c + 2*dc
            r3, c3 = r + 3*dr, c + 3*dc
            if (
                0 <= r3 < size and 0 <= c3 < size and
                board[r1, c1] == opp and
                board[r2, c2] == opp and
                board[r3, c3] == player
            ):
                captures += 1
        return captures * 20  # peso forte para capturas potenciais

    # -------------------- MCTS PLAYOUT -------------------- #
    def _playout(self, state):
        node = self._root
        while not node.is_leaf():
            move, node = node.select(self.c_puct)
            if move is None:
                break

        action_probs = self._heuristic_policy(node.state)
        if action_probs:
            node.expand(action_probs)
        leaf_value = self._simulate(node.state)
        node.update_recursive(-leaf_value)

    def _simulate(self, state, max_depth=3):
        depth = 0
        while not state.is_game_over() and depth < max_depth:
            moves = state.get_legal_moves()
            if not moves:
                break

            # Vitória imediata (por 5 em linha ou 5 capturas)
            for move in moves:
                test = copy.deepcopy(state)
                test.do_move(move)
                if test.check_winner() == state.current_player:
                    state.do_move(move)
                    return 1

            # Bloqueio imediato
            for move in moves:
                test = copy.deepcopy(state)
                test.current_player = 3 - state.current_player
                test.do_move(move)
                if test.check_winner() == (3 - state.current_player):
                    state.do_move(move)
                    return 0.8

            # fallback heurístico
            action_probs = self._heuristic_policy(state)
            if action_probs:
                moves_list, probs = zip(*action_probs)
                move = random.choices(moves_list, weights=probs, k=1)[0]
            else:
                move = random.choice(moves)
            state.do_move(move)
            depth += 1

        winner = state.check_winner()
        if winner == 0:
            return 0.5
        elif winner == state.current_player:
            return 1
        else:
            return 0

    # -------------------- GET MOVE -------------------- #
    def get_move(self, state):
        self._root = TreeNode(state=copy.deepcopy(state))
        for _ in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        if not self._root._children:
            moves = state.get_legal_moves()
            return random.choice(moves) if moves else None

        return max(self._root._children.items(), key=lambda x: x[1]._n_visits)[0]