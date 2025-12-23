import os
import time
import random
from collections import deque
from typing import List, Tuple, Optional
import numpy as np
from network import PyTorchModel
# from mcts.mcts_alpha import MCTS
# from mcts.mcts_alpha_with_noise import MCTS
from mcts.new_mcts_alpha import MCTS
from players.player_alpha import Player
from games.gomoku import Gomoku as GameClass
from games.pente import Pente as GameClass
from datetime import datetime
from copy import deepcopy
import sys
import gc

# mapear nomes de jogos para classes dinamicamente dentro do loop

# -------------------------
#  Utils
# -------------------------
def softmax_temperature(pi: np.ndarray, temp: float) -> np.ndarray:
    if temp <= 0:
        return pi
    logits = np.log(pi + 1e-15)
    logits = logits / temp
    exps = np.exp(logits - np.max(logits))
    p = exps / np.sum(exps)
    return p


def sample_action_from_pi(pi: np.ndarray, temp: float) -> int:
    if temp == 0:
        return int(np.argmax(pi))
    p = softmax_temperature(pi, temp)
    return int(np.random.choice(len(p), p=p))


# -------------------------
#  Replay Buffer
# -------------------------
class ReplayBuffer:
    def __init__(self, capacity: int = 20000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """
        Adiciona lista de exemplos (state_enc, pi, z)
        state_enc: (C,H,W)
        pi: (action_size,)
        z: scalar (-1,0,1)
        """
        for ex in examples:
            self.buffer.append(ex)

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, k=batch_size)
        states, pis, zs = zip(*batch)
        states = np.stack(states, axis=0).astype(np.float32)
        pis = np.stack(pis, axis=0).astype(np.float32)
        zs = np.array(zs, dtype=np.float32).reshape(-1, 1)
        return states, pis, zs

    def __len__(self):
        return len(self.buffer)


# -------------------------
#  Self-play single game
# -------------------------
def play_game_and_collect(mcts: MCTS, game, temp_fn, max_moves=225, use_symmetries=True):
    """
    Joga um jogo completo e retorna exemplos augmentados:
    final_examples: list of (state_enc (C,H,W), pi (A,), z scalar)
    winner: 0/1/2
    """
    examples = []
    move_number = 0

    while True:
        state_enc = game.get_encoded_state()  # perspective-invariant is expected
        pi = mcts.run(game, len(game.move_history))  # vector (action_size,)
        pi_for_store = pi.copy()

        temp = temp_fn(move_number)
        action = sample_action_from_pi(pi, temp)

        # fallback safety: if chosen action illegal, use argmax
        valid_mask = game.get_valid_moves()
        if valid_mask[action] != 1.0:
            action = int(np.argmax(pi))
        # store (state, pi, player)
        examples.append((state_enc, pi_for_store, int(game.current_player)))

        # apply move
        r, c = divmod(action, game.size)
        game.do_move((r, c))

        move_number += 1

        if game.is_game_over() or move_number >= max_moves:
            break

    winner = game.get_winner()  # 0/1/2

    # convert examples to (state_aug, pi_aug, z)
    final_examples = []
    for state_enc, pi_vec, player in examples:
        if winner == 0:
            z = 0.0
        else:
            z = 1.0 if winner == player else -1.0

        if use_symmetries:
            syms = mcts.symmetries(state_enc, pi_vec)
            for s_aug, pi_aug in syms:
                final_examples.append((s_aug.astype(np.float32), pi_aug.astype(np.float32), z))
        else:
            final_examples.append((state_enc.astype(np.float32), pi_vec.astype(np.float32), z))

    return final_examples, winner


# -------------------------
#  Evaluation between models
# -------------------------
def evaluate_models(model_new: PyTorchModel,
                model_best: PyTorchModel,
                game_name: str,
                n_games: int = 20,
                n_simulations: int = 100,
                cpuct: float = 1.0) -> Tuple[float, int]:
    """
    Joga n_games entre model_new e model_best (alternando quem começa).
    Retorna (win_rate_of_new, draws)
    """
    # select game class
    if game_name.lower().startswith("pente"):
        rules_name = "pente"
    else:
        rules_name = "gomoku"

    new_wins = 0
    draws = 0
    total = n_games

    for i in range(n_games):
        game = GameClass(size=model_new.board_size)

        r = random.randint(0, 14)
        c = random.randint(0, 14)
        game.do_move((r, c))

        # Determine quem começa: new starts em jogos pares
        new_starts = (i % 2 == 0)
        move_number = 1

        # create MCTS instances for both players (they each use their model when expanded)
        mcts_new = MCTS(game_class=GameClass, n_simulations=n_simulations, nn_model=model_new, cpuct=cpuct, dirichlet_alpha=0.03, epsilon=0.03, apply_dirichlet_n_first_moves=10, add_dirichlet_noise=False)
        mcts_best = MCTS(game_class=GameClass, n_simulations=n_simulations, nn_model=model_best, cpuct=cpuct, dirichlet_alpha=0.03, epsilon=0.03, apply_dirichlet_n_first_moves=10, add_dirichlet_noise=False)

        while not game.is_game_over():
            # Decida quem vai jogar baseado em quem é o atual jogador e quem começou
            if (game.current_player == 1 and new_starts) or (game.current_player == 2 and not new_starts):
                pi = mcts_new.run(game, len(game.move_history))
            else:
                pi = mcts_best.run(game, len(game.move_history))

            # Escolha determinística (argmax)
            action = int(np.argmax(pi))
            r, c = divmod(action, game.size)
            game.do_move((r, c))
            move_number += 1
            if move_number > game.size * game.size:
                break

        winner = game.get_winner()
        if winner == 0:
            draws += 1
        else:
            # determine if new model got the win
            if (winner == 1 and new_starts) or (winner == 2 and not new_starts):
                new_wins += 1

        mcts_new.clear_tree()
        mcts_best.clear_tree()
        del mcts_new
        del mcts_best
        gc.collect()

    win_rate = new_wins / float(total)
    return new_wins, win_rate, draws


# -------------------------
#  Main train loop
# -------------------------
def train_alphazero(
    game_name: str = "gomoku",
    board_size: int = 15,
    num_iterations: int = 5,
    games_per_iteration: int = 8,
    n_simulations: int = 50,
    buffer_size: int = 10000,
    batch_size: int = 128,
    epochs_per_iter: int = 2,
    temp_threshold: int = 8,
    eval_games: int = 12,
    eval_mcts_simulations: int = 200,
    win_rate_threshold: float = 0.55,
    cpuct: float = 1.2,
    model_dir: str = "models",
    save_every: int = 1,
    pretrained_model_path: Optional[str] = None,  # Novo parâmetro para passar o modelo pré-treinado
    next_iteration_continuation: int = 1
):
    """
    Pipeline central de treino.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Calcular o tamanho da ação baseado no board_size
    action_size = board_size * board_size  # para Gomoku (ou Pente), as ações são posições no tabuleiro

    # Verificar se há um modelo pré-existente
    if pretrained_model_path and os.path.exists(pretrained_model_path):
        print(f"Carregando modelo pré-existente de: {pretrained_model_path}")
        model_best = PyTorchModel(board_size=board_size, action_size=action_size)
        model_best.load(pretrained_model_path)  # Carregar o modelo pré-treinado
        model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
        model_candidate.net.load_state_dict(model_best.net.state_dict())
        print("Modelo pré-existente carregado com sucesso.")
    else:
        print("Nenhum modelo pré-existente encontrado. Inicializando novo modelo.")
        model_best = PyTorchModel(board_size=board_size, action_size=action_size)
        model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)

    # replay buffer
    buffer = ReplayBuffer(capacity=buffer_size)

    # temperature schedule
    def temp_fn(move_number: int):
        return max(0.0, 1.0 - move_number / temp_threshold)

    for it in range(next_iteration_continuation, next_iteration_continuation + num_iterations + 1):
        t0 = time.time()
        print(f"\n=== ITER {it}/{next_iteration_continuation + num_iterations}: Self-play generation (games={games_per_iteration}, sims={n_simulations}) ===")

        # self-play generation using candidate model
        winners = {0: 0, 1: 0, 2: 0}
        for g in range(games_per_iteration):
            mcts_play = MCTS(game_class=GameClass, n_simulations=n_simulations, nn_model=model_candidate, cpuct=cpuct, dirichlet_alpha=0.03, epsilon=0.03, apply_dirichlet_n_first_moves=10, add_dirichlet_noise=True)
            game = GameClass(size=board_size)
            game.current_player = 1
            examples, winner = play_game_and_collect(mcts_play, game, temp_fn, max_moves=board_size * board_size, use_symmetries=True)
            buffer.add(examples)
            winners[winner] = winners.get(winner, 0) + 1
            print(f"  gen game {g+1}/{games_per_iteration} -> winner={winner}, buffer_size={len(buffer)}")

            mcts_play.clear_tree()
            del mcts_play
            gc.collect()

        # training candidate model if enough samples
        if len(buffer) >= batch_size:
            print(f"\nTraining candidate model: buffer={len(buffer)}, batch_size={batch_size}, epochs_per_iter={epochs_per_iter}")
            n_batches = max(1, len(buffer) // batch_size)
            for epoch in range(epochs_per_iter):
                epoch_t0 = time.time()
                for b in range(n_batches):
                    states_b, pis_b, zs_b = buffer.sample(batch_size)
                    loss_info = model_candidate.train_batch(states_b, pis_b, zs_b, epochs=1)
                epoch_t1 = time.time()
                print(f"  epoch {epoch+1}/{epochs_per_iter} finished in {epoch_t1 - epoch_t0:.1f}s, last_loss={loss_info}")
        else:
            print(f"Not enough examples to train (buffer={len(buffer)}, need {batch_size}). Skipping training this iter.")

        # evaluation
        print("\nEvaluating candidate vs best...")
        try:
            new_wins, win_rate, draws = evaluate_models(model_candidate, model_best, game_name, n_games=eval_games, n_simulations=eval_mcts_simulations, cpuct=cpuct)
        except Exception as e:
            print("Evaluation failed (exception):", e)
            win_rate, draws = 0.0, 0

        print(f" Candidate win rate = {win_rate:.3f} ({new_wins}/{eval_games}) (draws={draws})")

        # accept/reject
        if win_rate >= win_rate_threshold:
            print(" Candidate accepted -> promote to best.")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(model_dir, f"model_best_iter{it}_{timestamp}.pt")
            model_candidate.save(path)
            model_best = model_candidate
            # create new candidate starting from best
            model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
            model_candidate.net.load_state_dict(model_best.net.state_dict())
            model_candidate.optimizer.load_state_dict(model_best.optimizer.state_dict())
        else:
            print(" Candidate rejected -> restore candidate from best.")
            # reset candidate from best weights
            model_candidate = PyTorchModel(board_size=board_size, action_size=action_size)
            model_candidate.net.load_state_dict(model_best.net.state_dict())
            model_candidate.optimizer.load_state_dict(model_best.optimizer.state_dict())

        # periodic save of best model status
        if it % save_every == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = os.path.join(model_dir, f"snapshot_iter{it}_{timestamp}.pt")
            model_best.save(snapshot_path)
            print(f" Saved snapshot: {snapshot_path}")

        t1 = time.time()
        print(f"Iteration {it} done in {(t1 - t0):.1f}s. Winners this iter: {winners}")

    print("\n=== TRAINING COMPLETE ===")

# -------------------------
#  Entrypoint
# -------------------------
if __name__ == "__main__":
    train_alphazero(
        game_name="gomoku",           # Jogo Gomoku
        board_size=15,                # Tamanho do tabuleiro (15x15)

        num_iterations=200,           # 200 iterações de treinamento
        games_per_iteration=70,       # 70 jogos por iteração de treinamento

        n_simulations=3000,          # 3000 simulações para MCTS
        cpuct=1.0,                   # Factor de exploração/exploração para MCTS

        buffer_size=25000,           # Buffer de replay com até 25.000 exemplos
        batch_size=128,               # 128 exemplos por batch de treinamento
        epochs_per_iter=3,           # 3 épocas por iteração de treinamento

        temp_threshold=10,           # Temperatura para exploração
        eval_games=50,               # 50 jogos de avaliação
        eval_mcts_simulations=3000,
        win_rate_threshold=0.55,     # Aceitar modelo candidato se vencer 55% dos jogos

        model_dir="models",          # Diretório para salvar os modelos
        save_every=1,                # Salvar modelo a cada iteração
        pretrained_model_path="models_gomoku/model_best_iter53_20251207_161711.pt",  # Caminho para o modelo pré-treinado

        next_iteration_continuation = 35
    )

