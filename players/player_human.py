class Player:
    """Jogador humano que escolhe jogadas via terminal."""

    def __init__(self, rules, size):
        self.rules = rules
        self.size = size

    def play(self, board, current_player, last_move):
        """Pede ao utilizador uma jogada válida (linha, coluna)."""
        while True:
            try:
                move_input = input("Digite a jogada (linha,coluna): ").strip()
                if move_input.lower() in ["q", "quit", "sair"]:
                    print("Jogo terminado pelo utilizador.")
                    exit(0)

                r, c = map(int, move_input.split(","))
                return (r - 1, c - 1)  # converter para índices 0-based

            except ValueError:
                print("Entrada inválida. Use o formato: linha,coluna (ex: 8,9)")
