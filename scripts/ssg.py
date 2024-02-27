import random
import graphviz

class SimpleStochasticGame:
    def __init__(self, graph):
        self.graph = graph
        self.current_state = 0  # Initial state
        self.is_player1_turn = True

    def play(self):
        while not self.is_terminal_state():
            self.print_current_state()
            if self.is_player1_turn:
                self.player1_move()
            else:
                self.player2_move()
            self.is_player1_turn = not self.is_player1_turn

        print("Game over. Final state:", self.current_state)
        self.print_final_payoff()

    def player1_move(self):
        # Player 1's move strategy: Choose the neighboring state with the maximum payoff.
        neighboring_states = self.graph.get(self.current_state, {})
        if neighboring_states:
            next_state = max(neighboring_states, key=lambda s: neighboring_states[s])
            self.current_state = next_state

    def player2_move(self):
        # Player 2's move strategy: Choose the neighboring state with the minimum payoff.
        neighboring_states = self.graph.get(self.current_state, {})
        if neighboring_states:
            next_state = min(neighboring_states, key=lambda s: neighboring_states[s])
            self.current_state = next_state

    def is_terminal_state(self):
        return self.current_state not in self.graph

    def print_current_state(self):
        print("Current state:", self.current_state)

    def print_final_payoff(self):
        if self.is_terminal_state():
            print("Player 1 payoff:", self.graph.get(self.current_state, 0))

import random

def generate_random_game(num_states, max_neighbors, max_payoff):
    # Generate a random directed graph representing the game
    game_graph = {}
    for state in range(num_states):
        num_neighbors = random.randint(1, max_neighbors)
        neighbors = {random.randint(0, num_states - 1): random.randint(-max_payoff, max_payoff) for _ in range(num_neighbors)}
        game_graph[state] = neighbors

    return game_graph

def plot_graph(game_graph):
    g = graphviz.Digraph('G', filename='ssg.gv')
    # g.edges(self.vertices)
    for state,neighbors in game_graph.items():
        for neighbor,payoff in neighbors.items():
            g.edge(str(state),str(neighbor),label=str(payoff))
        
    g.view()


random_game = generate_random_game(num_states=25, max_neighbors=4, max_payoff=100)
print(random_game)
plot_graph(random_game)
# random_ssg = SimpleStochasticGame(random_game)
# random_ssg.play()

