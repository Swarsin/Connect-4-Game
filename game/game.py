class Node:
    def __init__(self, move, parent):
        self.move = move #what move this node represents
        self.parent = parent #parents of this node
        self.wins = 0 #total games won after this move
        self.total_games = 0 #total games played using this move
        self.children = [] #children of this node

    def CreateChildren(self, children):
        for child in children:
            self.children.append(child)

    def GetUCTValue(self):
        explore = math.sqrt(2)
        if self.total_games == 0:
            return 0 if explore == 0 else math.inf
        return self.wins / self.total_games + explore * math.sqrt(math.log(self.parent.total_games) / self.total_games)

class MCTS:
    def __init__(self, current_board, player1, player2):
        self.root_state = copy.deepcopy(current_board)
        self.root = Node(None, None)
        self.run_time = 0
        self.node_count = 0
        self.simulations = 0
        self.player1 = copy.deepcopy(player1)
        self.player2 = copy.deepcopy(player2)

    def Select(self):
        node = self.root
        state = copy.deepcopy(self.root_state)

        while len(node.children) != 0:
            children = node.children
            values = [child.GetUCTValue() for child in children]
            max_value = max(values)
            best_children = [children[i] for i in range(len(children)) if values[i] == max_value]

            node = random.choice(best_children)
            state.SimulateMove(node.move, self.player1, self.player2)

            if node.total_games == 0:
                return node, state
            
        if self.Expand(node, state):
            node = random.choice(list(node.children))
            state.SimulateMove(node.move, self.player1, self.player2)
        return node, state
    
    def Expand(self, parent, state):
        if state.is_terminal_node():
            return False
        
        children = [Node(move, parent) for move in state.GetValidMoves()]
        parent.CreateChildren(children)
        return True
