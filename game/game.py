import pygame, sys, random, copy, math, time
#Libraries used:
#Pygame - GUI for the game, displays board and allows the user to drop pieces onto the board
#sys - sys.exit() upon user quitting window
#random - to randomly choose who gets first turn
#copy - to create a deepcopy of the game board class - used to evaluate simulated game states and decide best move in Minimax algorithm
#math - math.inf for best and -math.inf for worst possibe move in minimax

pygame.init()

SQUARE_SIZE = 100
RADIUS = int(SQUARE_SIZE/2 - 10)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)


info = pygame.display.Info()
screen = pygame.display.set_mode((info.current_w, info.current_h), pygame.RESIZABLE)
font = pygame.font.SysFont("arial", 75)
clock = pygame.time.Clock()
PLAYER_TURN = 0
AI_TURN = 1

EMPTY = 0

class Stack:
    def __init__(self):
        self.items = []
        self.pointer = 0

    def IsEmpty(self):
        return self.pointer == 0
    
    def Push(self, item):
        if len(self.items) <= self.pointer:
            self.items += [0] * (self.pointer - len(self.items) + 1)
        self.items[self.pointer] = item
        self.pointer += 1

    def Pop(self): #change this so it uses pointers - wouldn't get marks as it is right now
        # if not self.IsEmpty():
        #     return self.items.pop(item)
        if self.pointer < 0:
            return None  # or raise an exception for an empty stack
        else:
            self.pointer -= 1  # decrement pointer after popping an item
            self.items[self.pointer] = 0
    
    def Fetch(self, index): #to get item at specific index of the stack object
        if 0 <= index < len(self.items):        
            return self.items[index]
        else:
            return 0
        
    def GetSize(self):
        return len(self.items)
    
    # def __str__(self): #for testing/debugging
    #     return str(self.items)
    
    # def __repr__(self): #for testing/debugging
    #     return str(self.items)

class Board:
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.board = [Stack() for column in range(self.columns)] #uses composition - creating more complex objects (board) by combining simpler ones(stack) - also if the board class is scrapped, the stacks would be destroyed too as the form an integral part of the board. 
        self.full_columns = []
        self.window_length = 4
        self.moves_history = []
    
    def GetRows(self):
        return self.rows
    
    def GetColumns(self):
        return self.columns

    def DropPiece(self, column, player_piece):
        self.board[column].Push(player_piece)

    def IsValidMove(self, column):
        if not  0 <= column <= 6:
            return False
        if self.board[column].GetSize() == 6:
            return False
        return True
    
    def GetValidMoves(self):
        valid_moves = [column for column in range(self.columns) if self.IsValidMove(column)]
        return valid_moves
    
    def SimulateMove(self, column, player1, player2):
        if self.IsValidMove(column):
            self.DropPiece(column, player1.GetPiece() if player1.GetTurn() else player2.GetPiece())
            if player1.GetTurn():
                player1.SetTurn(False)
                player2.SetTurn(True)
            else:
                player2.SetTurn(False)
                player1.SetTurn(True)

    def CheckForWin(self, player_piece):
        #Check for horizontal win
        for column in range(self.columns-3): #(column_count - 3) since the last 3 columns can't contain 4 pieces in a row horizontally
            for row in range(self.rows):
                if self.board[column].Fetch(row) == player_piece and self.board[column + 1].Fetch(row) == player_piece and self.board[column + 2].Fetch(row) == player_piece and self.board[column + 3].Fetch(row) == player_piece:
                    return True
        #Check for vertical win        
        for column in range(self.columns):
            for row in range(self.rows-3): #(row_count - 3) since the top 3 rows can't contain 4 pieces in a row vertically
                if self.board[column].Fetch(row) == player_piece and self.board[column].Fetch(row + 1) == player_piece and self.board[column].Fetch(row + 2) == player_piece and self.board[column].Fetch(row + 3) == player_piece:
                    return True
        #Check for win where diagonal going up
        for column in range(self.columns-3): #can't have a diagonal going up beyond the middle (4th) column so subtract 3
            for row in range(self.rows-3): #can't have a diagonal going up beyond the 3rd row so subtract 3
                if self.board[column].Fetch(row) == player_piece and self.board[column + 1].Fetch(row + 1) == player_piece and self.board[column + 2].Fetch(row + 2) == player_piece and self.board[column + 3].Fetch(row + 3) == player_piece:
                    return True
        #Check for win where diagonal going down
        for column in range(self.columns-3): #can't have a diagonal going down beyond the 4th row
            for row in range(3, self.columns): #cant have a diagonal going down before the 4th column
                if self.board[column].Fetch(row) == player_piece and self.board[column + 1].Fetch(row - 1) == player_piece and self.board[column + 2].Fetch(row - 2) == player_piece and self.board[column + 3].Fetch(row - 3) == player_piece:
                    return True
        return False
    
    def EvaluateWindow(self, window, player_piece):
        score = 0 
        opponent_piece = 1 #PLAYER_PIECE
        #evaluating player move:
        if player_piece == 1:
            opponent_piece = 2 #AI_PIECE
        if window.count(player_piece) == 4:
            score += 100   
        elif window.count(player_piece) == 3 and window.count(EMPTY) == 1:
            score += 5
        elif window.count(player_piece) == 2 and window.count(EMPTY) == 2:
            score += 2
        return score
        
    def ScorePosition(self, player_piece):
        score = 0

        #score centre column
        centre_array = [self.board[self.columns//2].Fetch(row) for row in range(self.rows)]
        centre_count = centre_array.count(player_piece)
        score += centre_count * 3

        #score horizontally
        for row in range(self.rows):
            row_array = [self.board[column].Fetch(row) for column in range(self.columns)] #creatint an array of the entire row
            for column in range(4):#(self.column - 3):
                window = row_array[column:column + self.window_length]
                score += self.EvaluateWindow(window, player_piece)

        #score vertically
        for column in range(self.columns):
            column_array = [self.board[column].Fetch(row) for row in range(self.rows)]
            for row in range(self.rows - 3):
                window = column_array[row:row + self.window_length]
                score += self.EvaluateWindow(window, player_piece)

        #score in a positively sloped diagonal
        for row in range(self.rows - 3):
            for column in range(self.columns - 3):
                window = [self.board[column + i].Fetch(row + i) for i in range(self.window_length)]
                score += self.EvaluateWindow(window, player_piece)

        #score in a negatively sloped diagonal
        for row in range(self.rows - 3):
            for column in range(self.columns - 3):
                window = [self.board[column + 3 - i].Fetch(row + i) for i in range(self.window_length)]
                score += self.EvaluateWindow(window, player_piece)

        return score

    # def GetBestMove(self, player_piece):
    #     valid_moves = self.GetValidMoves()#self.board.GetValidMoves()
    #     highest_score = -10000
    #     best_move = random.choice(valid_moves)
    #     for move in valid_moves: #for each valid column
    #         temp_board = copy.deepcopy(board1)
    #         temp_board.DropPiece(move, player_piece)
    #         score = temp_board.ScorePosition(player_piece)
    #         if score > highest_score:
    #             highest_score = score
    #             best_move = move
    #     return best_move
    
    def CheckForDraw(self):
        for column in range(self.columns):
            if self.board[column].GetSize() == 6:
                if column not in self.full_columns:
                    self.full_columns.append(column)
        if len(self.full_columns) == 7:
            return True
        return False
        

    def PrintBoard(self): #displays on terminal - for original testing before GUI
        for row in range(self.rows - 1, -1, -1): #is self.rows - 1 because the row indices is 0 to 5 and this for loops runs from 0 to 5 (matches the row indices of the board)
            for column in range(self.columns):
                print(self.board[column].Fetch(row), end = " ")
            print("")
        print("")

    def DisplayBoard(self, player1, player2):
        rect = pygame.Rect(0, 0, 100, 100)
        for column in range(self.columns):
            for row in range(self.rows):
                rect.center = ((info.current_w//2) - 300 + column*(SQUARE_SIZE), (info.current_h/2) - 250 + row*(SQUARE_SIZE))
                pygame.draw.rect(screen, BLUE, rect)
                pygame.draw.circle(screen, BLACK, ((info.current_w//2) - 300 + column*(SQUARE_SIZE), (info.current_h/2) - 250 + row*(SQUARE_SIZE)), RADIUS)
        for column in range(self.columns):
            for row in range(self.rows):        
                if self.board[column].Fetch(row) == 1:
                    pygame.draw.circle(screen, player1.GetColour(), ((info.current_w//2) - 300 + column*(SQUARE_SIZE), (info.current_h - ((info.current_h/2) - 250 + row*(SQUARE_SIZE)))), RADIUS) #I don't understand why +280 here, maybe the board isn't actually centred?
                elif self.board[column].Fetch(row) == 2:
                    pygame.draw.circle(screen, player2.GetColour(), ((info.current_w//2) - 300 + column*(SQUARE_SIZE), (info.current_h - ((info.current_h/2) - 250 + row*(SQUARE_SIZE)))), RADIUS) #I don't understand why +280 here, maybe the board isn't actually centred?
        pygame.display.update()

    def is_terminal_node(self):
        return self.CheckForWin(1) or self.CheckForWin(2) or self.CheckForDraw()
    
    def minimax(self, board, depth, alpha, beta, max_player):
        valid_moves = board.GetValidMoves()
        is_terminal = board.is_terminal_node()
        if depth == 0 or is_terminal:
            if is_terminal:
                if board.CheckForWin(2):
                    return (None, 1000000000)
                elif board.CheckForWin(1):
                    return (None, -1000000000)
                else: #game over, so no more valid moves
                    return (None, 0)
            else: #depth = zero
                return (None, board.ScorePosition(2))
        if max_player:
            value = -math.inf
            column = random.choice(valid_moves)
            for move in valid_moves: #for column in valid columns
                temp_board = copy.deepcopy(board)
                temp_board.DropPiece(move, 2)
                #new_score = max(value, minimax(temp_board, depth-1, False))
                new_score = board.minimax(temp_board, depth-1, alpha, beta, False)[1]
                if new_score > value:
                    value = new_score
                    column = move
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return column, value
            
        else: #for minimising player
            value = math.inf
            column = random.choice(valid_moves)
            for move in valid_moves:
                temp_board = copy.deepcopy(board)
                temp_board.DropPiece(move, 1)
                new_score = board.minimax(temp_board, depth-1, alpha, beta, True)[1]
                if new_score < value:
                    value = new_score
                    column = move
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return column, value
    
    def Store(self, move):
        self.moves_history.append(move)
        
    def Undo(self):
        for i in range(2):
            last_move = self.moves_history.pop(-1)
            self.board[last_move].Pop()

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
    
    def Simulate(self, state):
        while not state.is_terminal_node():
            state.SimulateMove(random.choice(state.GetValidMoves()), self.player1, self.player2)
        return (self.player1.GetPiece() if state.CheckForWin(self.player1.GetPiece()) else self.player2.GetPiece() if state.CheckForWin(self.player2.GetPiece()) else None)
    

    def Backpropagate(self, node, turn, outcome):
        reward = 0 if outcome == turn else 1

        if node is not None:
            node.total_games += 1
            node.wins += reward
            if outcome is not None:
                self.Backpropagate(node.parent, turn, None)
            else:
                self.Backpropagate(node.parent, turn, 1 - reward)
                
    def Search(self, time_limit):
        start_time = time.process_time()

        simulations = 0

        while time.process_time() - start_time < time_limit:
            node, state = self.Select()
            outcome = self.Simulate(state)
            self.Backpropagate(node, self.player1.GetPiece() if self.player1.GetTurn() else self.player2.GetPiece(), outcome)
            simulations += 1

        run_time = time.process_time() - start_time
        self.run_time = run_time
        self.simulations = simulations

    def GetBestMove(self):
        if self.root_state.is_terminal_node():
            return -1
        max_value = max(self.root.children, key=lambda x: x.total_games).total_games
        max_nodes = [x for x in self.root.children if x.total_games == max_value]
        best_child = random.choice(max_nodes)
        print(best_child.move)
        return best_child.move
    
    def UpdateMove(self, move):
        if move in self.root.children:
            self.root_state.SimulateMove(move, self.player1, self.player2)
            self.root = self.root.children[move]
        
        self.root_state.SimulateMove(move, self.player1, self.player2)
        self.root = Node(None, None)
    
    def GetStatistics(self):
        return self.run_time, self.simulations

class Player:
    def __init__(self, piece, turn, colour, username):
        self.username = username
        self.piece = piece
        self.turn = turn
        self.colour = colour
    
    def GetUsername(self):
        return self.username
    
    def GetPiece(self):
        return self.piece
    
    def GetTurn(self):
        return self.turn
    
    def SetTurn(self, new):
        self.turn = new

    def GetColour(self):
        return self.colour

class AIPlayer(Player): #Inherits from Player class but still need to implement into the actual game
    def __init__(self, piece, turn, colour, difficulty):
        super().__init__(piece, turn, colour, username="")
        self.difficulty = difficulty
    
    def GetUsername(self): #MAKE A BETTER EXAMPLE OF OVERRIDING
        return "AI Player"

    def GetPiece(self):
        return self.piece
    
    def GetTurn(self):
        return self.turn
    
    def SetTurn(self, new):
        self.turn = new

    def GetColour(self):
        return self.colour
    
    def GetDifficulty(self):
        return self.difficulty


board = Board()
player_one = Player(1, False, RED, "Player 1")
player_two = AIPlayer(2, True, YELLOW, 5)
mcts = MCTS(board, player_one, player_two)


def MainMCTS(player1, player2, board1, mcts_obj):
    game_over = False
    last_click_time = 0
    board1.PrintBoard()
    board1.DisplayBoard(player1, player2)
    while not game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                if event.type == pygame.MOUSEMOTION:
                    pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2)))
                    x_pos = event.pos[0]
                    if player1.GetTurn():
                        pygame.draw.circle(screen, player1.GetColour(), (x_pos, (((info.current_h-600)/2)-RADIUS)), RADIUS)
                    else:
                        pygame.draw.circle(screen, player2.GetColour(), (x_pos, (((info.current_h-600)/2)-RADIUS)), RADIUS)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        try:
                            board1.Undo()
                            board1.PrintBoard()
                            board1.DisplayBoard(player1, player2)
                        except Exception:
                            continue
                        #ignore this
                        # if player1.GetTurn():
                        #     player2.SetTurn(False)
                        #     player1.SetTurn(True)
                        # else:
                        #     player2.SetTurn(True)
                        #     player1.SetTurn(False)
                        #continue
                # board1.PrintBoard()
                # board1.DisplayBoard(player1, player2)
                pygame.display.update()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    current_time = pygame.time.get_ticks()
                    if current_time - last_click_time > 1000:
                        last_click_time = current_time
                        x_pos = event.pos[0]
                        current_column = (x_pos - ((info.current_w - 700)//2)) // SQUARE_SIZE #floor division to get whole number value of column, -600 to offset the 600 pixels i added to centre the board 
                        if not board1.IsValidMove(current_column): #doesn't allow user to drop piece outside the board
                            continue
                        if player1.GetTurn():
                            board1.DropPiece(current_column, player1.GetPiece())
                            mcts_obj.UpdateMove(current_column)
                            board1.Store(current_column)
                            player1.SetTurn(False)
                            player2.SetTurn(True)
