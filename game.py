import random, sys, copy, math, time, sqlite3, pygame, pygame_menu

pygame.init() #initialise pygame

info = pygame.display.Info() #get display info - used to set the size of the window
screen = pygame.display.set_mode((info.current_w, info.current_h))
font = pygame.font.SysFont("arial", 75) #set font of text to be used in the game

played_ai = False #used to check if the ai game has been played
played_2p = False #used to check if the two player game has been played

SQUARE_SIZE = 100 #size of each square of the board which contain the pieces
RADIUS = int(SQUARE_SIZE/2 - 10) #radius of the circle in the middle of the square
BLUE = (0, 0, 255) #colour of the squares in the game board
BLACK = (0, 0, 0) #colour of the circles (denotes empty space in the game board)
EMPTY = 0 #empty space in the board

def merge_sort(array, wins=lambda x: x[1]): #used to sort the leaderboard
    if len(array) <= 1: #base case
        return array

    mid = len(array) // 2 #split the array in half
    left = array[:mid] #left half of the array
    right = array[mid:] #right half of the arra

    merge_sort(left, wins) #recursively sort the left half
    merge_sort(right, wins) #recursively sort the right half

    left_index = 0 #index for the left array 
    right_index = 0 #index for the right array
    array_index = 0 #index for the main array
    
    while left_index < len(left) and right_index < len(right): #while there are elements in the left and right arrays
        if wins(left[left_index]) > wins(right[right_index]): #if the current element in the left array is greater than the corresponding in the right array, add it to the main array
            array[array_index] = left[left_index] #add it to the main array
            left_index += 1 #increment the index of the left array
        else:
            array[array_index] = right[right_index] #else (bigger in right array) add it to the main array
            right_index += 1 #increment the index of the right array
        array_index += 1 #increment the index of the main array

    while left_index < len(left): #copies remaining element from the left array into the main array
        array[array_index] = left[left_index] #add it to the main array
        left_index += 1 #increment the index of the left array
        array_index += 1 #increment the index of the main array

    while right_index < len(right): #copies any remaining elements from the right array into the main array
        array[array_index] = right[right_index] #add it to the main array
        right_index += 1 #increment the index of the right array
        array_index += 1 #increment the index of the main array

    return array #returns the sorted array

def GetList(): #used to get the fields of information that will be be displayed in the leaderboard
    try:
        connect = sqlite3.connect("c4db.db")
        cursor = connect.cursor()

        cursor.execute("""
                    SELECT username, wins 
                    FROM users
                    """)

        rows = cursor.fetchall()
        list = [(row[0], row[1]) for row in rows]
        return list
    except Exception as error:
        print(error)
    finally:
        connect.close()

class Stack: #stack class for the game board
    def __init__(self): #initialise the stack
        self.items = [] #list of items in the stack
        self.pointer = 0 #pointer - points to the top of the stack

    def IsEmpty(self): #check if the stack is empty
        return self.pointer == 0
    
    def Push(self, item): #add item to the top of the stack
        if len(self.items) <= self.pointer:
            self.items += [0] * (self.pointer - len(self.items) + 1)
        self.items[self.pointer] = item
        self.pointer += 1

    def Pop(self): #remove item from the top of the stack
        if self.pointer < 0: 
            return None  # or raise an exception for an empty stack
        else:
            self.pointer -= 1  # decrement pointer after popping an item
            self.items = self.items[:self.pointer]
    
    def Fetch(self, index): #to get item at specific index of the stack object
        if 0 <= index < len(self.items):        
            return self.items[index] #return the item at the specified index
        else: #if the index is out of range
            return 0  #return 0
        
    def GetSize(self): #to get the size of the stack
        return len(self.items)
    
    # def __str__(self): #for testing/debugging
    #     return str(self.items)
    
    # def __repr__(self): #for testing/debugging
    #     return str(self.items)

class Board: #board class for the game - contains most of the functions used in this project
    def __init__(self):
        self.rows = 6 #number of rows in the board (typical game board is 6x7)
        self.columns = 7 #number of columns in the board (typical game board is 6x7)
        self.board = [Stack() for column in range(self.columns)] #uses composition - creating more complex objects (board) by combining simpler ones(stack) - also if the board class is scrapped, the stacks would be destroyed too as the form an integral part of the board. 
        self.window_length = 4 #length of the window in the evaluation function
        self.moves_history = [] #list of moves - used to undo moves
    
    def GetRows(self): #to get the number of rows
        return self.rows
    
    def GetColumns(self): #to get the number of columns
        return self.columns

    def DropPiece(self, column, player_piece): #to drop a piece in the board
        self.board[column].Push(player_piece)

    def IsValidMove(self, column): #to check if the move is valid
        if not  0 <= column <= 6: #if the column is not between 0 and 6
            return False
        if self.board[column].GetSize() == 6: #if the column is full
            return False
        return True #if none of the above, move is valid
    
    def GetValidMoves(self): #to get the valid moves
        valid_moves = [column for column in range(self.columns) if self.IsValidMove(column)] #create a list of valid moves
        return valid_moves #and returning it
    
    def SimulateMove(self, column, player1, player2): #to simulate a move (used later in MCTS)
        if self.IsValidMove(column):
            self.DropPiece(column, player1.GetPiece() if player1.GetTurn() else player2.GetPiece()) #Drop piece if column is valid
            if player1.GetTurn(): #Switch turn depending on whose turn it is currently
                player1.SetTurn(False)
                player2.SetTurn(True)
            else:
                player2.SetTurn(False)
                player1.SetTurn(True)

    def CheckForWin(self, player_piece): #Check for win
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
        score = 0 #initialising score to 0
        #evaluating player move:
        if player_piece == 1:
            opponent_piece = 2 #AI_PIECE
        if window.count(player_piece) == 4:
            score += 100 #if 4 in a row, add 100 to score
        elif window.count(player_piece) == 3 and window.count(EMPTY) == 1:
            score += 5 #if 3 in a row, add 5 to score
        elif window.count(player_piece) == 2 and window.count(EMPTY) == 2:
            score += 2 #if 2 in a row, add 2 to score
        return score
        
    def ScorePosition(self, player_piece):
        score = 0 #initialising score to 0

        #score centre column
        centre_array = [self.board[self.columns//2].Fetch(row) for row in range(self.rows)]
        centre_count = centre_array.count(player_piece)
        score += centre_count * 3 #for each piece of player_piece in centre column, add 3 to score

        #score horizontally
        for row in range(self.rows):
            row_array = [self.board[column].Fetch(row) for column in range(self.columns)] #creatint an array of the entire row
            for column in range(4):#(self.column - 3): #for each 4 pieces in the row
                window = row_array[column:column + self.window_length] #creating a window of 4 pieces
                score += self.EvaluateWindow(window, player_piece) #evaluating the window and adding it to score

        #score vertically
        for column in range(self.columns): #for each column
            column_array = [self.board[column].Fetch(row) for row in range(self.rows)] #creating an array of the entire column
            for row in range(self.rows - 3): #for each 4 pieces in the column
                window = column_array[row:row + self.window_length] #creating a window of 4 pieces
                score += self.EvaluateWindow(window, player_piece) #evaluating the window and adding it to score

        #score in a positively sloped diagonal
        for row in range(self.rows - 3):
            for column in range(self.columns - 3): #for each 4 pieces in the positive diagonal
                window = [self.board[column + i].Fetch(row + i) for i in range(self.window_length)] #creating a window of 4 pieces
                score += self.EvaluateWindow(window, player_piece) #evaluating the window and adding it to score

        #score in a negatively sloped diagonal
        for row in range(self.rows - 3): #for each 4 pieces in the negative diagonal
            for column in range(self.columns - 3):
                window = [self.board[column + 3 - i].Fetch(row + i) for i in range(self.window_length)] #creating a window of 4 pieces
                score += self.EvaluateWindow(window, player_piece) #evaluating the window and adding it to score

        return score
    
    def CheckForDraw(self):
        full = 0 #represents number of full columns found so far
        for column in range(self.columns): #checks each column to see if it is full
            if self.board[column].GetSize() == 6:
                full += 1 #if full column found, add 1 to full
        if full == 7: #if all columns are full, return True
            return True
        return False #otherwise, return False
        

    def PrintBoard(self): #displays on terminal - for original testing before GUI
        for row in range(self.rows - 1, -1, -1): #is self.rows - 1 because the row indices is 0 to 5 and this for loops runs from 0 to 5 (matches the row indices of the board)
            for column in range(self.columns):
                print(self.board[column].Fetch(row), end = " ") #Get the piece at the current row and column and print it
            print("") #new line
        print("") #new line

    def DisplayBoard(self, player1, player2):
        rect = pygame.Rect(0, 0, 100, 100) #Create a rectangle object - represents the square on the board which contains the piece
        for column in range(self.columns):
            for row in range(self.rows): #For each square on the board
                rect.center = ((info.current_w//2) - 300 + column*(SQUARE_SIZE), (info.current_h/2) - 250 + row*(SQUARE_SIZE)) #Set the centre of the square to correspoding to the current row and column so that the board is centred when finished
                pygame.draw.rect(screen, BLUE, rect) #Draw the square
                pygame.draw.circle(screen, BLACK, ((info.current_w//2) - 300 + column*(SQUARE_SIZE), (info.current_h/2) - 250 + row*(SQUARE_SIZE)), RADIUS) #draw a circle inn the centre of the square
        for column in range(self.columns):
            for row in range(self.rows): #For each square on the board        
                if self.board[column].Fetch(row) == 1: #If the square contains a 1 - meaning it belongs to player one, make the circle their colour and draw it in the appropriate location
                    pygame.draw.circle(screen, player1.GetColour(), ((info.current_w//2) - 300 + column*(SQUARE_SIZE), (info.current_h - ((info.current_h/2) - 250 + row*(SQUARE_SIZE)))), RADIUS) #I don't understand why +280 here, maybe the board isn't actually centred?
                elif self.board[column].Fetch(row) == 2: #If the square contains a 2 - meaning it belongs to player two, make the circle their colour and draw it in the appropriate location
                    pygame.draw.circle(screen, player2.GetColour(), ((info.current_w//2) - 300 + column*(SQUARE_SIZE), (info.current_h - ((info.current_h/2) - 250 + row*(SQUARE_SIZE)))), RADIUS) #I don't understand why +280 here, maybe the board isn't actually centred?
        pygame.display.update() #update the pygame window with the new board

    def is_terminal_node(self): #checks if the current board is a terminal node - meaning ganme is over
        return self.CheckForWin(1) or self.CheckForWin(2) or self.CheckForDraw()
    
    def minimax(self, board, depth, alpha, beta, max_player):
        valid_moves = board.GetValidMoves() # Get valid moves for the current board
        is_terminal = board.is_terminal_node()  # Check if the current board is a terminal node
        if depth == 0 or is_terminal:
            if is_terminal: # Check if the current board is a terminal node - either a win or a draw
                if board.CheckForWin(2):
                    return (None, 1000000000) # Return a large positive score if player 2 wins
                elif board.CheckForWin(1):
                    return (None, -1000000000) # Return a large negative score if player 1 wins
                else: #game over, so no more valid moves
                    return (None, 0) # Return 0 if the game is a draw
            else: #depth = zero
                return (None, board.ScorePosition(2)) # Return the score of the current board for player 2 (if large positive, then more beneificial for player 2, if large negative, then more beneficial for player 1)
        if max_player: #If current turn is maximising players'
            value = -math.inf # Set the initial value to the lowest possible score - as we're finding the best score for player 2, this should hopefully be changed later when a better score is found (move is beneficial to player 2)
            column = random.choice(valid_moves) # Choose a random column from the valid moves - just a placeholder
            for move in valid_moves: #for column in valid columns
                temp_board = copy.deepcopy(board) # Create a temporary board as a copy of the current board
                temp_board.DropPiece(move, 2) # Drop a piece for player 2 in the temporary board
                #new_score = max(value, minimax(temp_board, depth-1, False))
                new_score = board.minimax(temp_board, depth-1, alpha, beta, False)[1] # Recursively call minimax for the temporary board with depth-1 and the minimizing player's turn
                if new_score > value: # Update the value and column if a higher score is found
                    value = new_score
                    column = move
                alpha = max(alpha, value) # Update alpha with the maximum value
                if alpha >= beta: # If alpha is greater than or equal to beta, break the loop
                    break
            return column, value # Return the selected column and the value
        
        else: #for minimising player
            value = math.inf #set the initial value to the highest possible score - as we're finding the best score for player 1, this should hopefully be changed later when a better score (large negative) is found (move is beneficial to player 1)
            column = random.choice(valid_moves) #hoose a random column from the valid moves - just a placeholder
            for move in valid_moves: #for column in valid columns
                temp_board = copy.deepcopy(board) # Create a temporary board as a copy of the current board
                temp_board.DropPiece(move, 1) # Drop a piece for player 1 in the temporary board
                new_score = board.minimax(temp_board, depth-1, alpha, beta, True)[1] # Recursively call minimax for the temporary board with depth-1 and the maximizing player's turn
                if new_score < value: # Update the value and column if a lower score is found
                    value = new_score
                    column = move
                beta = min(beta, value) # Update beta with the minimum value
                if alpha >= beta:
                    break # If alpha is greater than or equal to beta, break the loop
            return column, value # Return the selected column and the value
    
    def Store(self, move): #stores the move that is passed in as a parameter in the moves_history list
        self.moves_history.append(move)
        
    def Undo(self): #undo the last 2 moves
        if len(self.moves_history) < 2:
            pass
        else:
            for _ in range(2):
                last_move = self.moves_history.pop(-1) #remove and return the last move from the moves_history list
                self.board[last_move].Pop() #remove the last move from the board
                # print(f"Length: {self.board[last_move].GetSize()}")
                # print(f"Length: {self.board[last_move+1].GetSize()}")
                # print(f"is valid: {self.IsValidMove(last_move)}")
                # self.PrintBoard()

class Player: #Class for player
    def __init__(self, piece, turn, colour, username):
        self.username = username #set username
        self.piece = piece #set piece
        self.turn = turn #set turn 
        self.colour = colour #set colour
    
    def GetUsername(self): #return username
        return self.username
    
    def GetPiece(self): #return piece
        return self.piece
    
    def GetTurn(self): #return turn
        return self.turn
    
    def SetTurn(self, new): #set turn parameter to the attribute
        self.turn = new

    def GetColour(self): #return colour
        return self.colour

class AIPlayer(Player): #Inherits from Player class 
    def __init__(self, piece, turn, colour, difficulty):
        super().__init__(piece, turn, colour, username="") #Inherit from Player class the attributes: piece, turn, colour
        self.difficulty = difficulty
    
    def GetUsername(self): #GetUsername method overriden
        return "AI Player"

    def GetPiece(self): #return piece
        return self.piece
    
    def GetTurn(self): #return turn
        return self.turn
    
    def SetTurn(self, new): #set turn parameter to the attribute
        self.turn = new

    def GetColour(self): #return colour
        return self.colour
    
    def GetDifficulty(self): #return difficulty
        return self.difficulty

class Node:
    def __init__(self, move, parent): #constructor
        self.move = move #what move this node represents
        self.parent = parent #parent of this node
        self.wins = 0 #total games won after this move
        self.total_games = 0 #total games played using this move
        self.children = [] #children of this node

    def SetChildren(self, children): #set children attribute to the children list that is passed in as parameter
        self.children = children

    def GetUCTValue(self): #Uses Upper Confidence Bound (UCT) formula for Trees to evaluate the value of a node
        explore = math.sqrt(2)
        if self.total_games == 0:
            if explore == 0:
                return 0
            else:
                return math.inf
        else:
            expolitation_term =  self.wins / self.total_games
            exploration_term = explore * math.sqrt(math.log(self.parent.total_games) / self.total_games)
            uct = expolitation_term + exploration_term
            return uct

class MCTS:
    def __init__(self, current_board, player1, player2): #constructor
        self.root_state = copy.deepcopy(current_board) #create a copy of the current board
        self.player1 = copy.deepcopy(player1) #Create copy of player1
        self.player2 = copy.deepcopy(player2) #Create copy of player2
        self.root = Node(None, None) #Create root node
        self.node_count = 0 #set node count to 0
        self.run_time = 0 #set run time to 0

    def Select(self):
        node, state = self.root, copy.deepcopy(self.root_state) #set node to root and state to copy of root state

        while len(node.children) != 0: #while children are still left/not currently at a leaf node
            children = node.children #set children to children of current node
            values = [child.GetUCTValue() for child in children] #create values list to store UCT values of children
            max_value = max(values) #select max value
            best_children = [children[i] for i in range(len(children)) if values[i] == max_value] #create best children list to store children with max value

            node = random.choice(best_children) #select random child from best children
            state.SimulateMove(node.move, self.player1, self.player2) #simulate move

            if node.total_games == 0: #if node has not been used in simulation
                return node, state #return node and state
            
        if self.Expand(node, state):  #If the current node is a terminal node, or no children are present, expand the tree
            node = random.choice(list(node.children)) # Randomly select one of the expanded children
            state.SimulateMove(node.move, self.player1, self.player2) # Simulate the selected move in the game state
        return node, state # Return the selected node and the updated game state

    
    def Expand(self, parent, state):
        if state.is_terminal_node(): #Check if state is terminal if it is then return false - means that game is over, so can't expand tree further
            return False
        else:
            children = [Node(move, parent) for move in state.GetValidMoves()] # Generate a list of child nodes based on valid moves in the current state
            parent.SetChildren(children) # add children list to the parent node's children attribute
            return True # Return True - indicates tree has been expanded

    def Simulate(self, state):
        while not state.is_terminal_node():
            state.SimulateMove(random.choice(state.GetValidMoves()), self.player1, self.player2)
        #print(f"Player Who Won: {player1.GetPiece() if state.CheckForWin(player1.GetPiece()) else player2.GetPiece() if state.CheckForWin(player2.GetPiece()) else None}")
        return (self.player1.GetPiece() if state.CheckForWin(self.player1.GetPiece()) else self.player2.GetPiece() if state.CheckForWin(self.player2.GetPiece()) else None)
    
    def Backpropagate(self, node, turn, outcome):
        if outcome == turn: #set score appropriately - depending on who won
            score = 0
        else:
            score = 1

        if node is not None: #if node is not root node
            node.total_games += 1 #increment total games
            node.wins += score #increment wins

            if outcome is None: #For DRAW
                score += 0
            else:
                score = 1 - score

            self.Backpropagate(node.parent, turn, outcome) #recursively call backpropagate

    def Search(self, time_limit):
        start_time = time.process_time() # set start time to limit the search within the specified time limit

        while time.process_time() - start_time < time_limit: #While there is still time left in the search time limit
            node, state = self.Select() #Select a node and the state it represents using Select
            outcome = self.Simulate(state) #Simulate the selected state
            self.Backpropagate(node, self.player1.GetPiece() if self.player1.GetTurn() else self.player2.GetPiece(), outcome) #Backpropagate the outcome to update node statistics

        #Calculate the total runtime of the search
        run_time = time.process_time() - start_time
        self.run_time = run_time

    def GetBestMove(self):
        if self.root_state.is_terminal_node(): #Check if the root state represents a terminal node
            return -1
        else:
            max_value = max(self.root.children, key=lambda x: x.total_games).total_games #Find the child node with the most games played
            max_nodes = [x for x in self.root.children if x.total_games == max_value]
            best_child = random.choice(max_nodes) #Randomly choose one of the best moves
            return best_child.move #and return it
    
    def UpdateMove(self, move):
        if move in self.root.children: #Check if the move is in the children of the root node
            self.root_state.SimulateMove(move, self.player1, self.player2) #Simulate the move in the root state
            self.root = self.root.children[move] #Update the root to the selected child
        
        self.root_state.SimulateMove(move, self.player1, self.player2) #If move is not in the children of the root node, simulate the move  
        self.root = Node(None, None) #and reset the root to a new node

def Main_2p(player1, player2, board1):
    winner = None #No winner at the start of the game
    game_over = False #Game not over yet
    board1.PrintBoard() #Print the initial board
    board1.DisplayBoard(player1, player2) #Display the initial board
    while not game_over: #While the game is not over
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.MOUSEMOTION: #If the user moves the cursor
                pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2)))
                x_pos = event.pos[0] #track the current horizontal position of the cursor
                if player1.GetTurn(): #and if it is player one's turn, display a piece on top the cursor with appropriate colour
                    pygame.draw.circle(screen, player1.GetColour(), (x_pos, (((info.current_h-600)/2)-RADIUS)), RADIUS)
                else: #and if it is player two's turn, display a piece on top the cursor with appropriate colour
                    pygame.draw.circle(screen, player2.GetColour(), (x_pos, (((info.current_h-600)/2)-RADIUS)), RADIUS)
            if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: #If the user pressses escape key
                        try: #try except block in case the user tries to undo when there is no move to undo
                            board1.Undo() #Call the undo method to undo the last 2 moves
                            board1.PrintBoard() #and print the new board
                            board1.DisplayBoard(player1, player2) #and display the new board
                        except Exception:
                            continue #The program will just ignore this exception and continue
            pygame.display.update() #update the pygame display
            pygame.event.set_allowed(pygame.MOUSEBUTTONDOWN) #allow mouse clicks (used to fix the bug where double clicking the mouse would cause the program to drop 2 pieces of the same colour one after each other, with a opponent piece in between)
            #print("Clicks allowed")
            #print(f"after allowed, mouse clicks blocked: {pygame.event.get_blocked(pygame.MOUSEBUTTONDOWN)}")
            if event.type == pygame.MOUSEBUTTONDOWN: #If the user clicks the mouse
                x_pos = event.pos[0] #Get the current horizontal position of the cursor
                current_column = (x_pos - ((info.current_w - 700)//2)) // SQUARE_SIZE #floor division to get whole number value of column, -600 to offset the 600 pixels i added to centre the board 
                if not board1.IsValidMove(current_column): #doesn't allow user to drop piece outside the board
                    continue
                if player1.GetTurn(): #If it is player one's turn
                    #print("Mouse Clicked")
                    pygame.event.set_blocked(pygame.MOUSEBUTTONDOWN) #block mouse clicks
                    #print(f"after mouse clicked, mouse clicks blocked: {pygame.event.get_blocked(pygame.MOUSEBUTTONDOWN)}")
                    board1.DropPiece(current_column, player1.GetPiece()) #Drop the piece
                    board1.Store(current_column) #store the move in moves_history
                    if board1.CheckForWin(player1.GetPiece()): #Check if player one has won
                        pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                        text = font.render(f"{player1.GetUsername()} won!", 1, player1.GetColour()) #Render the winning text
                        text_rect = text.get_rect(center=(info.current_w/2, 76//2)) #centres the text
                        screen.blit(text, text_rect) #display the winning text
                        game_over = True #The game is over
                        winner = player1.GetUsername() #Set the winner
                    elif board1.CheckForDraw(): #Check if the game is a draw
                        pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                        text = font.render("DRAW!", 1, (255, 255, 255)) #Render the draw text
                        text_rect = text.get_rect(center=(info.current_w/2, 76//2)) #centres the text
                        screen.blit(text, text_rect) #display the draw text
                        game_over = True #The game is over
                    player1.SetTurn(False) #Switch turns
                    player2.SetTurn(True)
                else:
                    #print("Mouse Clicked")
                    pygame.event.set_blocked(pygame.MOUSEBUTTONDOWN) #block mouse clicks (used to fix the bug where double clicking the mouse would cause the program to drop 2 pieces of the same colour one after each other, with a opponent piece in between)
                    #print(f"after mouse clicked, mouse clicks blocked: {pygame.event.get_blocked(pygame.MOUSEBUTTONDOWN)}")
                    board1.DropPiece(current_column, player2.GetPiece()) #Drop the piece
                    board1.Store(current_column) #store the move in moves_history
                    if board1.CheckForWin(player2.GetPiece()): #Check if player two has won
                        pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                        text = font.render(f"{player2.GetUsername()} won!", 1, player2.GetColour()) #Render the winning text
                        text_rect = text.get_rect(center=(info.current_w/2, 75//2)) #centres the text
                        screen.blit(text, text_rect) #display the winning text
                        game_over = True #The game is over
                        winner = player2.GetUsername() #Set the winner
                    elif board1.CheckForDraw(): #Check if the game is a draw
                        pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                        text = font.render("DRAW!", 1, (255, 255, 255)) #Render the draw text
                        text_rect = text.get_rect(center=(info.current_w/2, 75//2)) #centres the text
                        screen.blit(text, text_rect) #display the draw text
                        game_over = True #The game is over
                    player2.SetTurn(False) #Switch turns
                    player1.SetTurn(True)    
                board1.PrintBoard() #Print the board
                board1.DisplayBoard(player1, player2) #Display the board
                if game_over:
                    pygame.event.set_allowed(pygame.MOUSEBUTTONDOWN) #Allow the user to click the mouse again
                    pygame.time.wait(3000) #Wait for 3 seconds before exiting
                    return winner #and return the winner

def Main(player1, player2, board1):
    game_over = False #To check if the game is over
    board1.PrintBoard() #Print the board
    board1.DisplayBoard(player1, player2) #Display the board
    while not game_over: 
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT: #If the user clicks the quit button or presses the close button of the window
                    sys.exit() #Exit the program
                if event.type == pygame.MOUSEMOTION: #If the user moves the mouse
                    pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                    x_pos = event.pos[0] #Get the horizontal position of the mouse
                    if player1.GetTurn(): #If it is player one's turn
                        pygame.draw.circle(screen, player1.GetColour(), (x_pos, (((info.current_h-600)/2)-RADIUS)), RADIUS) #Draw a circle with the user's colour at the position of the mouse
                    else: #If it is player two's turn
                        pygame.draw.circle(screen, player2.GetColour(), (x_pos, (((info.current_h-600)/2)-RADIUS)), RADIUS) #Draw a circle with the user's colour at the position of the mouse
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: #If the user presses escape
                        try: #try except block in case the user tries to undo a move when there aren't any moves to undo
                            board1.Undo() #Undo the last 2 moves
                            board1.PrintBoard() #Print the board
                            board1.DisplayBoard(player1, player2) #Display the board
                        except Exception:
                            continue
                pygame.display.update() #Update the display
                pygame.event.set_allowed(pygame.MOUSEBUTTONDOWN) #allow mouse clicks (used to fix the bug where double clicking the mouse would cause the program to drop 2 pieces of the same colour one after each other, with a opponent piece in between)
                #print("Clicks allowed")
                #print(f"after allowed, mouse clicks blocked: {pygame.event.get_blocked(pygame.MOUSEBUTTONDOWN)}")
                if event.type == pygame.MOUSEBUTTONDOWN and not pygame.event.get_blocked(pygame.MOUSEBUTTONDOWN): #If the user clicks the mouse and it isn't blocked
                    #print("Mouse Clicked")
                    pygame.event.set_blocked(pygame.MOUSEBUTTONDOWN) #block mouse clicks
                    #print(f"after mouse clicked, mouse clicks blocked: {pygame.event.get_blocked(pygame.MOUSEBUTTONDOWN)}")
                    x_pos = event.pos[0] #Get the horizontal position of the mouse
                    current_column = (x_pos - ((info.current_w - 700)//2)) // SQUARE_SIZE #floor division to get whole number value of column, -600 to offset the 600 pixels i added to centre the board 
                    if not board1.IsValidMove(current_column): #doesn't allow user to drop piece outside the board
                        continue
                    if player1.GetTurn(): #If it is player one's turn
                        board1.DropPiece(current_column, player1.GetPiece()) #Drop the piece
                        board1.Store(current_column) #Store the move in moves_history
                        if board1.CheckForWin(player1.GetPiece()): #Check if player one won
                            pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                            text = font.render("Player One won!", 1, player1.GetColour()) #Render the winning text
                            text_rect = text.get_rect(center=(info.current_w/2, 76//2)) #Centre the text
                            screen.blit(text, text_rect) #Display the text
                            game_over = True #The game is over
                        elif board1.CheckForDraw(): #Check if the game is a draw
                            pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                            text = font.render("DRAW!", 1, (255, 255, 255)) #Render the draw text
                            text_rect = text.get_rect(center=(info.current_w/2, 76//2)) #Centre the text
                            screen.blit(text, text_rect) #Display the text
                            game_over = True #The game is over
                        player1.SetTurn(False) #Switch turns
                        player2.SetTurn(True)
                        board1.PrintBoard() #Print the board
                        board1.DisplayBoard(player1, player2) #Display the board

                if player2.GetTurn() and not game_over: #If it is player two's turn and the game isn't over
                    #print("AI Move")
                    #print(f"after ai move, mouse clicks blocked: {pygame.event.get_blocked(pygame.MOUSEBUTTONDOWN)}")
                    ai_move = board1.minimax(board1, player2.GetDifficulty(), -math.inf, math.inf, True)[0] #Get the AI's move by calling the minimax function
                    board1.DropPiece(ai_move, 2)#board1.DropPiece(random.randint(0, 6), 2) #Drop the piece
                    board1.Store(ai_move) #Store the move
                    if board1.CheckForWin(player2.GetPiece()): #Check if player two won
                        pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                        text = font.render("AI Player won!", 1, player2.GetColour()) #Render the winning text
                        text_rect = text.get_rect(center=(info.current_w/2, 75//2)) #Centre the text
                        screen.blit(text, text_rect) #Display the text
                        game_over = True #The game is over
                    elif board1.CheckForDraw(): #Check if the game is a draw
                        pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                        text = font.render("DRAW!", 1, (255, 255, 255)) #Render the draw text
                        text_rect = text.get_rect(center=(info.current_w/2, 75//2)) #Centre the text
                        screen.blit(text, text_rect) #Display the text
                        game_over = True  #The game is over
                    player2.SetTurn(False) #Switch turns
                    player1.SetTurn(True)
                    board1.PrintBoard() #Print the board
                    board1.DisplayBoard(player1, player2) #Display the board
                if game_over: #If the game is over, wait for 3 seconds before closing the game
                    pygame.event.set_allowed(pygame.MOUSEBUTTONDOWN) #allow mouse clicks
                    pygame.time.wait(3000)

def MainMCTS(player1, player2, board1, mcts_obj):
    game_over = False #Game not over yet
    board1.PrintBoard() #Print the initial board
    board1.DisplayBoard(player1, player2) #Display the initial board
    while not game_over: #While the game is not over
            for event in pygame.event.get(): 
                if event.type == pygame.QUIT: #If the user closes the window
                    sys.exit() #Exit the program
                if event.type == pygame.MOUSEMOTION: #If the user moves the cursor
                    pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                    x_pos = event.pos[0] #get the current horizontal position of the cursor
                    if player1.GetTurn(): #If it is player one's turn
                        pygame.draw.circle(screen, player1.GetColour(), (x_pos, (((info.current_h-600)/2)-RADIUS)), RADIUS) #Display a piece on top the cursor with appropriate colour
                    else: #If it is player two's turn
                        pygame.draw.circle(screen, player2.GetColour(), (x_pos, (((info.current_h-600)/2)-RADIUS)), RADIUS) #Display a piece on top the cursor with appropriate colour
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: #If the user pressses escape key
                        try: #try except block in case the user tries to undo when there is no move to undo
                            board1.Undo() #Call the undo method to undo the last 2 moves
                            board1.PrintBoard() #print the new board
                            board1.DisplayBoard(player1, player2) #and display the new board
                        except Exception:
                            continue
                pygame.display.update() #update the pygame display
                pygame.event.set_allowed(pygame.MOUSEBUTTONDOWN) #allow mouse clicks (used to fix the bug where double clicking the mouse would cause the program to drop 2 pieces of the same colour one after each other, with a opponent piece in between)
                #print("Clicks allowed")
                #print(f"after allowed, mouse clicks blocked: {pygame.event.get_blocked(pygame.MOUSEBUTTONDOWN)}")
                if event.type == pygame.MOUSEBUTTONDOWN and not pygame.event.get_blocked(pygame.MOUSEBUTTONDOWN): #If the user clicks the mouse and the mouse clicks are allowed
                    #print("Mouse Clicked")
                    pygame.event.set_blocked(pygame.MOUSEBUTTONDOWN) #block mouse clicks
                    #print(f"after mouse clicked, mouse clicks blocked: {pygame.event.get_blocked(pygame.MOUSEBUTTONDOWN)}")
                    x_pos = event.pos[0] #get the current horizontal position of the cursor
                    current_column = (x_pos - ((info.current_w - 700)//2)) // SQUARE_SIZE #floor division to get whole number value of column, -600 to offset the 600 pixels i added to centre the board 
                    if not board1.IsValidMove(current_column): #doesn't allow user to drop piece outside the board
                        continue
                    if player1.GetTurn():#If it is player one's turn
                        board1.DropPiece(current_column, player1.GetPiece()) #drop the piece
                        mcts_obj.UpdateMove(current_column) #update the MCTS tree
                        board1.Store(current_column) #store the move
                        if board1.CheckForWin(player1.GetPiece()): #Check if player one has won
                            pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                            text = font.render("Player One wins!", 1, player1.GetColour()) #Render the winning text
                            text_rect = text.get_rect(center=(info.current_w/2, 76//2)) #Centre the winning text
                            screen.blit(text, text_rect) #Display the winning text
                            game_over = True #Game is now over
                        elif board1.CheckForDraw(): #Check if the game is a draw
                            pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                            text = font.render("DRAW!", 1, (255, 255, 255)) #Render the draw text
                            text_rect = text.get_rect(center=(info.current_w/2, 76//2)) #Centre the draw text
                            screen.blit(text, text_rect) #Display the draw text
                            game_over = True #Game is now over
                        player1.SetTurn(False) #Switch turns
                        player2.SetTurn(True)
                        board1.PrintBoard() #Print the board
                        board1.DisplayBoard(player1, player2) #Display the board

                if player2.GetTurn() and not game_over: #If it is player two's turn
                    #print("AI Move")
                    #print(f"after ai move, mouse clicks blocked: {pygame.event.get_blocked(pygame.MOUSEBUTTONDOWN)}")
                    mcts_obj.Search(player2.GetDifficulty()) #Search the MCTS tree with appropriate time limit
                    ai_move = mcts_obj.GetBestMove() #Get the best move using MCTS
                    board1.DropPiece(ai_move, 2) #Drop the piece
                    mcts_obj.UpdateMove(ai_move) #Update the MCTS tree
                    board1.Store(ai_move) #Store the move in moves_history
                    if board1.CheckForWin(player2.GetPiece()): #Check if player two has won
                        pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                        text = font.render("AI Player wins!", 1, player2.GetColour()) #Render the winning text
                        text_rect = text.get_rect(center=(info.current_w/2, 75//2)) #Centre the winning text
                        screen.blit(text, text_rect) #Display the winning text
                        game_over = True #Game is now over
                    elif board1.CheckForDraw(): #Check if the game is a draw
                        pygame.draw.rect(screen, BLACK, (0, 0, info.current_w, ((info.current_h-600)/2))) #Draw a black rectangle to cover up the user's piece
                        text = font.render("DRAW!", 1, (255, 255, 255)) #Render the draw text
                        text_rect = text.get_rect(center=(info.current_w/2, 75//2)) #Centre the draw text
                        screen.blit(text, text_rect) #Display the draw text
                        game_over = True #Game is now over
                    player2.SetTurn(False) #Switch turns
                    player1.SetTurn(True)
                    board1.PrintBoard() #Print the board
                    board1.DisplayBoard(player1, player2) #Display the board
                if game_over: #If the game is over, allow mouse clicks and wait 3 seconds before exiting
                    pygame.event.set_allowed(pygame.MOUSEBUTTONDOWN) 
                    pygame.time.wait(3000)


#FROM HERE, CODE FOR MENU
def secure_password(password):
    min_length = 8 # Minimum length of the password
    uppercase = 0 # Number of uppercase letters in the password
    lowercase = 0 # Number of lowercase letters in the password
    digit = 0 # Number of digits in the password

    # Check each character in the password and increment the appropriate variable
    for char in password:
        if 'A' <= char <= 'Z':
            uppercase += 1
        elif 'a' <= char <= 'z':
            lowercase += 1
        elif '0' <= char <= '9':
            digit += 1

    # Check if the password meets the criteria:
    if len(password) >= min_length and (uppercase + lowercase + digit >= 8 and (uppercase >= 1 and lowercase >= 1 and digit >= 1)):
        return True #return True if it does
    else:
        return False #False if it doesn't
    
def add_user(username, password, user_id):
    try:
        conn = sqlite3.connect("c4db.db") #Connect to the database
        cursor = conn.cursor() #Create a cursor object
        # Check if user_id already exists
        cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
        user = cursor.fetchone()
        # If user_id already exists, find the next available location
        while user:
            user_id += 1
            cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
            user = cursor.fetchone()
        # Insert the new user
        with conn:
            cursor.execute("INSERT INTO users VALUES (?, ?, ?, 0)", (user_id, username, password))
    except Exception as error: #catch any errors
        print(error)
    finally:
        conn.close() #Close the connection


def search_user(user_id, username):
    try:
        conn = sqlite3.connect("c4db.db") #Connect to the database
        cursor = conn.cursor() #Create a cursor object
        # Check if user_id already exists
        cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
        user = cursor.fetchone()
        # If user_id already exists, find the next available location
        while user[1] != username:
            user_id += 1
            cursor.execute("SELECT * FROM users WHERE id=?", (user_id,))
            user = cursor.fetchone()
        return user
    except Exception as error:
        print(error)
    finally:
        conn.close()


def update_score(username): #update the score of a user in the database using their username
    try:
        conn = sqlite3.connect("c4db.db")
        cursor = conn.cursor()
        with conn:
            cursor.execute("UPDATE users SET wins = wins + 1 WHERE username = ?", (username,))
    except Exception as error: #catch any errors
        print(error)
    finally:
        conn.close()

def go_to_leaderboard(): #open leaderboard menu
    # code for play with bot
    main_menu._open(leaderboard)

def quit_game(): #quit the game
    # code for back
    print("Quitting...")
    sys.exit()

def open_options_menu(): #open options menu
    main_menu._open(options_menu)

def play_with_friend(): #open login menu
    main_menu._open(login_menu)

def play_with_bot(): #open customise menu
    main_menu._open(customise_menu)

def hash_function(input): #hash function for username, which returns the modulus of the ascii value of the string - representing the id
    value = 0
    [value := value + ord(i) for i in input] #calculates ascii value of a string input
    return value % 200

def login():
    user = username_input.get_value() #get username from the username input widget in the login menu
    pwd = password_input.get_value() #get password from the password input widget in the login menu
    id = hash_function(user) #hash the username
    result = search_user(id, user) #search for the user in the database

    user_not_found = "User not found!" #message to display if user is not found
    wrong_combo = "Incorrect username or password!" #message to display if username or password is incorrect
    widget_titles = [widget.get_title() for widget in login_menu.get_widgets()] #get the titles of the widgets in the login menu

    if result == None: #if user is not found
        if user_not_found not in widget_titles: #add the message to the login menu if it is not already there
            login_menu.add.label(user_not_found)
    else:
        #CHECK PASSWORD AND IF CORRECT GO TO NEXT SCREEN
        if pwd == result[2]:
            for widget in login_menu.get_widgets(): #remove the error messages from the login menu if they are there
                if widget.get_title() == user_not_found or widget.get_title() == wrong_combo:
                    login_menu.remove_widget(widget)
            main_menu._open(login_menu_p2) #go to next login screen
            return (user, pwd) #and return the username and password
        else:
            if wrong_combo not in widget_titles: #add the wrong password message to the login menu if it is not already there
                login_menu.add.label(wrong_combo)

def login_p2():
    user = username_input_p2.get_value() #get username from the username input widget in the second login menu
    pwd = password_input_p2.get_value() #get password from the password input widget in the second login menu
    id = hash_function(user) #hash the username of the second player
    result = search_user(id, user) #search for the user in the database
    previous_user_details = login() #get the details of the previous player

    double_login = "User already logged in!" #message to display if user is already logged in
    user_not_found = "User not found!" #message to display if user is not found
    wrong_combo = "Incorrect username or password!" #message to display if username or password is incorrect
    widget_titles = [widget.get_title() for widget in login_menu_p2.get_widgets()] #get the titles of the widgets in the second login menu

    if previous_user_details[0] == user: #if user is already logged in
        if double_login not in widget_titles: #add the message to the second login menu if it is not already there
            login_menu_p2.add.label(double_login)
    elif result == None: #if user is not found
        if user_not_found not in widget_titles: #add the message to the second login menu if it is not already there
            login_menu_p2.add.label(user_not_found)
    else:
        #CHECK PASSWORD AND IF CORRECT GO TO NEXT SCREEN
        if pwd == result[2]:
            for widget in login_menu_p2.get_widgets(): #remove the error messages from the second login menu if they are there
                if widget.get_title() == user_not_found or widget.get_title() == wrong_combo or widget.get_title() == double_login:
                    login_menu_p2.remove_widget(widget)
            main_menu._open(customise_2p) #go to next login screen
            return (user, pwd) #and return the username and password
        else:
            if wrong_combo not in widget_titles: #add the wrong password message to the second login menu if it is not already there
                login_menu_p2.add.label(wrong_combo)

def register(): 
    user = username_input.get_value() #get username from the username input widget in the login menu
    pwd = password_input.get_value() #get password from the password input widget in the login menu
    id = hash_function(user) #hash the username
    result = search_user(id, user) #search for the user in the 
    
    registered_successfully = "User successfully registered!" #message to display if user is registered successfully
    already_registered = "User has already been registered, please log-in instead!" #message to display if user is already registered
    weak_password = "Password must be at least 8 characters, and have number, capital, and lowercase letters!" #message to display if password is weak
    widget_titles = [widget.get_title() for widget in login_menu.get_widgets()] #get the titles of the widgets in the login menu

    if result != None: #if user is already registered
        if already_registered not in widget_titles: #add the message to the login menu if it is not already there
            login_menu.add.label(already_registered)
    elif secure_password(pwd):#if password is secure
        if registered_successfully not in widget_titles:
            add_user(user, pwd, id) #add the user to the database
            login_menu.add.label(registered_successfully)#add the registered successfully message to the login menu
    else:
        if weak_password not in widget_titles: #add the weak password message to the login menu if it is not already there
            login_menu.add.label(weak_password)

def register_p2():
    user = username_input_p2.get_value() #get username from the username input widget in the second login menu
    pwd = password_input_p2.get_value() #get password from the password input widget in the second login menu
    id = hash_function(user) #hash the username
    result = search_user(id, user) #search for the user in the database
    
    registered_successfully = "User successfully registered!" #message to display if user is registered successfully
    already_registered = "User has already been registered, please log-in instead!" #message to display if user is already registered
    weak_password = "Password must be at least 8 characters, and have number, capital, and lowercase letters!" #message to display if password is weak
    widget_titles = [widget.get_title() for widget in login_menu_p2.get_widgets()] #get the titles of the widgets in the second login menu

    if result != None: #if user is already registered
        if already_registered not in widget_titles:#add the message to the second login menu if it is not already there
            login_menu_p2.add.label(already_registered)
    elif secure_password(pwd): #if password is secure
        if registered_successfully not in widget_titles: #add the registered successfully message to the second login menu if its not already there
            add_user(user, pwd, id) #add the user to the database
            login_menu_p2.add.label(registered_successfully)
    else:
        if weak_password not in widget_titles: #add the weak password message to the second login menu if it is not already there
            login_menu_p2.add.label(weak_password)

# Function to be called when the user clicks on the "Play" button
def start_ai_game():
    global played_ai  # global variable to track if the user has played an AI game

    missing_options = "You must choose an option from all dropdown lists!" # message to display if the user does not select an option
    same_colour = "Both players can't choose the same colour!" # message to display if both players choose the same colour
    widget_titles = [widget.get_title() for widget in customise_menu.get_widgets()] # get the titles of the widgets in the customise menu

    try:
        selected_value1 = drop_down1.get_value()  # get_value() returns the user choice as a tuple (since that's how it's defined in the code) with the index of the selected choice in the corresponding choices array
        selected_value2 = drop_down2.get_value()  # example: (('Hard', 4), 3), (('Green', (0, 255, 0)), 1), (('Blue', (0, 0, 255)), 2), (('Player 1', 1), 0)
        selected_value3 = drop_down3.get_value() #get the selected values from the dropdown menus
        selected_value4 = drop_down4.get_value()
        if selected_value2 == selected_value3: #check if both players choose the same colour
            raise NameError #raise an error if both players choose the same colour
        played_ai = False #set played_ai to False
        board = Board() #create a new board obhject
        human_player = Player(1, True if selected_value4[0][1] == 1 else False, selected_value2[0][1], "You") #create a new player object
        ai_player = AIPlayer(2, False if selected_value4[0][1] == 1 else True, selected_value3[0][1], selected_value1[0][1]) #create a new AI player object
        for widget in customise_menu.get_widgets(): #remove the error messages from the customise menu if they are there
            if widget.get_title() == missing_options or widget.get_title() == same_colour:
                customise_menu.remove_widget(widget)
        main_menu.disable() #disable the main menu
        screen.fill((0, 0, 0)) #fill the screen with black
        if selected_value1[1] == 0: #if the user chooses MCTS
            mcts = MCTS(board, human_player, ai_player) #create a new MCTS object
            MainMCTS(human_player, ai_player, board, mcts) #start the MCTS game
        else:
            Main(human_player, ai_player, board) #start the normal minimax game
        main_menu.enable() #enable the main menu when done with the game
    
    except ValueError: #if the user does not select an option
        if missing_options not in widget_titles: #add the missing options message to the customise menu if it is not already there
            customise_menu.add.label(missing_options)
    
    except NameError: #if both players choose the same colour
        if same_colour not in widget_titles: #add the same colour message to the customise menu if it is not already there
            customise_menu.add.label(same_colour)

def start_2p_game():
    global played_2p

    missing_options = "You must choose an option from all dropdown lists!"
    same_colour = "Both players can't choose the same colour!"
    widget_titles = [widget.get_title() for widget in customise_2p.get_widgets()]

    try:
        selected_value1 = dropdown1_2p.get_value()  # get_value() returns the user choice as a tuple (since that's how it's defined in the code) with the index of the selected choice in the corresponding choices array
        selected_value2 = dropdown2_2p.get_value()  # example: (('Hard', 4), 3), (('Green', (0, 255, 0)), 1), (('Blue', (0, 0, 255)), 2), (('Player 1', 1), 0)
        selected_value3 = dropdown3_2p.get_value() #get the selected values from the dropdown menus
        if selected_value1 == selected_value2: #check if both players choose the same colour
            raise NameError #raise an error if both players choose the same colour
        played_2p = False 
        p1_details = login()  #get the username and password of player 1
        p2_details = login_p2() #get the username and password of player 2
        board = Board() #create a new board object
        player_one = Player(1, True if selected_value3[0][1] == 1 else False, selected_value1[0][1], p1_details[0]) #create a new player 1 object
        player_two = Player(2, False if selected_value3[0][1] == 1 else True, selected_value2[0][1], p2_details[0]) #create a new player 2 object
        for widget in customise_2p.get_widgets(): #remove the error messages from the customise menu if they are there
            if widget.get_title() == missing_options or widget.get_title() == same_colour:
                customise_2p.remove_widget(widget)
        main_menu.disable() #disable the main menu
        screen.fill((0, 0, 0)) #fill the screen with black
        winner = Main_2p(player_one, player_two, board) #start the game
        if winner != None: #if the game is over and not a draw
            update_score(winner) #update the score of the winner
        main_menu.enable() #enable the main menu when done with the game
    except ValueError: #if the user does not select an option(s)
        if missing_options not in widget_titles: #add the missing options message to the customise menu if it is not already there
            missing_options_widget = customise_2p.add.label(missing_options)
    except NameError:
        if same_colour not in widget_titles: #add the same colour message to the customise menu if it is not already there
            same_colour_widget = customise_2p.add.label(same_colour)

def remove_all_widgets(menu): #remove all widgets from a menu
    for widget in menu.get_widgets():
        menu.remove_widget(widget)

def update_leaderboard(): #update the leaderboard - called when the user clicks on the "update" button
    remove_all_widgets(leaderboard)
    # main_menu._open(leaderboard)
    leaderboard.add.button("Update", update_leaderboard)
    i = 0
    for user in merge_sort(GetList(), wins=lambda x: x[1]):
        i += 1
        leaderboard.add.label(str(i) + ". " + str(user[0]) + "\t\t\t\t\t\twins: " + str(user[1]))

# create main menu
main_menu = pygame_menu.Menu("Connect Four Game", info.current_w, info.current_h, theme=pygame_menu.themes.THEME_SOLARIZED)

main_menu.add.button("Play game", open_options_menu) #adding buttons to main menu
main_menu.add.button("Leaderboard", go_to_leaderboard)
main_menu.add.button("Quit", quit_game)

# create game options menu
options_menu = pygame_menu.Menu("Connect Four Game Options", info.current_w, info.current_h, theme=pygame_menu.themes.THEME_SOLARIZED)

options_menu.add.button("Login & Play with Friend", play_with_friend) #adding buttons to options menu
options_menu.add.button("Play with Bot (No login)", play_with_bot)

# Create leaderboard screen
leaderboard = pygame_menu.Menu("Leaderboard", info.current_w, info.current_h, theme=pygame_menu.themes.THEME_SOLARIZED)

leaderboard.add.button("Update", update_leaderboard) #adding update button to leaderboard
update_leaderboard()

# Create login menu
login_menu = pygame_menu.Menu("Login for Player 1", info.current_w, info.current_h, theme=pygame_menu.themes.THEME_SOLARIZED)
username_input = login_menu.add.text_input("Username: ", maxchar=20)  # Add text input for username
password_input = login_menu.add.text_input("Password: ", maxchar=20, password=True)  # Add password input
login_menu.add.button("Login", login)  # add login button
login_menu.add.button("Register", register)  # add register button


login_menu_p2 = pygame_menu.Menu("Login for Player 2", info.current_w, info.current_h, theme=pygame_menu.themes.THEME_SOLARIZED) #create login menu
username_input_p2 = login_menu_p2.add.text_input("Username: ", maxchar=20)  # Add text input for username
password_input_p2 = login_menu_p2.add.text_input("Password: ", maxchar=20, password=True)  # Add password input
login_menu_p2.add.button("Login", login_p2)  # add login button
login_menu_p2.add.button("Register", register_p2)  # add register button
login_menu_p2.add.label("Player 1 logged in successfully!")


# Create a Pygame Menu
customise_menu = pygame_menu.Menu("Customise Game", info.current_w, info.current_h, theme=pygame_menu.themes.THEME_SOLARIZED)

# Add four drop-down lists
drop_down1_choices = [
    ("Easiest (MCTS)", 5),
    ("Easier (Minimax)", 1),
    ("Easy (Minimax)", 2),
    ("Medium (Minimax)", 4),
    ("Hard (Minimax)", 6),
    ("Hardest (Minimax)", 7),
]

drop_down1 = customise_menu.add.dropselect("Difficulty: ", drop_down1_choices, onchange=None)

drop_down2_choices = [
    ("Red", (255, 0, 0)),
    ("Green", (0, 255, 0)),
    ("Yellow", (255, 255, 0)),
    ("Purple", (160, 32, 240)),
]

drop_down2 = customise_menu.add.dropselect("1st Player Colour: ", drop_down2_choices, onchange=None)

drop_down3_choices = [
    ("Red", (255, 0, 0)),
    ("Green", (0, 255, 0)),
    ("Yellow", (255, 255, 0)),
    ("Purple", (160, 32, 240)),
]

drop_down3 = customise_menu.add.dropselect("2nd Player Colour: ", drop_down3_choices, onchange=None) #add a drop down menus to the customise menu

drop_down4_choices = [("Player 1", 1), ("Player 2", 2), ("Random", int(random.randint(1,2)))]

drop_down4 = customise_menu.add.dropselect("First Move: ", drop_down4_choices, onchange=None)

# Add a "Play" button to trigger the start_ai_game function
play = customise_menu.add.button("Play", start_ai_game)

customise_2p = pygame_menu.Menu("2P Customise", info.current_w, info.current_h, theme=pygame_menu.themes.THEME_SOLARIZED) #add a 2 player mode customisation menu

customise_2p.add.label("Both players logged in successfully!") #Show that both players logged in successfully

dropdown1_2p = customise_2p.add.dropselect("1st Player Colour: ", drop_down2_choices, onchange=None) #Add a drop down menu to the 2 player mode customise menu

dropdown2_2p = customise_2p.add.dropselect("2nd Player Colour: ", drop_down3_choices, onchange=None)

dropdown3_2p = customise_2p.add.dropselect("First Move: ", drop_down4_choices, onchange=None)

play_2p = customise_2p.add.button("Play", start_2p_game) #Add a button to start the game

while True: 
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT: #If the user clicks the quit button or presses the close button of the window then exit
            pygame.quit()
            sys.exit()
    if main_menu.is_enabled(): #If the main menu is enabled
        main_menu.update(events) #Update the main menu
        main_menu.draw(screen) #Draw menu to the screen
    pygame.display.update() #and update display
