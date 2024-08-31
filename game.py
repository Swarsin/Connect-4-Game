#Made changes to the classes & functions below:

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
