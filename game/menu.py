import pygame, pygame_menu, sys, sqlite3
from leaderboard import *
from createtable import *
from ai_mode_adjusted import *
from connect4_mcts_test import *
from main import Main_2p

pygame.init()

info = pygame.display.Info()
screen = pygame.display.set_mode((info.current_w, info.current_h))

played_ai = False
played_2p = False

def secure_password(password):
    min_length = 8
    has_uppercase = False
    has_lowercase = False
    has_digit = False

    # Check each character in the password
    for char in password:
        if 'A' <= char <= 'Z':
            has_uppercase = True
        elif 'a' <= char <= 'z':
            has_lowercase = True
        elif '0' <= char <= '9':
            has_digit = True

    # Check if the password meets the criteria
    if len(password) >= min_length and has_uppercase and has_lowercase and has_digit:
        return True
    else:
        return False

def play_game():
    # code for play with friend
    print("Playing game...")

def search_user(user_id):
    try:
        conn = sqlite3.connect("c4db.db")
        cursor = conn.cursor()
        with conn:
            cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
            result = cursor.fetchall()
        return result
    except Exception as error:
        print(error)
    finally:
        conn.close()

def add_user(username, password, user_id):
    try:
        conn = sqlite3.connect("c4db.db")
        cursor = conn.cursor()
        with conn:
            cursor.execute("INSERT INTO users VALUES (?, ?, ?, 0)", (user_id, username, password))
    except Exception as error:
        print(error)
    finally:
        conn.close()

def update_score(username):
    try:
        conn = sqlite3.connect("c4db.db")
        cursor = conn.cursor()
        with conn:
            cursor.execute("UPDATE users SET wins = wins + 1 WHERE username = ?", (username,))
    except Exception as error:
        print(error)
    finally:
        conn.close()

def go_to_leaderboard():
    # code for opening leaderboard
    main_menu._open(leaderboard)

def quit_game():
    # code for back
    print("Quitting...")
    exit()

def open_options_menu():
    main_menu._open(options_menu)

def play_with_friend():
    # code for play with friend
    print("Playing with friend...")
    main_menu._open(login_menu)

def play_with_bot():
    # code for play with bot
    print("Playing with bot...")
    main_menu._open(customise_menu)

def hash_function(input):
    value = 0
    [value := value + ord(i) for i in input] #calculates ascii value of a string input
    return value % 200


def login():
    user = username_input.get_value()
    pwd = password_input.get_value()
    id = hash_function(user)
    result = search_user(id)

    user_not_found = "User not found!"
    wrong_combo = "Incorrect username or password!"
    widget_titles = [widget.get_title() for widget in login_menu.get_widgets()]

    if result == []:
        if user_not_found not in widget_titles:
            login_menu.add.label(user_not_found)
    else:
        #CHECK PASSWORD AND IF CORRECT GO TO NEXT SCREEN
        if pwd == result[0][2]:
            #go to next screen
            #print("Go to next screen")
            main_menu._open(login_menu_p2)
            return (user, pwd)
        else:
            if wrong_combo not in widget_titles:
                login_menu.add.label(wrong_combo)

def login_p2():
    user = username_input_p2.get_value()
    pwd = password_input_p2.get_value()
    id = hash_function(user)
    result = search_user(id)
    previous_user_details = login()

    double_login = "User already logged in!"
    user_not_found = "User not found!"
    wrong_combo = "Incorrect username or password!"
    widget_titles = [widget.get_title() for widget in login_menu_p2.get_widgets()]

    if previous_user_details[0] == user:
        if double_login not in widget_titles:
            login_menu_p2.add.label(double_login)
    elif result == []:
        if user_not_found not in widget_titles:
            login_menu_p2.add.label(user_not_found)
    else:
        #CHECK PASSWORD AND IF CORRECT GO TO NEXT SCREEN
        if pwd == result[0][2]:
            #go to next screen
            #print("Go to next screen")
            main_menu._open(customise_2p)
            return (user, pwd)
        else:
            if wrong_combo not in widget_titles:
                login_menu_p2.add.label(wrong_combo)
def register():
    user = username_input.get_value()
    pwd = password_input.get_value()
    id = hash_function(user)
    result = search_user(id)
    
    registered_successfully = "User successfully registered!"
    already_registered = "User has already been registered, please log-in instead!"
    weak_password = "Password must have number, capitals, and lowercase letters!"
    widget_titles = [widget.get_title() for widget in login_menu.get_widgets()]

    if result != []:
        if already_registered not in widget_titles:#not already_registered:
            login_menu.add.label(already_registered)
    elif secure_password(pwd):
        if registered_successfully not in widget_titles:
            add_user(user, pwd, id)
            login_menu.add.label(registered_successfully)
    else:
        if weak_password not in widget_titles:
            login_menu.add.label(weak_password)

def register_p2():
    user = username_input_p2.get_value()
    pwd = password_input_p2.get_value()
    id = hash_function(user)
    result = search_user(id)
    
    registered_successfully = "User successfully registered!"
    already_registered = "User has already been registered, please log-in instead!"
    weak_password = "Password must have number, capitals, and lowercase letters!"
    widget_titles = [widget.get_title() for widget in login_menu_p2.get_widgets()]

    if result != []:
        if already_registered not in widget_titles:#not already_registered:
            login_menu_p2.add.label(already_registered)
    elif secure_password(pwd):
        if registered_successfully not in widget_titles:
            add_user(user, pwd, id)
            login_menu_p2.add.label(registered_successfully)
    else:
        if weak_password not in widget_titles:
            login_menu_p2.add.label(weak_password)

# Function to be called when the user clicks on the "Play" button
def start_ai_game():
    global played_ai

    missing_options = "You must choose an option from all dropdown lists!"
    same_colour = "Both players can't choose the same colour!"
    widget_titles = [widget.get_title() for widget in customise_menu.get_widgets()]

    try:
        selected_value1 = drop_down1.get_value()  # get_value() returns the user choice as a tuple (since that's how it's defined in the code) with the index of the selected choice in the corresponding choices array
        print(selected_value1)
        selected_value2 = drop_down2.get_value()  # example: (('Hard', 4), 3), (('Green', (0, 255, 0)), 1), (('Blue', (0, 0, 255)), 2), (('Player 1', 1), 0)
        selected_value3 = drop_down3.get_value()
        selected_value4 = drop_down4.get_value()
        if selected_value2 == selected_value3:
            raise NameError
        played_ai = False
        board = Board()
        human_player = Player(1, True if selected_value4[0][1] == 1 else False, selected_value2[0][1], "You")
        ai_player = AIPlayer(2, False if selected_value4[0][1] == 1 else True, selected_value3[0][1], selected_value1[0][1])
        main_menu.disable()
        main_menu.full_reset()
        screen.fill((0, 0, 0))
        if selected_value1[1] == 0:
            mcts = MCTS(board, human_player, ai_player)
            MainMCTS(human_player, ai_player, board, mcts)
        else:
            Main(human_player, ai_player, board)
        main_menu.enable()
    except ValueError:
        if missing_options not in widget_titles:
            customise_menu.add.label(missing_options)
    except NameError:
        if same_colour not in widget_titles:
            customise_menu.add.label(same_colour)

def start_2p_game():
    global played_2p

    missing_options = "You must choose an option from all dropdown lists!"
    same_colour = "Both players can't choose the same colour!"
    widget_titles = [widget.get_title() for widget in customise_2p.get_widgets()]

    try:
        selected_value1 = dropdown1_2p.get_value()  # get_value() returns the user choice as a tuple (since that's how it's defined in the code) with the index of the selected choice in the corresponding choices array
        selected_value2 = dropdown2_2p.get_value()  # example: (('Hard', 4), 3), (('Green', (0, 255, 0)), 1), (('Blue', (0, 0, 255)), 2), (('Player 1', 1), 0)
        selected_value3 = dropdown3_2p.get_value()
        if selected_value1 == selected_value2:
            raise NameError
        played_2p = False
        p1_details = login()
        p2_details = login_p2()
        board = Board()
        player_one = Player(1, True if selected_value3[0][1] == 1 else False, selected_value1[0][1], p1_details[0])
        player_two = Player(2, False if selected_value3[0][1] == 1 else True, selected_value2[0][1], p2_details[0])
        main_menu.disable()
        #main_menu.full_reset() 
        screen.fill((0, 0, 0))
        winner = Main_2p(player_one, player_two, board)
        if winner != None:
            update_score(winner)
        main_menu.enable()
    except ValueError:
        if missing_options not in widget_titles:
            customise_2p.add.label(missing_options)
    except NameError:
        if same_colour not in widget_titles:
            customise_2p.add.label(same_colour)

def remove_all_widgets(menu):
    for widget in menu.get_widgets():
        menu.remove_widget(widget)

def update_leaderboard():
    remove_all_widgets(leaderboard)
    # main_menu._open(leaderboard)
    leaderboard.add.button("Update", update_leaderboard)
    i = 0
    for user in merge_sort(GetList(), wins=lambda x: x[1]):
        i += 1
        leaderboard.add.label(str(i) + ". " + str(user[0]) + "\t\t\t\t\t\twins: " + str(user[1]))
