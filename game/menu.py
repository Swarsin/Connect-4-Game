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
