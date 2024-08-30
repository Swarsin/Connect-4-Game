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

