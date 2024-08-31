# Connect-4-Game

Through Harvard CS50's Artificial Intelligence with Python course and further research into AlphaGo, I learned about adversarial search, Monte Carlo Tree Search and Minimax algorithm and their applications in board games. 
Intrigued, I set out to make an AI player for Connect 4, using a minimax algorithm.

I have used an array of 7 stacks objects to represent the game board, which isn't ideal, and makes things a bit more complicated. 

Done:
- made a functioning menu system
- made a functional connect 4 game
- made a functional connect 4 ai bot using minimax algorithm with alpha beta pruning to allow for a more advanced bot that can see farther into the future (can branch up to a depth of 10, with some delays between moves at the start of the game)
- made a functional, but not great ai bot using MCTS

How to play:
- Download compressed files
- Extract files
- open the extracted folder in an IDE
- ensure the following python libraries are installed: random, sys, copy, math, time, sqlite3, pygame, pygame_menu
- run full_game.py
