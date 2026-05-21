game.py :: server side

player.py :: client side


# game.py

Run game.py in terminal 1
Options are Create/Status/Quit

Typing Create + Enter - creates new game
Typing Status + Enter - shows status of games in current session

----
Relevant constants:
```
TURN_TIMEOUT_SEC = 10 
GAME_TIME_LIMIT_SEC = 1 * 60
```

TURN_TIMEOUT_SEC :: turn moves on to the next player if no move recieved within TURN_TIMEOUT_SEC seconds. This is currently set arbitrarily, may change for final version.

GAME_TIME_LIMIT_SEC :: Total allowed game time. This is currently set arbitrarily, may change for final version.

** Please let me know your average time taken per move and average total time to finish a game. Once I recieve responses from atleast 5 different teams, I will base the final values of these parameters accordingly. **


# player.py

Run player.py in terminal 2/3/4...
Once you create required number of players, press Enter in each terminal to start playing.

Current game logic is player requests server for list of valid moves, and chooses one move at random from the list. 
This is marked as 'PLAYING LOGIC' in the code.

[You'll need to replace the code inside 'PLAYING LOGIC' to your own logic.]

