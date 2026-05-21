# ==========================================================
# game.py — SERVER
# ==========================================================

import os
import json
import uuid
import time
import socket
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
import math
import random
# Game logic imports
from checkers_board import HexBoard
from checkers_pins import Pin
import pandas as pd


round_number = 11  # Update this for each new round of the tournament to track games and players in the round data file
# ==========================================================
# Utilities
# ==========================================================
def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_path(game_id: str) -> str:
    os.makedirs(f"games_round_{round_number}", exist_ok=True)
    return os.path.join(f"games_round_{round_number}", f"game_{game_id}.log")


def write_log(game_id: str, msg: str):
    with open(log_path(game_id), "a", encoding="utf-8") as f:
        f.write(f"[{ts()}] {msg}\n")


def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj)
    except Exception as e:
        return json.dumps({"ok": False, "error": f"json-encode-failed: {e}"})


# ==========================================================
# Game Constants
# ==========================================================
COLOUR_ORDER = ['red', 'lawn green', 'yellow', 'blue', 'gray0', 'purple']
PRIMARY_COLOURS = ['red', 'lawn green', 'yellow']
#random.shuffle(PRIMARY_COLOURS)
COMPLEMENT = {'red': 'blue', 'lawn green': 'gray0', 'yellow': 'purple'}
MAX_PLAYERS = 6
TURN_TIMEOUT_SEC = 2
GAME_TIME_LIMIT_SEC = 60


# ==========================================================
# Player Class
# ==========================================================
class Player:
    def __init__(self, pid: str, name: str, colour: str):
        self.player_id = pid
        self.name = name
        self.colour = colour
        self.ready = False
        self.status = "PLAYING"

        # Scoring stats
        self.move_count = 0
        self.wrong_moves = {}
        self.time_taken_sec = 0.0


# ==========================================================
# Game Class
# ==========================================================
class Game:
    def __init__(self):
        self.game_id = str(uuid.uuid4())
        self.board = HexBoard()
        self.players: List[Player] = []
        self.pins_by_colour: Dict[str, List[Pin]] = {}
        self.status = "AVAILABLE"
        self.created_ts = ts()
        self.joined_primary_index = 0
        self.lock_joining = False
        self.primary_colours=random.sample(PRIMARY_COLOURS, len(PRIMARY_COLOURS))  # Randomize primary colour assignment order for each game

        # Timing
        self.total_start_ns = None
        self.turn_started_ns = None

        # Turn rotation
        self.turn_order: List[str] = []
        self.current_turn_index = 0

        # Moves
        self.move_count = 0
        self.move_times_ms: List[float] = []
        self.last_move = None
        self.turn_timeout_notice = None
        self.wrong_moves_by_colour: Dict[str, List[str]] = {}

        # Score dictionary
        self.scores: Dict[str, Dict[str, float]] = {}

    # ----------------------------------
    # Assign Colour
    # ----------------------------------
    def assign_colour(self) -> Optional[str]:
        n = len(self.players) + 1
        if n > MAX_PLAYERS:
            return None
        if n % 2 == 1:  # primary
            if self.joined_primary_index >= len(PRIMARY_COLOURS):
                return None
            return self.primary_colours[self.joined_primary_index]
        else:
            primary = self.primary_colours[self.joined_primary_index]
            self.joined_primary_index += 1
            return COMPLEMENT[primary]

    # ----------------------------------
    def init_pins(self, colour: str):
        if colour in self.pins_by_colour:
            return
        idxs = self.board.axial_of_colour(colour)[:10]
        self.pins_by_colour[colour] = [
            Pin(self.board, idxs[i], id=i, color=colour)
            for i in range(len(idxs))
        ]

    # ----------------------------------
    def compute_turn_order(self):
        present = [p.colour for p in self.players]
        first = present[0]
        if first in COLOUR_ORDER:
            idx = COLOUR_ORDER.index(first)
            rotated = COLOUR_ORDER[idx:] + COLOUR_ORDER[:idx]
        else:
            rotated = COLOUR_ORDER[:]
        self.turn_order = [c for c in rotated if c in present]
        self.current_turn_index = 0

    # ----------------------------------
    def current_turn_colour(self):
        if self.status != "PLAYING":
            return None
        if not self.turn_order:
            return None
        return self.turn_order[self.current_turn_index]

    # ----------------------------------
    def advance_turn(self):
        if self.turn_order:
            self.current_turn_index = (self.current_turn_index + 1) % len(self.turn_order)
            self.turn_started_ns = time.perf_counter_ns()

    # ----------------------------------
    def ensure_time_limits(self):
        # Game time limit
        if self.total_start_ns:
            elapsed = (time.perf_counter_ns() - self.total_start_ns) / 1e9
            if elapsed > GAME_TIME_LIMIT_SEC*self.players.__len__():
                self.status = "FINISHED"
                self.turn_timeout_notice = "GAME TIME LIMIT REACHED."
                self.compute_scores()
                write_log(self.game_id, self.turn_timeout_notice)
                print(self.game_id, ':',self.turn_timeout_notice)
                return

        # Per-turn timeout
        if self.status == "PLAYING" and self.turn_started_ns:
            turn_elapsed = (time.perf_counter_ns() - self.turn_started_ns) / 1e9
            if turn_elapsed > TURN_TIMEOUT_SEC:
                colour = self.current_turn_colour()
                self.move_count += 1
                if colour in self.wrong_moves_by_colour:
                    self.wrong_moves_by_colour[colour].append(str(self.move_count)+" (turn timeout)")
                else:
                    self.wrong_moves_by_colour[colour] = [str(self.move_count)+" (turn timeout)"]
                self.turn_timeout_notice = (
                    f"Player with colour {colour} exceeded {TURN_TIMEOUT_SEC}s at move {self.move_count}. Turn skipped."
                )
                self.compute_scores()
                write_log(self.game_id, f"TURN TIMEOUT: {self.turn_timeout_notice}")
                print(self.game_id, ':',self.turn_timeout_notice)
                self.advance_turn()

    # ----------------------------------
    def check_player_status(self, colour: str) -> str:
        opposite = self.board.colour_opposites[colour]
        pins = self.pins_by_colour[colour]

        # WIN: all pins reached opposite target zones
        if all(self.board.cells[p.axialindex].postype == opposite for p in pins):
            return "WIN"

        # DRAW: no possible moves
        if all(len(p.getPossibleMoves()) == 0 for p in pins):
            return "DRAW"

        return "PLAYING"

    # ----------------------------------
    # Scoring Logic
    # ----------------------------------
    def compute_scores(self):
        def axial_dist(a, b):
            dq = abs(a.q - b.q)
            dr = abs(a.r - b.r)
            ds = abs((-a.q - a.r) - (-b.q - b.r))
            return max(dq, dr, ds)

        for pl in self.players:
            colour = pl.colour
            pins = self.pins_by_colour[colour]
            opposite = self.board.colour_opposites[colour]

            # Time score
            time_score = max(0.0, 100.0 - pl.time_taken_sec) if pl.time_taken_sec > 0 else 0

            # Move score — asymmetric Gaussian
            move_score_func = lambda x: math.exp(-((x - 45) ** 2) /
                                                 (2 * ((4 if x < 45 else 18) ** 2)))
            move_score = 100*move_score_func(pl.move_count) if pl.move_count > 0 else 0

            # Pins in goal
            pins_in_goal = sum(
                1 for p in pins
                if self.board.cells[p.axialindex].postype == opposite
            )
            pin_goal_score = pins_in_goal * 100.0

            # Distance score
            target_idxs = self.board.axial_of_colour(opposite)
            target_cells = [self.board.cells[i] for i in target_idxs]
            total_dist = 0
            for p in pins:
                if self.board.cells[p.axialindex].postype != opposite:
                    best = min(axial_dist(self.board.cells[p.axialindex], tgt)
                               for tgt in target_cells)
                    total_dist += best
            distance_score = max(0.0, 400.0 - 2*total_dist) if pl.move_count > 0 else 0

            final_score = time_score + move_score + pin_goal_score + distance_score

            #if pl.status == "WIN", add a big bonus to final_score to ensure wins rank higher than any non-win
            if pl.status == "WIN":
                final_score += 1000.0
                win_bonus = 1000.0
            else:
                win_bonus = 0.0

            self.scores[pl.player_id] = {
                "final_score": final_score,
                "time_score": time_score,
                "move_score": move_score,
                "pin_goal_score": pin_goal_score,
                "distance_score": distance_score,
                "moves": pl.move_count,
                "pins_in_goal": pins_in_goal,
                "total_distance": total_dist,
                "time_taken_sec": pl.time_taken_sec,
                "win_bonus": win_bonus
            }

            write_log(
                self.game_id,
                f"SCORE {pl.name} ({colour}): Final={final_score:.1f}, "
                f"Time={time_score:.1f}, Moves({pl.move_count})={move_score:.1f}, "
                f"Pins({pins_in_goal})={pin_goal_score:.1f}, Dist={distance_score:.1f}, Win Bonus={win_bonus:.1f}"
            )
        write_log(self.game_id, f"ALL WRONG MOVES BY COLOUR: {self.wrong_moves_by_colour}")
        #update round_df with final score and status if this game is in the round data
        if SESSION.round_df is not None:
            idx = SESSION.round_df.index[SESSION.round_df['game_id'] == self.game_id].tolist()
            if idx:
                idx = idx[0]
                final_scores_str = ';'.join([f"{pl.name}:{self.scores[pl.player_id]['final_score']:.1f}" for pl in self.players])
                SESSION.round_df.at[idx, 'final_scores'] = final_scores_str
                SESSION.round_df.at[idx, 'status'] = self.status
                SESSION.round_df.at[idx, 'winner'] = next((pl.name for pl in self.players if pl.status == "WIN"), "None")
                SESSION.round_df.at[idx, 'time_scores'] = ';'.join([f"{pl.name}:{self.scores[pl.player_id]['time_score']:.1f}" for pl in self.players]) 
                SESSION.round_df.at[idx, 'distance_scores'] = ';'.join([f"{pl.name}:{self.scores[pl.player_id]['distance_score']:.1f}" for pl in self.players]) 
                SESSION.round_df.at[idx, 'pin_scores'] = ';'.join([f"{pl.name}:{self.scores[pl.player_id]['pin_goal_score']:.1f}" for pl in self.players])  
                SESSION.round_df.at[idx, 'move_scores'] = ';'.join([f"{pl.name}:{self.scores[pl.player_id]['move_score']:.1f}" for pl in self.players])  
                SESSION.round_df.at[idx, 'valid_moves'] = ';'.join([f"{pl.name}:{pl.move_count}" for pl in self.players])
                SESSION.round_df.at[idx, 'skipped_turns'] = ';'.join([f"{pl.name}:{len(self.wrong_moves_by_colour[pl.colour]) if pl.colour in self.wrong_moves_by_colour else 0}" for pl in self.players])
                try:
                    SESSION.round_df.to_csv(SESSION.round_path, sep=',', index=False)
                except Exception as e:
                    error_msg = f"Error occurred while saving round_df after score update: {e}"
                    print(error_msg)
                    write_log("SESSION", error_msg)

    # ----------------------------------
    # Public State
    # ----------------------------------
    def to_public_state(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "status": self.status,
            "players": [
                {
                    "player_id": pl.player_id,
                    "name": pl.name,
                    "colour": pl.colour,
                    "ready": pl.ready,
                    "status": pl.status,
                    "score": self.scores.get(pl.player_id),
                }
                for pl in self.players
            ],
            "pins": {
                colour: [p.axialindex for p in pins]
                for colour, pins in self.pins_by_colour.items()
            },
            "move_count": self.move_count,
            "current_turn_colour": self.current_turn_colour(),
            "turn_order": self.turn_order,
            "last_move": self.last_move,
            "turn_timeout_notice": self.turn_timeout_notice,
        }


# ==========================================================
# Session
# ==========================================================
class Session:
    def __init__(self):
        self.games: Dict[str, Game] = {}
        self.session_games: List[str] = []
        self.lock = threading.RLock()
        self.round_path = f"round{round_number}.txt"
        if os.path.exists(self.round_path):
            with open(self.round_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.round_data = [line.strip().split(',') for line in lines]
        else:
            self.round_data = []
        print(self.round_data)
        self.round_headers = ["game_number", "game_id", "player1", "player2", "player3", "player4","player5","player6", "status", "joined", "final_scores", "time_scores", "distance_scores", "pin_scores","move_scores", "valid_moves", "skipped_turns", "winner"]
        self.round_df = None
        if self.round_data:
            self.round_df = pd.DataFrame(self.round_data[1:], columns=self.round_headers) 
        else:
            error_msg = f"Round data file {self.round_path} not found. Round tracking disabled."
            print(error_msg)
            write_log("SESSION", error_msg)

    # Create one game for each row in round_df that has status NOT_CREATED, and update round_df with the new game_id and status GAME_CREATED. This allows us to track which games correspond to which rows in the round data.
    def create_round_games(self) -> str:
        print ("Creating round games...")
        if self.round_df is None:
            return "No round data available."
        with self.lock:
            rowcnt=0
            for idx, row in self.round_df.iterrows():
                if row['status'] == 'NOT_CREATED':
                    gid = self.create_game()
                    self.round_df.at[idx, 'game_id'] = gid
                    self.round_df.at[idx, 'status'] = 'GAME_CREATED'
                    write_log("SESSION", f"Round game created: {gid} for row {idx}")
                    rowcnt+=1
            if rowcnt==0:
                print("No games to be created. All games already created for this round.")
            else:
                print(f"{rowcnt} round games created for Round {round_number}.")
            # Save updated round_df back to file
            # try to save the round_df back to the round file, and log any errors that occur during saving without crashing the server
            try:
                self.round_df.to_csv(self.round_path, sep=',', index=False) 
            except Exception as e:
                error_msg = f"Error occurred while saving round_df: {e}"
                print(error_msg)
                write_log("SESSION", error_msg)
            return "Round games created and round data updated."
    # ------------------------------
    def create_game(self) -> str:
        with self.lock:
            g = Game()
            self.games[g.game_id] = g
            self.session_games.append(g.game_id)
            write_log(g.game_id, "GAME CREATED")

            return g.game_id

    # ------------------------------
    def pick_available_game(self) -> Optional[Game]:
        for gid in self.session_games:
            g = self.games[gid]
            if not g.lock_joining and len(g.players) < MAX_PLAYERS:
                if g.status in ("waiting for other player", "AVAILABLE", "READY_TO_START"):
                    return g
        return None

    def find_round_game_for_player(self, player_name: str) -> Dict[str, Any]:
        if self.round_df is None:
            return {"ok": False, "error": "No round data available."}
        with self.lock:
            # Filter rows where player_name is in any of the player columns and status is GAME_CREATED or WAITING_FOR_OTHER_PLAYER and player has not already joined the game (to prevent joining multiple times if player refreshes)
            candidate_rows = self.round_df[
                (self.round_df[['player1', 'player2', 'player3', 'player4', 'player5', 'player6']].apply(lambda row: player_name in row.values, axis=1) &
                (self.round_df['status'].isin(['GAME_CREATED', 'WAITING_FOR_OTHER_PLAYER'])) )]
            for row in candidate_rows.itertuples():
                candidate_game_id = row.game_id
                candidate_game_players = self.games[candidate_game_id].players
                if any(p.name == player_name for p in candidate_game_players):
                    candidate_rows = candidate_rows.drop(row.Index)
            #print(f"Found {len(candidate_rows)} candidate rows for player {player_name} in round data.")

            if candidate_rows.empty:
                return {"ok": False, "error": f"No available game found for player {player_name}. Check if the player name is correct and if there are games waiting for players."}

            # Prioritize rows with WAITING_FOR_OTHER_PLAYER status
            waiting_rows = candidate_rows[candidate_rows['status'] == 'WAITING_FOR_OTHER_PLAYER']
            #print(f"Found {len(waiting_rows)} candidate rows with WAITING_FOR_OTHER_PLAYER status for player {player_name}.")
            if not waiting_rows.empty:
                candidate_rows = waiting_rows

            # Pick the row with the maximum number of players already joined
            candidate_rows['joined_count'] = candidate_rows.apply(lambda row: sum(pd.notna(row[['player1', 'player2', 'player3', 'player4', 'player5', 'player6']])), axis=1)
            selected_row = candidate_rows.sort_values(by='joined_count', ascending=False).iloc[0]

            game_id = selected_row['game_id']
            g = self.games.get(game_id)
            if not g:
                return {"ok": False, "error": "Game not found for the selected round row."}

            colour = g.assign_colour()
            if not colour:
                return {"ok": False, "error": "Game full"}

            pid = str(uuid.uuid4())
            pl = Player(pid, player_name, colour)
            g.players.append(pl)
            g.init_pins(colour)

            # Update game status based on how many players have joined compared to how many are expected from the round data
            all_players_in_row = selected_row[['player1', 'player2', 'player3', 'player4', 'player5', 'player6']]
            all_players_in_row = all_players_in_row[all_players_in_row != 'NA']
            '''try:
                with open(f"joined_{round_number}.txt", "w") as f:
                    for gid in self.session_games:
                        g = self.games[gid]
                        for p in g.players:
                            f.write(f"{p.name} joined game {gid} as {p.colour}\n")
                    not_joined = [p for p in all_players_in_row if p != player_name and p not in joined_players]
                    f.write(f"Players not joined yet for game {game_id}: {', '.join(not_joined) if not_joined else 'None'}\n")
            except Exception as e:
                error_msg = f"Error occurred while writing to joined_{round_number}.txt: {e}"
                print(error_msg)
                write_log("SESSION", error_msg)'''
            if self.round_df.at[selected_row.name, 'joined'] == 'NA':
                self.round_df.at[selected_row.name, 'joined'] = player_name+':'+colour
            else:
                self.round_df.at[selected_row.name, 'joined'] += ';'+player_name+':'+colour
            if len(g.players) < len(all_players_in_row):
                g.status = "waiting for other player"
                self.round_df.at[selected_row.name, 'status'] = 'WAITING_FOR_OTHER_PLAYER'
            else:
                g.status = "READY_TO_START"
                self.round_df.at[selected_row.name, 'status'] = 'READY_TO_START'
            write_log(g.game_id, f"PLAYER JOINED: {player_name} as {colour}")
            print(f"Player {player_name} joined game {g.game_id} as {colour}.")
            try:
                self.round_df.to_csv(self.round_path, sep=',', index=False) 
            except Exception as e:
                error_msg = f"Error occurred while saving round_df: {e}"
                print(error_msg)
                write_log("SESSION", error_msg)


            return {
                "ok": True,
                "game_id": g.game_id,
                "player_id": pid,
                "colour": colour,
                "status": g.status,
            }
        return {"ok": False, "error": "Unexpected error in finding round game for player."}
    '''def find_round_game_for_player(self, player_name: str) -> Optional[Dict[str, Any]]:
        if self.round_df is None:
            return None
        with self.lock:
            for idx, row in self.round_df.iterrows():
                if player_name in row[['player1', 'player2', 'player3', 'player4']].values and row['status'] in ('GAME_CREATED', 'WAITING_FOR_OTHER_PLAYER'):
                    game_id = row['game_id']
                    g = self.games.get(game_id)
                    if g:
                        colour = g.assign_colour()
                        if colour:
                            pid = str(uuid.uuid4())
                            pl = Player(pid, player_name, colour)
                            g.players.append(pl)
                            g.init_pins(colour)

                            #count number of non-NA player names in the row to determine how many players are expected in this game
                            all_players_in_row = row[['player1', 'player2', 'player3', 'player4']]
                            all_players_in_row = all_players_in_row[all_players_in_row != 'NA']
                            if len(g.players) < len(all_players_in_row):
                                g.status = "waiting for other player"
                                self.round_df.at[idx, 'status'] = 'WAITING_FOR_OTHER_PLAYER'
                            else:
                                g.status = "READY_TO_START"
                                self.round_df.at[idx, 'status'] = 'READY_TO_START'

                            write_log(g.game_id, f"PLAYER JOINED: {player_name} as {colour}")
                            print(f"Player {player_name} joined game {g.game_id} as {colour}.")

                            # Save updated round_df back to file
                            self.round_df.to_csv(self.round_path, sep=',', index=False)

                            return {
                                "ok": True,
                                "game_id": g.game_id,
                                "player_id": pid,
                                "colour": colour,
                                "status": g.status,
                            }
        return {
            "ok": False,
            "error": f"No available game found for player {player_name}. Check if the player name is correct and if there are games waiting for players."
        }
        #return None'''
    
    # ------------------------------
    def join_request(self, player_name: str) -> Dict[str, Any]:
        with self.lock:
            g = self.pick_available_game()
            if not g:
                return {"ok": False, "error": "No available game. Ask admin to Create."}

            colour = g.assign_colour()
            if not colour:
                return {"ok": False, "error": "Game full"}

            pid = str(uuid.uuid4())
            pl = Player(pid, player_name, colour)
            g.players.append(pl)
            g.init_pins(colour)

            if len(g.players) == 1:
                g.status = "waiting for other player"
            else:
                g.status = "READY_TO_START"

            write_log(g.game_id, f"PLAYER JOINED: {player_name} as {colour}")
            print(f"Player {player_name} joined game {g.game_id} as {colour}.")


            return {
                "ok": True,
                "game_id": g.game_id,
                "player_id": pid,
                "colour": colour,
                "status": g.status,
            }


    '''
    forcestart_game is a method that can be called by the admin to force start a game if players are waiting for others to join. This is useful in cases where some players may have left or are not joining, and we don't want the remaining players to be stuck waiting indefinitely. It will mark all current players as ready and start the game if there are at least 2 players, even if the expected number of players from the round data has not been reached.
    '''
    def forcestart_game(self, game_id: str) -> Dict[str, Any]:
        with self.lock:
            g = self.games.get(game_id)
            if not g:
                return {"ok": False, "error": "Game not found"}

            for pl in g.players:
                pl.ready = True
                write_log(g.game_id, f"PLAYER START (FORCE): {pl.name} ({pl.colour})")
            

            if g.status in ["READY_TO_START", "waiting for other player"]:
                g.lock_joining = True
                if len(g.players) >= 2 :
                    g.status = "PLAYING"
                    g.total_start_ns = time.perf_counter_ns()
                    g.compute_turn_order()
                    g.turn_started_ns = time.perf_counter_ns()
                    write_log(g.game_id, f"GAME FORCE-STARTED — turn order {g.turn_order}")

            # Update round_df status to PLAYING if this game is in the round data
            if self.round_df is not None:
                idx = self.round_df.index[self.round_df['game_id'] == game_id].tolist()
                if idx:
                    idx = idx[0]
                    if g.status == "PLAYING":
                        self.round_df.at[idx, 'status'] = 'PLAYING'
                        # try to save the round_df back to the round file, and log any errors that occur during saving without crashing the server
                        try:
                            self.round_df.to_csv(self.round_path, sep=',', index=False) 
                        except Exception as e:
                            error_msg = f"Error occurred while saving round_df: {e}"
                            print(error_msg)
                            write_log("SESSION", error_msg)

            return {"ok": True, "status": g.status}
    '''
    If all players have joined a game, and the game status is READY_TO_START, then mark the player as ready. If all players are ready, change game status to PLAYING and compute turn order. This allows us to track when players are ready and when the game starts in the round data.
    '''
    def start_game(self, game_id: str) -> Dict[str, Any]:
        with self.lock:
            g = self.games.get(game_id)
            if not g:
                return {"ok": False, "error": "Game not found"}

            for pl in g.players:
                pl.ready = True
                write_log(g.game_id, f"PLAYER START: {pl.name} ({pl.colour})")
            

            if g.status == "READY_TO_START":
                g.lock_joining = True
                if len(g.players) >= 2 and all(pl.ready for pl in g.players):
                    g.status = "PLAYING"
                    g.total_start_ns = time.perf_counter_ns()
                    g.compute_turn_order()
                    g.turn_started_ns = time.perf_counter_ns()
                    write_log(g.game_id, f"GAME START — turn order {g.turn_order}")

            # Update round_df status to PLAYING if this game is in the round data
            if self.round_df is not None:
                idx = self.round_df.index[self.round_df['game_id'] == game_id].tolist()
                if idx:
                    idx = idx[0]
                    if g.status == "PLAYING":
                        self.round_df.at[idx, 'status'] = 'PLAYING'
                        # try to save the round_df back to the round file, and log any errors that occur during saving without crashing the server
                        try:
                            self.round_df.to_csv(self.round_path, sep=',', index=False) 
                        except Exception as e:
                            error_msg = f"Error occurred while saving round_df: {e}"
                            print(error_msg)
                            write_log("SESSION", error_msg)

            return {"ok": True, "status": g.status}
    # ------------------------------
    def mark_start_ready(self, game_id: str, player_id: str) -> Dict[str, Any]:
        with self.lock:
            g = self.games.get(game_id)
            if not g:
                return {"ok": False, "error": "Game not found"}

            for pl in g.players:
                if pl.player_id == player_id:
                    pl.ready = True
                    write_log(g.game_id, f"PLAYER START: {pl.name} ({pl.colour})")
                    break

            if g.status == "READY_TO_START":
                g.lock_joining = True
                if len(g.players) >= 2 and all(pl.ready for pl in g.players):
                    g.status = "PLAYING"
                    g.total_start_ns = time.perf_counter_ns()
                    g.compute_turn_order()
                    g.turn_started_ns = time.perf_counter_ns()
                    write_log(g.game_id, f"GAME START — turn order {g.turn_order}")

            return {"ok": True, "status": g.status}

    def get_legal_moves(self, game_id: str, player_id: str):
        with self.lock:
            g = self.games.get(game_id)
            if not g:
                return {"ok": False, "error": "Game not found"}

            pl = next((p for p in g.players if p.player_id == player_id), None)
            if not pl:
                return {"ok": False, "error": "Player not found"}

            pins = g.pins_by_colour[pl.colour]
            legal = {}

            for i, pin in enumerate(pins):
                legal[i] = pin.getPossibleMoves()

            return {"ok": True, "legal_moves": legal}

    # ------------------------------
    def validate_and_apply_move(self, game_id: str, player_id: str, pin_id: int, to_index: int):
        with self.lock:
            g = self.games.get(game_id)
            if not g:
                return {"ok": False, "error": "Game not found"}

            g.ensure_time_limits()

            if g.status != "PLAYING":
                return {"ok": False, "error": f"Game not in PLAYING: {g.status}"}

            pl = next((p for p in g.players if p.player_id == player_id), None)
            if not pl:
                return {"ok": False, "error": "Player not in game"}

            if g.current_turn_colour() != pl.colour:
                '''g.move_count += 1
                if pl.colour in g.wrong_moves_by_colour:
                    g.wrong_moves_by_colour[pl.colour].append(str(g.move_count)+" (not player's turn)")
                else:
                    g.wrong_moves_by_colour[pl.colour] = [str(g.move_count)+" (not player's turn)"]'''
                return {"ok": False, "error": f"Not {pl.colour}'s turn. "}

            pins = g.pins_by_colour[pl.colour]
            if not (0 <= pin_id < len(pins)):
                g.move_count += 1
                if pl.colour in g.wrong_moves_by_colour:
                    g.wrong_moves_by_colour[pl.colour].append(str(g.move_count)+" (invalid pin_id)")
                else:
                    g.wrong_moves_by_colour[pl.colour] = [str(g.move_count)+" (invalid pin_id)"]
                return {"ok": False, "error": "Invalid pin ID"}

            pin = pins[pin_id]
            legal = pin.getPossibleMoves()
            if to_index not in legal:
                g.move_count += 1
                if pl.colour in g.wrong_moves_by_colour:
                    g.wrong_moves_by_colour[pl.colour].append(str(g.move_count)+" (illegal move)")
                else:
                    g.wrong_moves_by_colour[pl.colour] = [str(g.move_count)+" (illegal move)"]
                return {"ok": False, "error": "Illegal move"}

            # Time tracking
            if g.turn_started_ns:
                dt = (time.perf_counter_ns() - g.turn_started_ns) / 1e9
                pl.time_taken_sec += dt

            # Apply move
            start_ns = time.perf_counter_ns()
            from_idx = pin.axialindex
            moved_ok = pin.placePin(to_index)
            end_ns = time.perf_counter_ns()
            move_ms = (end_ns - start_ns) / 1e6

            if not moved_ok:
                return {"ok": False, "error": "Could not move"}

            pl.move_count += 1
            g.move_count += 1
            g.move_times_ms.append(move_ms)
            if g.turn_timeout_notice is not None:
                timeout_movenum = int(g.turn_timeout_notice.split('move')[1].split('.')[0].replace(' ',''))
                print(g.turn_timeout_notice.split('move')[1], g.turn_timeout_notice.split('move')[1].split('.'))
                if timeout_movenum<g.move_count:
                    g.turn_timeout_notice = None

            g.last_move = {
                "pin_id": pin_id,
                "from": from_idx,
                "to": to_index,
                "by": pl.name,
                "colour": pl.colour,
                "move_ms": move_ms,
            }
            write_log(
                g.game_id,
                f"MOVE {g.move_count}: {pl.name} ({pl.colour}) {from_idx}->{to_index} [{move_ms:.2f}ms]"
            )

            # Check WIN / DRAW
            pl.status = g.check_player_status(pl.colour)

            if pl.status == "WIN":
                g.status = "FINISHED"
                g.compute_scores()

                return {"ok": True, "status": "WIN", "state": g.to_public_state(),
                        "msg": f"{pl.name} Wins"}

            if pl.status == "DRAW":
                # Check if all others draw except one
                live = g.players
                draws = [p for p in live if g.check_player_status(p.colour) == "DRAW"]
                if len(draws) == len(live) - 1:
                    winner = next(p for p in live if p not in draws)
                    g.status = "FINISHED"
                    g.compute_scores()

                    return {"ok": True, "status": "WIN", "state": g.to_public_state(),
                            "msg": f"{winner.name} Wins, others Draw."}

            # Continue game
            g.advance_turn()
            g.compute_scores()

            return {"ok": True, "status": "CONTINUE",
                    "state": g.to_public_state()}

    # ------------------------------
    def game_status_list(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [
                {
                    "game_id": g.game_id,
                    "players": [{"name": p.name, "colour": p.colour} for p in g.players],
                    "moves": g.move_count,
                    "status": g.status,
                }
                for gid in self.session_games
                for g in [self.games[gid]]
            ]


SESSION = Session()

# ==========================================================
# RPC HANDLER
# ==========================================================
def handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    op = req.get("op")

    if op == "get_legal_moves":
        return SESSION.get_legal_moves(
            req.get("game_id"),
            req.get("player_id")
        )

    if op == "join":
        #return SESSION.join_request(req.get("player_name"))
        return SESSION.find_round_game_for_player(req.get("player_name"))

    if op == "start":
        #return SESSION.mark_start_ready(req.get("game_id"), req.get("player_id"))
        return SESSION.start_game(req.get("game_id"))
    if op == "get_state":
        game_id = req.get("game_id")
        g = SESSION.games.get(game_id)
        if not g:
            return {"ok": False, "error": "Game not found"}
        g.ensure_time_limits()
        return {"ok": True, "state": g.to_public_state()}

    if op == "move":
        return SESSION.validate_and_apply_move(
            req.get("game_id"),
            req.get("player_id"),
            int(req.get("pin_id")),
            int(req.get("to_index")),
        )

    if op == "status":
        return {"ok": True, "games": SESSION.game_status_list()}


    return {"ok": False, "error": f"Unknown op {op}"}


# ==========================================================
# SERVER LOOP
# ==========================================================
def server_loop():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("0.0.0.0", 50555))
    s.listen(50)
    print("[Server] Listening on 0.0.0.0:50555")

    while True:
        conn, addr = s.accept()
        try:
            conn.settimeout(10.0)
            data = conn.recv(65535)
            if not data:
                conn.close()
                continue
            try:
                req = json.loads(data.decode("utf-8"))
            except:
                resp = {"ok": False, "error": "bad-json"}
                conn.sendall(json.dumps(resp).encode("utf-8"))
                conn.close()
                continue

            resp = handle_request(req)
            js = safe_json(resp)
            conn.sendall(js.encode("utf-8"))

        finally:
            conn.close()




# ==========================================================
# CLI LOOP
# ==========================================================
def cli_loop():
    print("Game Manager")
    print("Commands: Create, Status, Quit\n")

    while True:
        cmd = input("Enter command: ").strip().lower()
        if cmd == "create":
            SESSION.create_round_games()
            #print("Game created:", gid)
        elif cmd == "status":
            for g in SESSION.game_status_list():
                print(g)
        elif cmd == "quit":
            os._exit(0)
        elif cmd.startswith("start"):
            start_game_id = cmd.split(' ')[1] if len(cmd.split(' ')) > 1 else None
            if start_game_id:
                result = SESSION.forcestart_game(start_game_id)
                if result.get("ok"):
                    print(f"Game {start_game_id} started.")
                else:
                    print(f"Error starting game {start_game_id}: {result.get('error')}")
        else:
            print("Invalid command")


# ==========================================================
# ENTRY POINT
# ==========================================================
if __name__ == "__main__":
    threading.Thread(target=server_loop, daemon=True).start()
    cli_loop()
