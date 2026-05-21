import random

def generate_games(players, num_rounds=3):
    # Create a list to hold rounds of games
    rounds = []
    
    # Create a set to track player matchups
    matchups = set()
    
    for round_num in range(num_rounds):
        round_games = []
        available_players = set(players)
        
        while len(available_players) >= 2:
            # Randomly select the number of players for this game (between 2 and 6)
            num_players_in_game = random.randint(2, min(6, len(available_players)))
            
            # Randomly select players for this game
            available_players = list(available_players)
            selected_players = random.sample(available_players, num_players_in_game)
            
            # Check if this matchup has occurred before
            matchup_tuple = tuple(sorted(selected_players))
            if matchup_tuple not in matchups:
                matchups.add(matchup_tuple)
                round_games.append(selected_players)
                available_players = set(available_players) - set(selected_players)
        
        rounds.append(round_games)
    
    return rounds

#player_list = ["CybSec", "Group7", "Gruppe 2","AlphaZeroClue","Apollo", "Gruppe 1","Robert", "Tiefes Verstarkendes Lernen", "Gruppe 67", "The bandits", "jat", "S&S", "Group 99", "Erlend Og Linor", "RL"]
player_list = ["A", "B", "C", "D", "E", "F"]
r = generate_games(player_list)

#pretty print the round
rnum=1
for ground in r:
    gnum=1
    for game in ground:
        print(f"Round {rnum} Game {gnum}",game)
        gnum+=1
    rnum+=1

#count total games for each player
player_game_count = {player: 0 for player in player_list}
for ground in r:
    for game in ground:
        for player in game:
            player_game_count[player] += 1
#print("Rounds of games:", r)
print("Player game counts:", player_game_count)

for round_num, ground in enumerate(r, start=11):
    round_text = "game_number,game_id,player1,player2,player3,player4,player5,player6,status,joined,final_scores,time_scores,distance_scores,pin_scores,move_scores,valid_moves,skipped_turns,winner\n"
    for game_num, game in enumerate(ground, start=1):
        game_number = f"R{round_num}G{game_num}"
        player1 = game[0] if len(game) > 0 else "NA"
        player2 = game[1] if len(game) > 1 else "NA"
        player3 = game[2] if len(game) > 2 else "NA"
        player4 = game[3] if len(game) > 3 else "NA"
        player5 = game[4] if len(game) > 4 else "NA"
        player6 = game[5] if len(game) > 5 else "NA"
        game_id = "NA"
        status = "NOT_CREATED"
        final_scores = "NA"
        time_scores = "NA"
        distance_scores = "NA"
        pin_scores = "NA"
        move_scores = "NA"
        valid_moves = 0
        skipped_turns = 0
        joined = "NA"
        winner = None
        round_text += f"{game_number},{game_id},{player1},{player2},{player3},{player4},{player5},{player6},{status},{joined},{final_scores},{time_scores},{distance_scores},{pin_scores},{move_scores},{valid_moves},{skipped_turns},{winner}\n"
        
    print(f"Round {round_num}:\n{round_text}")
    #write to file round{round_num}.txt
    with open(f"round{round_num}.txt", "w") as f:
        f.write(round_text)
    with open(f"round{round_num}_initialcopy.txt", "w") as f:
        f.write(round_text)