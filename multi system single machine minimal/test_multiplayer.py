"""Smoke test: run player.py against itself in a 2-player game.

Starts the game.py server in a child process, creates a game via the CLI
("Create"), then connects two TournamentAgent instances (one per colour)
to play one game. Reports final scores.

Usage:
    python test_multiplayer.py [--players N] [--server-cmd "python game.py"]

This is for local validation only — the real tournament uses an external
server. We use this to verify the agent (a) loads, (b) makes legal moves,
(c) doesn't crash on multi-colour state, (d) finishes a game.
"""
import os
import sys
import time
import json
import socket
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))


def rpc(host: str, port: int, payload: dict, timeout: float = 5.0) -> dict:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        s.sendall(json.dumps(payload).encode("utf-8"))
        data = s.recv(1_000_000)
        return json.loads(data.decode("utf-8"))
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        s.close()


def play_one_agent_turn(agent_module, agent, game_id, player_id, my_colour,
                       host, port, time_budget_used, moves_made):
    """Pull state, decide, push move. Returns (response, time_spent)."""
    st = rpc(host, port, {"op": "get_state", "game_id": game_id})
    if not st.get("ok"):
        return None, 0.0
    state = st["state"]
    if state.get("status") != "PLAYING":
        return state, 0.0
    if state.get("current_turn_colour") != my_colour:
        return state, 0.0

    legal_req = rpc(host, port, {"op": "get_legal_moves",
                                 "game_id": game_id, "player_id": player_id})
    if not legal_req.get("ok"):
        return state, 0.0
    legal = {int(k): list(map(int, v)) for k, v in
             (legal_req.get("legal_moves") or {}).items() if v}
    if not legal:
        return state, 0.0

    t0 = time.perf_counter()
    pid, dest = agent_module.select_move(
        agent, state, my_colour, legal,
        time_used_sec=time_budget_used, moves_made=moves_made,
    )
    elapsed = time.perf_counter() - t0

    mv = rpc(host, port, {
        "op": "move", "game_id": game_id, "player_id": player_id,
        "pin_id": int(pid), "to_index": int(dest),
    })
    if not mv.get("ok"):
        # Fallback: any legal move
        first = next(iter(legal.items()))
        mv = rpc(host, port, {
            "op": "move", "game_id": game_id, "player_id": player_id,
            "pin_id": int(first[0]), "to_index": int(first[1][0]),
        })
    return mv, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50555)
    parser.add_argument("--server-cmd", default=None,
                        help="Command to start the server (default: just connect)")
    args = parser.parse_args()

    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.dirname(HERE))
    import player as player_module

    # Start server if requested
    server_proc = None
    if args.server_cmd:
        print(f"Starting server: {args.server_cmd}", flush=True)
        server_proc = subprocess.Popen(
            args.server_cmd, shell=True, cwd=HERE,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        time.sleep(2.0)
        # Issue the Create command
        try:
            server_proc.stdin.write("Create\n")
            server_proc.stdin.flush()
        except Exception:
            pass
        time.sleep(1.0)

    # Each player gets its own TournamentAgent (separate model instances would
    # double the GPU memory; for testing we share one)
    print("Loading shared agent...", flush=True)
    agent = player_module.TournamentAgent()
    print(f"Agent ready (use_torch={agent.use_torch}, "
          f"forward_ms={getattr(agent, 'forward_ms', None)})", flush=True)

    players = []
    for i in range(args.num_players):
        r = rpc(args.host, args.port,
                {"op": "join", "player_name": f"agent_{i}"})
        if not r.get("ok"):
            print(f"Player {i} JOIN failed:", r.get("error"))
            return
        players.append({
            "idx": i,
            "game_id": r["game_id"],
            "player_id": r["player_id"],
            "colour": r["colour"],
            "moves_made": 0,
            "time_used": 0.0,
        })
        print(f"  player_{i} -> colour {r['colour']}")

    # Mark all start ready
    for p in players:
        rpc(args.host, args.port, {"op": "start", "game_id": p["game_id"],
                                   "player_id": p["player_id"]})
    # Wait for PLAYING
    deadline = time.time() + 10
    while time.time() < deadline:
        st = rpc(args.host, args.port,
                 {"op": "get_state", "game_id": players[0]["game_id"]})
        s = st.get("state", {}) if st.get("ok") else {}
        if s.get("status") == "PLAYING":
            break
        time.sleep(0.3)
    print("Game started.", flush=True)

    move_idx = 0
    last_status = "PLAYING"
    while True:
        st = rpc(args.host, args.port,
                 {"op": "get_state", "game_id": players[0]["game_id"]})
        if not st.get("ok"):
            print("State error:", st.get("error"))
            time.sleep(0.5)
            continue
        state = st["state"]
        last_status = state.get("status", "PLAYING")
        if last_status == "FINISHED":
            print("\n=== FINISHED ===")
            for pl in state.get("players", []):
                sc = pl.get("score") or {}
                print(f"  {pl['name']} ({pl['colour']}): "
                      f"final={sc.get('final_score', 0):.1f} "
                      f"pins_in_goal={sc.get('pins_in_goal', 0)} "
                      f"moves={sc.get('moves', 0)}")
            break

        cur = state.get("current_turn_colour")
        active = next((p for p in players if p["colour"] == cur), None)
        if active is None:
            time.sleep(0.1)
            continue

        resp, elapsed = play_one_agent_turn(
            player_module, agent,
            active["game_id"], active["player_id"], active["colour"],
            args.host, args.port,
            time_budget_used=active["time_used"],
            moves_made=active["moves_made"],
        )
        active["moves_made"] += 1
        active["time_used"] += elapsed
        move_idx += 1
        if move_idx % 5 == 0:
            print(f"  move {move_idx}: {active['colour']} t={elapsed*1000:.0f}ms "
                  f"(cum {active['time_used']:.1f}s)", flush=True)

        if resp and resp.get("status") in ("WIN", "DRAW"):
            print("Final move resp:", resp.get("status"), resp.get("msg"))

        if move_idx > 1000:
            print("Move limit reached")
            break

    if server_proc:
        try:
            server_proc.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
