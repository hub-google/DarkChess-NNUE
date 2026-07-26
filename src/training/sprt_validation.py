import argparse
import torch
import numpy as np
import os
from train import DarkChessNNUE, extract_features
from board import DarkChessBoardPy

def get_best_move(model, board, moves, color):
    import copy
    best_eval = -float('inf')
    best_move = moves[0]
    
    batch_features = []
    
    for m in moves:
        b2 = copy.deepcopy(board)
        b2.make_move(m)
        feat = extract_features(b2)
        batch_features.append(feat)
        
    batch_tensor = torch.tensor(np.array(batch_features), dtype=torch.float32)
    with torch.no_grad():
        evals = model(batch_tensor).squeeze().numpy()
        
    if len(moves) == 1:
        return moves[0]
        
    if color == 0:
        best_idx = np.argmax(evals)
    else:
        best_idx = np.argmin(evals)
        
    return moves[best_idx]

def play_game(model_red, model_black):
    board = DarkChessBoardPy()
    
    for ply in range(100):
        if board.side_to_move == 2: # NONE, means we must flip
            moves = board.generate_legal_moves()
            if len(moves) == 0:
                return 0.0 # Draw
            m = np.random.choice(moves)
            board.make_move(m)
            continue
            
        color = board.side_to_move
        moves = board.generate_legal_moves()
        if len(moves) == 0:
            # Current player has no moves -> loses
            return -1.0 if color == 0 else 1.0
            
        if color == 0:
            m = get_best_move(model_red, board, moves, color)
        else:
            m = get_best_move(model_black, board, moves, color)
            
        board.make_move(m)
        
    return 0.0 # Draw by 100 moves

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--champion', required=True)
    parser.add_argument('--challenger', required=True)
    args = parser.parse_args()
    
    champion = DarkChessNNUE()
    challenger = DarkChessNNUE()
    
    if os.path.exists(args.champion):
        champion.load_state_dict(torch.load(args.champion))
    if os.path.exists(args.challenger):
        challenger.load_state_dict(torch.load(args.challenger))
        
    champion.eval()
    challenger.eval()
    
    games = 50
    challenger_score = 0.0
    
    print(f"Starting SPRT validation: Challenger vs Champion ({games} games)")
    for i in range(games):
        if i % 2 == 0:
            res = play_game(challenger, champion) # Challenger is RED
            if res == 1.0: challenger_score += 1.0
            elif res == 0.0: challenger_score += 0.5
        else:
            res = play_game(champion, challenger) # Challenger is BLACK
            if res == -1.0: challenger_score += 1.0
            elif res == 0.0: challenger_score += 0.5
            
        if (i+1) % 10 == 0:
            print(f"Played {i+1} games, Challenger Score: {challenger_score}")
            
    win_rate = challenger_score / games
    print(f"Challenger Win Rate: {win_rate*100:.2f}%")
    
    if win_rate > 0.55:
        print("SPRT Validation PASSED")
        if 'GITHUB_ENV' in os.environ:
            with open(os.environ['GITHUB_ENV'], 'a') as f:
                f.write("PASSED=true\n")
    else:
        print("SPRT Validation FAILED")
        if 'GITHUB_ENV' in os.environ:
            with open(os.environ['GITHUB_ENV'], 'a') as f:
                f.write("PASSED=false\n")

if __name__ == '__main__':
    main()
