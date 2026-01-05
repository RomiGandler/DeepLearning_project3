import chess.pgn
import csv
import sys
import argparse
import os

# Convert PGN file to CSV of FENs
def convert_pgn_to_csv(pgn_path, csv_output_path):
    print(f"Reading PGN: {pgn_path}")
    
    fens = []
    game_count = 0
    
    with open(pgn_path, encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            game_count += 1
            board = game.board()
            
            # initial position
            fens.append(board.fen())
            
            # subsequent positions
            for move in game.mainline_moves():
                board.push(move)
                fens.append(board.fen())
                
    print(f"Processed {game_count} games.")
    print(f"Extracted {len(fens)} positions.")

    # Write to CSV
    with open(csv_output_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['fen']) # Header
        for fen in fens:
            writer.writerow([fen])
            
    print(f"Saved to: {csv_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('pgn_file', help="Input PGN file")
    parser.add_argument('csv_file', help="Output CSV file")
    args = parser.parse_args()
    
    convert_pgn_to_csv(args.pgn_file, args.csv_file)