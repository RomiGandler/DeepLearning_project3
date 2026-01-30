"""
FEN string utilities for chess board augmentation.

Provides functions for manipulating FEN (Forsyth-Edwards Notation) strings,
particularly for color inversion (swapping black and white pieces).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def invert_fen_colors(fen: str) -> str:
    """
    Invert piece colors in a FEN string (swap black <-> white).

    In FEN notation:
    - Uppercase letters (KQRBNP) = White pieces
    - Lowercase letters (kqrbnp) = Black pieces

    This function swaps the case of all piece characters, effectively
    swapping black and white pieces on the board.

    Args:
        fen: FEN string (full or just the board portion)

    Returns:
        FEN string with piece colors inverted

    Example:
        >>> invert_fen_colors("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        'RNBQKBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr'
    """
    # Extract just the board portion (first part before space)
    parts = fen.split()
    board_part = parts[0]

    # Swap case for the board portion
    inverted_board = board_part.swapcase()

    # If there were additional FEN fields (turn, castling, etc.), preserve them
    # but also swap the active color (w <-> b)
    if len(parts) > 1:
        other_parts = parts[1:]
        # Swap active color if present
        if other_parts[0] in ('w', 'b'):
            other_parts[0] = 'b' if other_parts[0] == 'w' else 'w'
        return f"{inverted_board} {' '.join(other_parts)}"

    return inverted_board


def add_inv_suffix(img_name: str) -> str:
    """
    Add '_inv' suffix to an image filename before the extension.

    Args:
        img_name: Original image filename (e.g., "game11_frame_013068.png")

    Returns:
        Filename with _inv suffix (e.g., "game11_frame_013068_inv.png")

    Example:
        >>> add_inv_suffix("game11_frame_013068.png")
        'game11_frame_013068_inv.png'
        >>> add_inv_suffix("board.jpg")
        'board_inv.jpg'
    """
    path = Path(img_name)
    return f"{path.stem}_inv{path.suffix}"


def has_inv_suffix(img_name: str) -> bool:
    """
    Check if an image filename already has the '_inv' suffix.

    Args:
        img_name: Image filename to check

    Returns:
        True if filename ends with '_inv' before extension

    Example:
        >>> has_inv_suffix("game11_frame_013068_inv.png")
        True
        >>> has_inv_suffix("game11_frame_013068.png")
        False
    """
    path = Path(img_name)
    return path.stem.endswith("_inv")


def parse_fen_board(fen: str) -> list[list[Optional[str]]]:
    """
    Parse a FEN string into an 8x8 board representation.

    Args:
        fen: FEN string (full or just board portion)

    Returns:
        8x8 list where each cell is either:
        - A piece character (K, Q, R, B, N, P, k, q, r, b, n, p)
        - None for empty squares

    Example:
        >>> board = parse_fen_board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR")
        >>> board[0][0]  # a8
        'r'
        >>> board[7][4]  # e1
        'K'
    """
    board_part = fen.split()[0]
    ranks = board_part.split('/')

    board = []
    for rank_str in ranks:
        rank = []
        for char in rank_str:
            if char.isdigit():
                rank.extend([None] * int(char))
            else:
                rank.append(char)
        board.append(rank)

    return board


def board_to_fen(board: list[list[Optional[str]]]) -> str:
    """
    Convert an 8x8 board representation back to FEN notation.

    Args:
        board: 8x8 list of pieces or None for empty squares

    Returns:
        FEN board string (just the board portion)

    Example:
        >>> board = [['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
        ...          ['p'] * 8,
        ...          [None] * 8,
        ...          [None] * 8,
        ...          [None] * 8,
        ...          [None] * 8,
        ...          ['P'] * 8,
        ...          ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']]
        >>> board_to_fen(board)
        'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR'
    """
    ranks = []
    for rank in board:
        rank_str = ""
        empty_count = 0
        for cell in rank:
            if cell is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += cell
        if empty_count > 0:
            rank_str += str(empty_count)
        ranks.append(rank_str)

    return "/".join(ranks)


def validate_fen(fen: str) -> bool:
    """
    Basic validation of a FEN string's board portion.

    Args:
        fen: FEN string to validate

    Returns:
        True if the board portion is valid, False otherwise
    """
    board_part = fen.split()[0]
    ranks = board_part.split('/')

    if len(ranks) != 8:
        return False

    valid_pieces = set("kqrbnpKQRBNP")

    for rank in ranks:
        file_count = 0
        for char in rank:
            if char.isdigit():
                file_count += int(char)
            elif char in valid_pieces:
                file_count += 1
            else:
                return False

        if file_count != 8:
            return False

    return True
