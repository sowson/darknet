#ifndef LIBCHESS_PIECE_HPP
#define LIBCHESS_PIECE_HPP

#include <array>

namespace libchess {

enum Piece : int
{
    Pawn = 0,
    Knight,
    Bishop,
    Rook,
    Queen,
    King,
    None,
};

inline constexpr std::array<Piece, 6> pieces = {{
    Piece::Pawn,
    Piece::Knight,
    Piece::Bishop,
    Piece::Rook,
    Piece::Queen,
    Piece::King,
}};

}  // namespace libchess

#endif
