#ifndef LIBCHESS_MOVEGEN_HPP
#define LIBCHESS_MOVEGEN_HPP

#include "bitboard.hpp"
#include "square.hpp"

namespace libchess::movegen {

Bitboard knight_moves(const Square sq);
Bitboard bishop_moves(const Square sq, const Bitboard &occ);
Bitboard rook_moves(const Square sq, const Bitboard &occ);
Bitboard queen_moves(const Square sq, const Bitboard &occ);
Bitboard king_moves(const Square sq);

}  // namespace libchess::movegen

#endif
