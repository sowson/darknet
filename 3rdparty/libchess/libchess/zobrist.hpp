#ifndef LIBCHESS_ZOBRIST_HPP
#define LIBCHESS_ZOBRIST_HPP

#include <cstdint>
#include "piece.hpp"
#include "side.hpp"
#include "square.hpp"

namespace libchess::zobrist {

[[nodiscard]] std::uint64_t turn_key();

[[nodiscard]] std::uint64_t castling_key(const int t);

[[nodiscard]] std::uint64_t piece_key(const Piece p, const Side s, const Square sq);

[[nodiscard]] std::uint64_t ep_key(const Square sq);

}  // namespace libchess::zobrist

#endif
