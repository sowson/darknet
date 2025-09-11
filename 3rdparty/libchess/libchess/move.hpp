#ifndef LIBCHESS_MOVE_HPP
#define LIBCHESS_MOVE_HPP

#include <cassert>
#include <cstdint>
#include <ostream>
#include "piece.hpp"
#include "square.hpp"

namespace libchess {

/*  Int packing:
 *  +6  6 - From
 *  +6 12 - To
 *  +3 15 - Type
 *  +3 18 - Piece
 *  +3 21 - Captured Piece
 *  +3 24 - Promotion Piece
 */

enum MoveType : int
{
    Normal = 0,
    Capture,
    Double,
    enpassant,
    ksc,
    qsc,
    promo,
    promo_capture
};

class Move {
   public:
    [[nodiscard]] constexpr Move() = default;

    [[nodiscard]] constexpr Move(const MoveType _t,
                                 const Square _fr,
                                 const Square _to,
                                 const Piece _piece,
                                 const Piece _cap = Piece::None,
                                 const Piece _promotion = Piece::None) {
        data_ = static_cast<unsigned int>(_fr);
        data_ |= static_cast<unsigned int>(_to) << 6;
        data_ |= static_cast<unsigned int>(_t) << 12;
        data_ |= static_cast<unsigned int>(_piece) << 15;
        data_ |= static_cast<unsigned int>(_cap) << 18;
        data_ |= static_cast<unsigned int>(_promotion) << 21;

#ifndef NDEBUG
        assert(piece() != Piece::None);
        assert(to() != from());
        assert(_t == type());
        assert(_fr == from());
        assert(_to == to());
        assert(_piece == piece());
        assert(_cap == captured());
        assert(_promotion == promotion());

        switch (type()) {
            case MoveType::Normal:
                assert(captured() == Piece::None);
                assert(promotion() == Piece::None);
                break;
            case MoveType::Capture:
                assert(captured() != Piece::None);
                assert(captured() != Piece::King);
                assert(promotion() == Piece::None);
                break;
            case MoveType::Double:
                assert(piece() == Piece::Pawn);
                assert(captured() == Piece::None);
                assert(promotion() == Piece::None);
                break;
            case MoveType::enpassant:
                assert(piece() == Piece::Pawn);
                assert(captured() == Piece::Pawn);
                assert(promotion() == Piece::None);
                break;
            case MoveType::ksc:
                assert(piece() == Piece::King || piece() == Piece::Rook);
                assert(captured() == Piece::None);
                assert(promotion() == Piece::None);
                break;
            case MoveType::qsc:
                assert(piece() == Piece::King || piece() == Piece::Rook);
                assert(captured() == Piece::None);
                assert(promotion() == Piece::None);
                break;
            case MoveType::promo:
                assert(piece() == Piece::Pawn);
                assert(captured() == Piece::None);
                assert(promotion() != Piece::None);
                assert(promotion() != Piece::Pawn);
                assert(promotion() != Piece::King);
                break;
            case MoveType::promo_capture:
                assert(piece() == Piece::Pawn);
                assert(captured() != Piece::None);
                assert(captured() != Piece::Pawn);
                assert(captured() != Piece::King);
                assert(promotion() != Piece::None);
                assert(promotion() != Piece::Pawn);
                assert(promotion() != Piece::King);
                break;
            default:
                abort();
        }
#endif
    }

    [[nodiscard]] constexpr Square from() const noexcept {
        return Square(data_ & 0x3F);
    }

    [[nodiscard]] constexpr Square to() const noexcept {
        return Square((data_ >> 6) & 0x3F);
    }

    [[nodiscard]] constexpr MoveType type() const noexcept {
        return MoveType((data_ >> 12) & 0x7);
    }

    [[nodiscard]] constexpr Piece piece() const noexcept {
        return static_cast<Piece>((data_ >> 15) & 0x7);
    }

    [[nodiscard]] constexpr Piece captured() const noexcept {
        return static_cast<Piece>((data_ >> 18) & 0x7);
    }

    [[nodiscard]] constexpr Piece promotion() const noexcept {
        return static_cast<Piece>((data_ >> 21) & 0x7);
    }

    [[nodiscard]] constexpr operator bool() const noexcept {
        return data_;
    }

    [[nodiscard]] operator std::string() const noexcept {
        std::string str;
        str += static_cast<std::string>(from());
        str += static_cast<std::string>(to());
        if (promotion() != Piece::None) {
            const char asd[] = {'n', 'b', 'r', 'q'};
            str += asd[promotion() - 1];
        }
        return str;
    }

    [[nodiscard]] constexpr bool operator==(const Move &rhs) const noexcept {
        return data_ == rhs.data_;
    }

    [[nodiscard]] constexpr bool operator!=(const Move &rhs) const noexcept {
        return data_ != rhs.data_;
    }

    [[nodiscard]] constexpr bool is_capturing() const noexcept {
        return type() == MoveType::Capture || type() == MoveType::promo_capture || type() == MoveType::enpassant;
    }

    [[nodiscard]] constexpr bool is_promoting() const noexcept {
        return type() == MoveType::promo || type() == MoveType::promo_capture;
    }

   private:
    std::uint32_t data_ = 0;
};

inline std::ostream &operator<<(std::ostream &os, const Move &move) noexcept {
    os << static_cast<std::string>(move);
    return os;
}

static_assert(sizeof(Move) == sizeof(std::uint32_t));
static_assert(!Move(MoveType::Normal, squares::A2, squares::A3, Piece::Pawn).is_promoting());
static_assert(!Move(MoveType::Normal, squares::A2, squares::A3, Piece::Pawn).is_capturing());
static_assert(Move(MoveType::promo, squares::A7, squares::A8, Piece::Pawn, Piece::None, Piece::Queen).is_promoting());
static_assert(!Move(MoveType::promo, squares::A7, squares::A8, Piece::Pawn, Piece::None, Piece::Queen).is_capturing());

}  // namespace libchess

#endif
