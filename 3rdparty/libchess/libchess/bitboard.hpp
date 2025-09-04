#ifndef LIBCHESS_BITBOARD_HPP
#define LIBCHESS_BITBOARD_HPP

#include <array>
#include <bit>
#include <cassert>
#include <cstdint>
#include <ostream>
#include "square.hpp"

namespace libchess {

class BitboardIterator {
   public:
    [[nodiscard]] constexpr BitboardIterator(const std::uint64_t &data) : data_{data} {
    }

    [[nodiscard]] constexpr Square operator*() const noexcept {
        return static_cast<Square>(std::countr_zero(data_));
    }

    constexpr BitboardIterator operator++() noexcept {
        data_ &= data_ - 1;
        return *this;
    }

    [[nodiscard]] constexpr bool operator!=(const BitboardIterator &rhs) const noexcept {
        return data_ != rhs.data_;
    }

   private:
    std::uint64_t data_ = 0;
};

class Bitboard {
   public:
    [[nodiscard]] constexpr Bitboard() = default;

    [[nodiscard]] constexpr explicit Bitboard(const std::uint64_t mask) : mask_{mask} {
    }

    [[nodiscard]] constexpr explicit Bitboard(const Square sq) : mask_{1ULL << static_cast<int>(sq)} {
    }

    [[nodiscard]] constexpr bool get(const Square sq) const noexcept {
        return (mask_ >> static_cast<int>(sq)) & 1;
    }

    constexpr void set(const Square sq) noexcept {
        mask_ |= 1ULL << static_cast<int>(sq);
    }

    [[nodiscard]] constexpr int count() const noexcept {
        return std::popcount(mask_);
    }

    [[nodiscard]] constexpr Bitboard adjacent() const noexcept {
        return north() | south() | east() | west() | north().east() | north().west() | south().east() | south().west();
    }

    constexpr void clear() noexcept {
        mask_ = 0;
    }

    [[nodiscard]] constexpr BitboardIterator begin() const noexcept {
        return BitboardIterator{mask_};
    }

    [[nodiscard]] constexpr BitboardIterator end() const noexcept {
        return BitboardIterator{0};
    }

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return mask_;
    }

    [[nodiscard]] constexpr bool operator==(const Bitboard &rhs) const noexcept {
        return mask_ == rhs.mask_;
    }

    [[nodiscard]] constexpr bool operator!=(const Bitboard &rhs) const noexcept {
        return mask_ != rhs.mask_;
    }

    [[nodiscard]] constexpr Bitboard operator~() const noexcept {
        return Bitboard(~mask_);
    }

    [[nodiscard]] constexpr Bitboard operator&(const Bitboard &rhs) const noexcept {
        return Bitboard(mask_ & rhs.mask_);
    }

    [[nodiscard]] constexpr Bitboard operator|(const Bitboard &rhs) const noexcept {
        return Bitboard(mask_ | rhs.mask_);
    }

    [[nodiscard]] constexpr Bitboard operator^(const Bitboard &rhs) const noexcept {
        return Bitboard(mask_ ^ rhs.mask_);
    }

    [[nodiscard]] constexpr Bitboard operator&(const Square &rhs) const noexcept {
        return Bitboard(mask_ & (1ULL << static_cast<int>(rhs)));
    }

    [[nodiscard]] constexpr Bitboard operator|(const Square &rhs) const noexcept {
        return Bitboard(mask_ | (1ULL << static_cast<int>(rhs)));
    }

    [[nodiscard]] constexpr Bitboard operator^(const Square &rhs) const noexcept {
        return Bitboard(mask_ ^ (1ULL << static_cast<int>(rhs)));
    }

    Bitboard constexpr operator&=(const Bitboard &rhs) noexcept {
        mask_ &= rhs.mask_;
        return *this;
    }

    Bitboard constexpr operator|=(const Bitboard &rhs) noexcept {
        mask_ |= rhs.mask_;
        return *this;
    }

    Bitboard constexpr operator^=(const Bitboard &rhs) noexcept {
        mask_ ^= rhs.mask_;
        return *this;
    }

    Bitboard constexpr operator&=(const Square &rhs) noexcept {
        mask_ &= 1ULL << static_cast<int>(rhs);
        return *this;
    }

    Bitboard constexpr operator|=(const Square &rhs) noexcept {
        mask_ |= 1ULL << static_cast<int>(rhs);
        return *this;
    }

    Bitboard constexpr operator^=(const Square &rhs) noexcept {
        mask_ ^= 1ULL << static_cast<int>(rhs);
        return *this;
    }

    [[nodiscard]] constexpr Bitboard operator<<(const int n) const noexcept {
        return Bitboard(mask_ << n);
    }

    [[nodiscard]] constexpr Bitboard operator>>(const int n) const noexcept {
        return Bitboard(mask_ >> n);
    }

    [[nodiscard]] constexpr Square lsb() const noexcept {
        assert(mask_);
        return Square(std::countr_zero(mask_));
    }

    [[nodiscard]] constexpr Square hsb() const noexcept {
        assert(mask_);
        return Square(63 - std::countl_zero(mask_));
    }

    [[nodiscard]] constexpr std::uint64_t value() const noexcept {
        return mask_;
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return mask_ == 0;
    }

    [[nodiscard]] constexpr Bitboard north() const noexcept {
        return Bitboard(mask_ << 8);
    }

    [[nodiscard]] constexpr Bitboard south() const noexcept {
        return Bitboard(mask_ >> 8);
    }

    [[nodiscard]] constexpr Bitboard east() const noexcept {
        return Bitboard((mask_ << 1) & ~0x0101010101010101ULL);
    }

    [[nodiscard]] constexpr Bitboard west() const noexcept {
        return Bitboard((mask_ >> 1) & ~0x8080808080808080ULL);
    }

   private:
    std::uint64_t mask_ = 0;
};

inline std::ostream &operator<<(std::ostream &os, const Bitboard &bb) noexcept {
    int i = 56;
    while (i >= 0) {
        const auto sq = Square(i);
        if (bb & Bitboard{sq}) {
            os << '1';
        } else {
            os << '0';
        }

        if (i % 8 == 7) {
            os << '\n';
            i -= 16;
        }

        i++;
    }

    return os;
}

constexpr std::array<std::array<Bitboard, 64>, 64> calculate_squares_between() {
    std::array<std::array<Bitboard, 64>, 64> result;

    for (int i = 0; i < 64; ++i) {
        for (int j = 0; j < 64; ++j) {
            auto sq1 = Square{i};
            auto sq2 = Square{j};

            const auto dx = (sq2.file() - sq1.file());
            const auto dy = (sq2.rank() - sq1.rank());
            const auto adx = dx > 0 ? dx : -dx;
            const auto ady = dy > 0 ? dy : -dy;

            if (dx == 0 || dy == 0 || adx == ady) {
                Bitboard mask;
                while (sq1 != sq2) {
                    if (dx > 0) {
                        sq1 = sq1.east();
                    } else if (dx < 0) {
                        sq1 = sq1.west();
                    }
                    if (dy > 0) {
                        sq1 = sq1.north();
                    } else if (dy < 0) {
                        sq1 = sq1.south();
                    }
                    mask |= Bitboard{sq1};
                }
                result[i][j] = mask & ~Bitboard{sq2};
            }
        }
    }

    return result;
}

constexpr auto lut_squares_between = calculate_squares_between();

[[nodiscard]] constexpr Bitboard squares_between(const Square sq1, const Square sq2) {
    return lut_squares_between[static_cast<int>(sq1)][static_cast<int>(sq2)];
}

namespace bitboards {

constexpr auto FileA = Bitboard(0x0101010101010101);
constexpr auto FileB = Bitboard(0x0202020202020202);
constexpr auto FileC = Bitboard(0x0404040404040404);
constexpr auto FileD = Bitboard(0x0808080808080808);
constexpr auto FileE = Bitboard(0x1010101010101010);
constexpr auto FileF = Bitboard(0x2020202020202020);
constexpr auto FileG = Bitboard(0x4040404040404040);
constexpr auto FileH = Bitboard(0x8080808080808080);

constexpr auto Rank1 = Bitboard(0x00000000000000ff);
constexpr auto Rank2 = Bitboard(0x000000000000ff00);
constexpr auto Rank3 = Bitboard(0x0000000000ff0000);
constexpr auto Rank4 = Bitboard(0x00000000ff000000);
constexpr auto Rank5 = Bitboard(0x000000ff00000000);
constexpr auto Rank6 = Bitboard(0x0000ff0000000000);
constexpr auto Rank7 = Bitboard(0x00ff000000000000);
constexpr auto Rank8 = Bitboard(0xff00000000000000);

constexpr Bitboard files[] = {FileA, FileB, FileC, FileD, FileE, FileF, FileG, FileH};
constexpr Bitboard ranks[] = {Rank1, Rank2, Rank3, Rank4, Rank5, Rank6, Rank7, Rank8};
constexpr Bitboard adjacent_files[] =
    {FileB, FileA | FileC, FileB | FileD, FileC | FileE, FileD | FileF, FileE | FileG, FileF | FileH, FileG};

constexpr auto LightSquares = Bitboard(0x55aa55aa55aa55aa);
constexpr auto DarkSquares = Bitboard(0xaa55aa55aa55aa55);
constexpr auto Empty = Bitboard(0x0000000000000000);
constexpr auto AllSquares = Bitboard(0xffffffffffffffff);
constexpr auto Edge = Bitboard(0xff818181818181ff);

static_assert(Empty.count() == 0);
static_assert(Empty.empty());
static_assert(Empty.adjacent() == Empty);
static_assert(AllSquares.count() == 64);
static_assert(!AllSquares.empty());
static_assert(AllSquares.adjacent() == AllSquares);
static_assert(LightSquares.count() == 32);
static_assert(DarkSquares.count() == 32);
static_assert((~Empty) == AllSquares);
static_assert(~AllSquares == Empty);
static_assert((LightSquares & DarkSquares) == Empty);
static_assert((LightSquares | DarkSquares) == AllSquares);
static_assert(~LightSquares == DarkSquares);
static_assert(~DarkSquares == LightSquares);
static_assert(Edge == (FileA | FileH | Rank1 | Rank8));
static_assert(Rank1.north() == Rank2);
static_assert(Rank2.north() == Rank3);
static_assert(Rank3.north() == Rank4);
static_assert(Rank4.north() == Rank5);
static_assert(Rank5.north() == Rank6);
static_assert(Rank6.north() == Rank7);
static_assert(Rank7.north() == Rank8);
static_assert(Rank8.north() == Empty);
static_assert(Rank1.south() == Empty);
static_assert(Rank2.south() == Rank1);
static_assert(Rank3.south() == Rank2);
static_assert(Rank4.south() == Rank3);
static_assert(Rank5.south() == Rank4);
static_assert(Rank6.south() == Rank5);
static_assert(Rank7.south() == Rank6);
static_assert(Rank8.south() == Rank7);
static_assert(FileA.east() == FileB);
static_assert(FileB.east() == FileC);
static_assert(FileC.east() == FileD);
static_assert(FileD.east() == FileE);
static_assert(FileE.east() == FileF);
static_assert(FileF.east() == FileG);
static_assert(FileG.east() == FileH);
static_assert(FileH.east() == Empty);
static_assert(FileA.west() == Empty);
static_assert(FileB.west() == FileA);
static_assert(FileC.west() == FileB);
static_assert(FileD.west() == FileC);
static_assert(FileE.west() == FileD);
static_assert(FileF.west() == FileE);
static_assert(FileG.west() == FileF);
static_assert(FileH.west() == FileG);
static_assert(FileA.adjacent() == (FileA | FileB));
static_assert(FileB.adjacent() == (FileA | FileB | FileC));
static_assert(FileC.adjacent() == (FileB | FileC | FileD));
static_assert(FileD.adjacent() == (FileC | FileD | FileE));
static_assert(FileE.adjacent() == (FileD | FileE | FileF));
static_assert(FileF.adjacent() == (FileE | FileF | FileG));
static_assert(FileG.adjacent() == (FileF | FileG | FileH));
static_assert(FileH.adjacent() == (FileG | FileH));
static_assert(Rank1.adjacent() == (Rank1 | Rank2));
static_assert(Rank2.adjacent() == (Rank1 | Rank2 | Rank3));
static_assert(Rank3.adjacent() == (Rank2 | Rank3 | Rank4));
static_assert(Rank4.adjacent() == (Rank3 | Rank4 | Rank5));
static_assert(Rank5.adjacent() == (Rank4 | Rank5 | Rank6));
static_assert(Rank6.adjacent() == (Rank5 | Rank6 | Rank7));
static_assert(Rank7.adjacent() == (Rank6 | Rank7 | Rank8));
static_assert(Rank8.adjacent() == (Rank7 | Rank8));
static_assert(Bitboard{squares::A1}.adjacent() == Bitboard{0x302});
static_assert(Bitboard{squares::H1}.adjacent() == Bitboard{0xc040});
static_assert(Bitboard{squares::A8}.adjacent() == Bitboard{0x203000000000000});
static_assert(Bitboard{squares::H8}.adjacent() == Bitboard{0x40c0000000000000});
static_assert(Bitboard{squares::D4}.adjacent() == Bitboard{0x1c141c0000});
static_assert(Bitboard{squares::A4}.lsb() == squares::A4);
static_assert(Bitboard{squares::D4}.lsb() == squares::D4);
static_assert(Bitboard{squares::H8}.lsb() == squares::H8);
static_assert(Bitboard{0x1c141c0000}.lsb() == squares::C3);

}  // namespace bitboards

static_assert(sizeof(Bitboard) == sizeof(std::uint64_t));
// Same file
static_assert(squares_between(squares::A1, squares::A4) == Bitboard(0x10100));
static_assert(squares_between(squares::A4, squares::A1) == Bitboard(0x10100));
// Same Rank
static_assert(squares_between(squares::A1, squares::D1) == Bitboard(0x6));
static_assert(squares_between(squares::D1, squares::A1) == Bitboard(0x6));
// Same diagonal a1-h8
static_assert(squares_between(squares::A1, squares::D4) == Bitboard(0x40200));
static_assert(squares_between(squares::D4, squares::A1) == Bitboard(0x40200));
// Same diagonal a8-h1
static_assert(squares_between(squares::A4, squares::D1) == Bitboard(0x20400));
static_assert(squares_between(squares::D1, squares::A4) == Bitboard(0x20400));
// Hmm
static_assert(squares_between(squares::A1, squares::A1).empty());
static_assert(squares_between(squares::A1, squares::B1).empty());
static_assert(squares_between(squares::A1, squares::B2).empty());
static_assert(squares_between(squares::A1, squares::A2).empty());
static_assert(squares_between(squares::A1, squares::B3).empty());
static_assert(squares_between(squares::A1, squares::C2).empty());
static_assert(squares_between(squares::B3, squares::A1).empty());
static_assert(squares_between(squares::B3, squares::A1).empty());

}  // namespace libchess

#endif
