#ifndef LIBCHESS_SQUARE_HPP
#define LIBCHESS_SQUARE_HPP

#include <cassert>
#include <cstdint>
#include <ostream>
#include <string>

namespace libchess {

class Square {
   public:
    [[nodiscard]] constexpr Square() = default;

    [[nodiscard]] constexpr explicit Square(const int n) : data_{static_cast<std::uint8_t>(n)} {
        assert(n < 64);
    }

    [[nodiscard]] constexpr Square(const int f, const int r) : data_{static_cast<std::uint8_t>(8 * r + f)} {
        assert(data_ < 64);
    }

    [[nodiscard]] Square(const std::string &str) : data_{} {
        const int file = str[0] - 'a';
        const int rank = str[1] - '1';
        data_ = static_cast<std::uint8_t>(rank * 8 + file);
    }

    [[nodiscard]] constexpr bool operator==(const Square &rhs) const noexcept {
        return data_ == rhs.data_;
    }

    [[nodiscard]] constexpr bool operator!=(const Square &rhs) const noexcept {
        return data_ != rhs.data_;
    }

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return data_ != 0xFF;
    }

    [[nodiscard]] constexpr explicit operator int() const noexcept {
        return data_;
    }

    [[nodiscard]] explicit operator std::string() const noexcept {
        std::string str;
        str += static_cast<char>('a' + file());
        str += static_cast<char>('1' + rank());
        return str;
    }

    [[nodiscard]] constexpr explicit operator unsigned int() const noexcept {
        return data_;
    }

    [[nodiscard]] constexpr int rank() const noexcept {
        return data_ / 8;
    }

    [[nodiscard]] constexpr int file() const noexcept {
        return data_ % 8;
    }

    [[nodiscard]] constexpr bool light() const noexcept {
        return (rank() + file()) % 2;
    }

    [[nodiscard]] constexpr bool dark() const noexcept {
        return !light();
    }

    [[nodiscard]] constexpr Square flip() const noexcept {
        return Square(data_ ^ 56);
    }

    [[nodiscard]] constexpr Square north() const noexcept {
        return Square(data_ + 8);
    }

    [[nodiscard]] constexpr Square south() const noexcept {
        return Square(data_ - 8);
    }

    [[nodiscard]] constexpr Square east() const noexcept {
        return Square(data_ + 1);
    }

    [[nodiscard]] constexpr Square west() const noexcept {
        return Square(data_ - 1);
    }

   private:
    std::uint8_t data_ = 0xFF;
};

inline std::ostream &operator<<(std::ostream &os, const Square &sq) noexcept {
    os << static_cast<std::string>(sq);
    return os;
}

// clang-format off
const std::string square_strings[] = {
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
};
// clang-format on

namespace squares {

constexpr auto A1 = Square(0);
constexpr auto B1 = Square(1);
constexpr auto C1 = Square(2);
constexpr auto D1 = Square(3);
constexpr auto E1 = Square(4);
constexpr auto F1 = Square(5);
constexpr auto G1 = Square(6);
constexpr auto H1 = Square(7);

constexpr auto A2 = Square(8);
constexpr auto B2 = Square(9);
constexpr auto C2 = Square(10);
constexpr auto D2 = Square(11);

constexpr auto A3 = Square(16);
constexpr auto B3 = Square(17);
constexpr auto C3 = Square(18);
constexpr auto D3 = Square(19);
constexpr auto E3 = Square(20);
constexpr auto F3 = Square(21);
constexpr auto G3 = Square(22);
constexpr auto H3 = Square(23);

constexpr auto A4 = Square(24);
constexpr auto D4 = Square(27);

constexpr auto A5 = Square(32);
constexpr auto B5 = Square(33);
constexpr auto C5 = Square(34);
constexpr auto D5 = Square(35);
constexpr auto E5 = Square(36);
constexpr auto F5 = Square(37);
constexpr auto G5 = Square(38);

constexpr auto A6 = Square(40);
constexpr auto H6 = Square(47);

constexpr auto A7 = Square(48);
constexpr auto B7 = Square(49);
constexpr auto F7 = Square(53);
constexpr auto G7 = Square(54);

constexpr auto A8 = Square(56);
constexpr auto B8 = Square(57);
constexpr auto C8 = Square(58);
constexpr auto D8 = Square(59);
constexpr auto E8 = Square(60);
constexpr auto F8 = Square(61);
constexpr auto G8 = Square(62);
constexpr auto H8 = Square(63);

constexpr Square OffSq;

static_assert(A1.dark());
static_assert(!A1.light());
static_assert(!H1.dark());
static_assert(H1.light());
static_assert(A1.flip() == A8);
static_assert(A8.flip() == A1);
static_assert(H1.flip() == H8);
static_assert(H8.flip() == H1);
static_assert(OffSq == Square());
// static_assert(Square("a1") == A1);

}  // namespace squares

}  // namespace libchess

#endif
