#ifndef LIBCHESS_SIDE_HPP
#define LIBCHESS_SIDE_HPP

#include <array>

namespace libchess {

enum Side : bool
{
    White = 0,
    Black
};

inline constexpr Side operator!(Side s) {
    return static_cast<Side>(!static_cast<bool>(s));
}

inline constexpr std::array<Side, 2> sides = {{
    Side::White,
    Side::Black,
}};

}  // namespace libchess

#endif
