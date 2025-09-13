#include <iostream>

#include <fstream>
#include <nlohmann/json.hpp>

#include "move.hpp"
#include "position.hpp"

#include <cstring>

#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <sstream>
#include <random>
#include <cfloat>
#include <unistd.h>

extern "C" {
extern void *tracked_calloc(size_t count, size_t size, const char *file, int line);
extern void tracked_free(void *ptr, const char *file, int line);
}

#include "system.h"

using json = nlohmann::json;

#include <deque>
#include <unordered_set>
#include <utility>

#include <string>
#include <algorithm>
#include <cctype>

std::string to_lowercase(const std::string& s) {
    std::string result = s;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

std::vector<libchess::Move> ch_legal_moves(const libchess::Position& sfen, libchess::Position& pos);

constexpr int MAX_CACHE_SIZE = 1000;

struct FenPair {
    std::string sfen_fen;
    std::string pos_fen;

    bool operator==(const FenPair& other) const {
        return sfen_fen == other.sfen_fen && pos_fen == other.pos_fen;
    }
};

struct FenPairHash {
    std::size_t operator()(const FenPair& p) const {
        return std::hash<std::string>()(p.sfen_fen) ^ (std::hash<std::string>()(p.pos_fen) << 1);
    }
};

class FenCache {
private:
    std::deque<FenPair> order;
    std::unordered_map<FenPair, float, FenPairHash> cache;

public:
    bool exists(const std::string& sfen, const std::string& pos) {
        return cache.find(FenPair{sfen, pos}) != cache.end();
    }

    float get(const std::string& sfen, const std::string& pos) {
        return cache.at(FenPair{sfen, pos});
    }

    void insert(const std::string& sfen, const std::string& pos, float value) {
        FenPair key{sfen, pos};
        if (cache.find(key) != cache.end()) return;

        if (order.size() >= MAX_CACHE_SIZE) {
            cache.erase(order.front());
            order.pop_front();
        }

        order.push_back(key);
        cache[key] = value;
    }
};

#define WHITE_PAWN     (+0.10f)
#define BLACK_PAWN     (-0.10f)
#define WHITE_BISHOP   (+0.30f)
#define BLACK_BISHOP   (-0.30f)
#define WHITE_KNIGHT   (+0.40f)
#define BLACK_KNIGHT   (-0.40f)
#define WHITE_ROOK     (+0.50f)
#define BLACK_ROOK     (-0.50f)
#define WHITE_QUEEN    (+0.90f)
#define BLACK_QUEEN    (-0.90f)
#define WHITE_KING     (+1.00f)
#define BLACK_KING     (-1.00f)
#define EMPTY          (0.000f)

static int evaluate_board_prepared = 0;
static std::map<libchess::Piece, float> weights;
static std::map<libchess::Side, std::map<libchess::Piece, std::vector<std::vector<float>>>> pst;

void evaluate_board_prepare_pst() {
    static std::map<libchess::Piece, std::vector<std::vector<float>>> pst_w;
    static std::map<libchess::Piece, std::vector<std::vector<float>>> pst_b;

    weights[libchess::None]   =   0;
    weights[libchess::Pawn]   =  10;
    weights[libchess::Bishop] =  30;
    weights[libchess::Knight] =  30;
    weights[libchess::Rook]   =  50;
    weights[libchess::Queen]  =  90;
    weights[libchess::King]   =   2;

    pst_w[libchess::Pawn] = {
            {+6.5, +6.5, +6.5, +6.5, +6.5, +6.5, +6.5, +6.5},
            {+5.5, +5.5, +5.5, +5.5, +5.5, +5.5, +5.5, +5.5},
            {+2.5, +2.5, +3.5, +4.5, +4.5, +3.5, +2.5, +2.5},
            {+2.0, +2.0, +2.5, +7.0, +7.0, +2.5, +2.0, +2.0},
            {+1.5, +1.5, +1.5, +7.5, +7.5, +1.5, +1.5, +1.5},
            {+2.0, +1.0, +0.5, +4.5, +4.5, +0.5, +1.0, +2.0},
            {+2.0, +2.5, +2.5, +1.5, +1.5, +2.5, +2.5, +2.0},
            {+1.5, +1.5, +1.5, +1.5, +1.5, +1.5, +1.5, +1.5},
    };

    pst_b[libchess::Pawn] = std::vector<std::vector<float>>(pst_w[libchess::Pawn].rbegin(), pst_w[libchess::Pawn].rend());

    pst_w[libchess::Knight] = {
            {-2.5, -1.5, -0.5, -0.5, -0.5, -0.5, -1.5, -2.5},
            {-1.5, +0.5, +2.5, +2.5, +2.5, +2.5, +0.5, -1.5},
            {-0.5, +2.5, +3.5, +4.0, +4.0, +3.5, +2.5, -0.5},
            {-0.5, +3.0, +4.0, +4.5, +4.5, +4.0, +3.0, -0.5},
            {-0.5, +2.5, +4.0, +4.5, +4.5, +4.0, +2.5, -0.5},
            {-0.5, +3.0, +1.5, +4.0, +4.0, +1.5, +3.0, -0.5},
            {-1.5, +0.5, +2.5, +3.0, +3.0, +2.5, +0.5, -1.5},
            {-2.5, -1.5, -0.5, -0.5, -0.5, -0.5, -1.5, -2.5},
    };

    pst_b[libchess::Knight] = std::vector<std::vector<float>>(pst_w[libchess::Knight].rbegin(), pst_w[libchess::Knight].rend());

    pst_w[libchess::Bishop] = {
            { -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0},
            { -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0},
            { -1.0,  0.0,  0.5,  1.0,  1.0,  0.5,  0.0, -1.0},
            { -1.0,  0.5,  0.5,  1.0,  1.0,  0.5,  0.5, -1.0},
            { -1.0,  0.0,  1.0,  1.0,  1.0,  1.0,  0.0, -1.0},
            { -1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0, -1.0},
            { -1.0,  0.5,  0.0,  0.0,  0.0,  0.0,  0.5, -1.0},
            { -2.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0},
    };

    pst_b[libchess::Bishop] = std::vector<std::vector<float>>(pst_w[libchess::Bishop].rbegin(), pst_w[libchess::Bishop].rend());

    pst_w[libchess::Rook] = {
            {  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0},
            {  0.5,  1.0,  1.0,  1.0,  1.0,  1.0,  1.0,  0.5},
            { -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5},
            { -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5},
            { -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5},
            { -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5},
            { -0.5,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -0.5},
            {  0.0,   0.0, 0.0,  0.5,  0.5,  0.0,  0.0,  0.0},
    };

    pst_b[libchess::Rook] = std::vector<std::vector<float>>(pst_w[libchess::Rook].rbegin(), pst_w[libchess::Rook].rend());

    pst_w[libchess::Queen] = {
            { -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0},
            { -1.0,  0.0,  0.0,  0.0,  0.0,  0.0,  0.0, -1.0},
            { -1.0,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0},
            { -0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5},
            { -0.5,  0.0,  0.5,  0.5,  0.5,  0.5,  0.0, -0.5},
            { -1.0,  0.5,  0.5,  0.5,  0.5,  0.5,  0.0, -1.0},
            { -1.0,  0.0,  0.5,  0.0,  0.0,  0.0,  0.0, -1.0},
            { -2.0, -1.0, -1.0, -0.5, -0.5, -1.0, -1.0, -2.0},
    };

    pst_b[libchess::Queen] = std::vector<std::vector<float>>(pst_w[libchess::Queen].rbegin(), pst_w[libchess::Queen].rend());

    pst_w[libchess::King] = {
            { -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0},
            { -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0},
            { -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0},
            { -3.0, -4.0, -4.0, -5.0, -5.0, -4.0, -4.0, -3.0},
            { -2.0, -3.0, -3.0, -4.0, -4.0, -3.0, -3.0, -2.0},
            { -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0},
            {  2.0,  2.0,  0.0,  0.0,  0.0,  0.0,  2.0,  2.0},
            {  2.0,  2.0,  3.0,  0.0,  0.0,  1.0,  3.0,  2.0},
    };

    pst_b[libchess::King] = std::vector<std::vector<float>>(pst_w[libchess::King].rbegin(), pst_w[libchess::King].rend());

    pst[libchess::White] = pst_w;
    pst[libchess::Black] = pst_b;

    evaluate_board_prepared = 1;
}

float ch_evaluate_board(const libchess::Position& sfen, const libchess::Position& game, libchess::Side color, float* powW = 0, float* powB = 0) {

    float sumW = 0;
    float sumB = 0;

    if (game.is_checkmate() || game.is_draw()) {
        if (game.is_checkmate()) {
            sumW += color == libchess::White ? +10.0 : 0.0;
            sumB += color == libchess::Black ? +10.0 : 0.0;
        }
        if (game.is_draw()) {
            sumW += color == libchess::White ? +1.000 : 0.0;
            sumB += color == libchess::Black ? +1.000 : 0.0;
        }

        if (powW != nullptr) *powW = sumW;
        if (powB != nullptr) *powB = sumB;

        return (color == libchess::White ? sumW - sumB : sumB - sumW);
    }

    if (evaluate_board_prepared == 0) evaluate_board_prepare_pst();

    int pst960map[8] = {0,1,2,3,4,5,6,7};

    if (true) {
        std::string sfen_c = sfen.get_fen();

        int rc=0;
        int rn=0;
        int rb=0;
        for (int i = 0; i < 8; ++i) {
            switch (sfen_c[i]) {
                case 'r': pst960map[i] = i % 2 == 0 && rc++ == 0 ? 0 : 7; break;
                case 'n': pst960map[i] = i % 2 == 1 && rn++ == 0 ? 1 : 6; break;
                case 'b': pst960map[i] = i % 2 == 0 && rb++ == 0 ? 2 : 5; break;
                case 'q': pst960map[i] = 3; break;
                case 'k': pst960map[i] = 4; break;
                default: break;
            }
        }
    }

    float attW = 0;
    float ptcW = 0;
    float picW = 0;
    float attB = 0;
    float ptcB = 0;
    float picB = 0;

    for (const libchess::Side side : libchess::sides) {
        for (const libchess::Piece piece: libchess::pieces) {
            for (const libchess::Square square: game.pieces(side, piece)) {
                if (game.piece_on(square) == libchess::None) continue;
                const int i = square.file();
                const int j = square.rank();
                if (side == libchess::White) {
                    attW += game.attackers(square, libchess::Black).count();
                    ptcW += game.attackers(square, libchess::White).count();
                    sumW += ((weights[piece] + pst[side][piece][i][pst960map[j]]));
                    picW += weights[piece];
                } else {
                    attB += game.attackers(square, libchess::White).count();
                    ptcB += game.attackers(square, libchess::Black).count();
                    sumB += ((weights[piece] + pst[side][piece][i][pst960map[j]]));
                    picB += weights[piece];
                }
            }
        }
    }

    sumW += (attW - ptcW) / (picW) + 2.0;
    sumB += (attB - ptcB) / (picB) + 2.0;

    sumW /= 500.0f;
    sumB /= 500.0f;

    if (powW != nullptr) *powW = sumW;
    if (powB != nullptr) *powB = sumB;

    return (color == libchess::White ? sumW - sumB : sumB - sumW) / 100.0f;
}

std::random_device rd;
std::mt19937 generator(rd());

float ch_min_max(const libchess::Position& sfen, libchess::Position& game, int depth, float alpha, float beta, bool is_max_p, libchess::Side color, int *counter) {
    *counter += 1;

    if (!game.valid() || game.is_terminal()) {
        return game.is_checkmate() ? 10000000.f : 0.f;
    }

    float best_move_max = -10000000.f;
    float best_move_min = +10000000.f;

    if (depth <= 0 || game.is_terminal()) {
        return ch_evaluate_board(sfen, game, color);
    }

    std::vector<libchess::Move> possible_next_moves = game.legal_moves();
    std::shuffle(possible_next_moves.begin(), possible_next_moves.end(), generator);

    int num_possible_moves_count = possible_next_moves.size();

    int step = 1; // depth > 2 ? 2 : 1;

    if (is_max_p) {
        for (int i = 0; i < num_possible_moves_count; i++) {
            game.makemove(possible_next_moves[i]);
            best_move_max = std::max(best_move_max, ch_min_max(sfen, game, depth - step, alpha, beta, !is_max_p, color, counter));
            game.undomove();
            alpha = std::max(alpha, best_move_max);
            if (alpha >= beta) {
                return best_move_max;
            }
        }
        return best_move_max;
    } else {
        for (int i = 0; i < num_possible_moves_count; i++) {
            game.makemove(possible_next_moves[i]);
            best_move_min = std::min(best_move_min, ch_min_max(sfen, game, depth - step, alpha, beta, !is_max_p, color, counter));
            game.undomove();
            beta = std::min(beta, best_move_min);
            if (alpha >= beta) {
                return best_move_min;
            }
        }
        return best_move_min;
    }
}

float ch_softmax(float *input, int n, float temp, int stride, float *output) {
    int i;
    int il;
    float sum = 0.0f;
    float largest = -FLT_MAX;
    for (i = 0; i < n; ++i) {
        float value = input[i * stride];
        input[i * stride] = (value != value) ? value : 1.0f / (float)n;
        if (input[i * stride] > largest) { largest = input[i * stride]; il=i; }
    }
    for (i = 0; i < n; ++i) {
        float scaled_value = (input[i * stride] - largest) / temp;
        float exp_value = exp(scaled_value);
        output[i * stride] = (exp_value != exp_value) ? exp_value : 1.0f / (float)n;
        sum += output[i * stride];
    }
    for (i = 0; i < n; ++i) {
        output[i * stride] = output[i * stride] / sum;
    }
    return output[il];
}

void ch_softmax_revert(const float* softmax_probs, size_t size, float z_max, float* logits) {
    if (softmax_probs == nullptr || logits == nullptr || size == 0) {
        return;
    }

    float ln_sum_exp_z = z_max < 0 ? -z_max : z_max;

    float min_logit = FLT_MAX;
    for (size_t i = 0; i < size; i++) {
        if (softmax_probs[i] > 0.0f) {
            logits[i] = logf(softmax_probs[i]) + ln_sum_exp_z;
            if (logits[i] < min_logit) {
                min_logit = logits[i];
            }
        } else {
            logits[i] = -FLT_MAX;
        }
    }

    if (z_max < 0) {
        float offset = fabsf(min_logit);
        for (size_t i = 0; i < size; i++) {
            if (logits[i] > -FLT_MAX) {
                logits[i] -= offset;
            }
        }
    }
}

void ch_softmax_scaled(const float *values, int size, float *output) {
    float max_value = values[0];
    float sum = 0.0;

    for (int i = 1; i < size; i++) {
        if (values[i] > max_value) {
            max_value = values[i];
        }
    }

    for (int i = 0; i < size; i++) {
        output[i] = exp(values[i] - max_value);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] = (2 * (output[i] / sum)) - 1;
    }
}

void ch_soft_tanh(const float *values, int size, float *output) {
    float sum = 0.f;

    for (int i = 0; i < size; i++) {
        output[i] = tanhf(values[i]);
        sum += fabsf(output[i]);
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

static FenCache recent_fens;

std::vector<libchess::Move> ch_calculate_top_n_moves(const int n, const libchess::Position& sfen, libchess::Position& game, int depth) {
    if (game.is_terminal() || !game.valid()) {
        return {};
    }

    auto color = game.turn();

    std::vector<libchess::Move> moves = game.legal_moves();
    int count = moves.size();

    if (count == 0) {
        return {};
    }

    const int top_n = std::min(n, count);
    std::string game_fen = game.get_fen();

    std::vector<std::pair<float, int>> scored_indices;
    scored_indices.reserve(count);

    for (int i = 0; i < count; ++i) {
        game.makemove(moves[i]);

        if (game.is_checkmate()) {
            count = 1;
            return {moves[i]};
        }

        std::string pos_fen = game.get_fen();
        float return_value;

        if (recent_fens.exists(game_fen, pos_fen)) {
            return_value = recent_fens.get(game_fen, pos_fen);
        } else {
            int counter = 0;
            return_value = ch_min_max(sfen, game, depth, -10000001.f, +10000001.f, false, color, &counter);
            recent_fens.insert(game_fen, pos_fen, return_value);
        }

        game.undomove();
        scored_indices.emplace_back(return_value, i);
    }

    std::sort(
            scored_indices.begin(),
            scored_indices.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; } // TODO!!
    );

    std::vector<libchess::Move> top_moves;
    for (int i = 0; i < top_n; ++i) {
        top_moves.emplace_back(moves[scored_indices[i].second]);
    }

    return top_moves;
}

std::vector<libchess::Move> ch_legal_moves(const libchess::Position& sfen, libchess::Position& pos) {
    //return pos.legal_moves();
    int n = 256;
    int d = 2;
    return ch_calculate_top_n_moves(n, sfen, pos, d);
}

int ch_calculate_best_move(const libchess::Position& sfen, libchess::Position& game, int depth, float &best_move, float *&best_move_values, int *counter) {
    std::vector<libchess::Move> moves = ch_legal_moves(sfen, game);
    *counter = moves.size();

    float best_move_max = -10000000.f;
    int best_move_idx = -1;
    auto color = game.turn();

    best_move_values = (float*)CALLOC(moves.size(), sizeof(float));

    for (int i = 0; i < moves.size(); ++i) {
        game.makemove(moves[i]);
        float return_value = ch_min_max(sfen, game, depth, -10000001.f, +10000001.f, false, color, counter);
        game.undomove();
        if (return_value > best_move_max) {
            best_move_max = return_value;
            best_move_values[i] = return_value;
            best_move = return_value;
            best_move_idx = i;
        }
    }

    return best_move_idx;
}

std::string ch_generate_deterministic_960(int seed) {
    std::array<char, 8> pieces = {' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '};
    int bishopIndex1 = (seed % 4) * 2;
    int bishopIndex2 = (seed / 4 % 4) * 2 + 1;
    pieces[bishopIndex1] = 'B';
    pieces[bishopIndex2] = 'B';

    int queenIndex = (seed / 16) % 6, count = 0;
    for (int i = 0; i < 8; i++) {
        if (pieces[i] == ' ') {
            if (count == queenIndex) {
                pieces[i] = 'Q';
                break;
            }
            count++;
        }
    }

    std::vector<int> availableSquares;
    for (int i = 0; i < 8; i++)
        if (pieces[i] == ' ') availableSquares.push_back(i);

    int knightIndex1 = availableSquares[(seed / 96) % 5];
    std::vector<int> validKnightSquares;
    for (int i : availableSquares)
        if (i % 2 != knightIndex1 % 2) validKnightSquares.push_back(i);

    int knightIndex2 = validKnightSquares[(seed / 480) % validKnightSquares.size()];
    pieces[knightIndex1] = 'N';
    pieces[knightIndex2] = 'N';

    std::vector<int> emptyIndices;
    for (int i = 0; i < 8; i++)
        if (pieces[i] == ' ') emptyIndices.push_back(i);

    pieces[emptyIndices[0]] = 'R';
    pieces[emptyIndices[1]] = 'K';
    pieces[emptyIndices[2]] = 'R';

    return std::string(pieces.begin(), pieces.end());
}

extern "C" {

int ch_eval_best_trivial_move(const char* sfen, const char* valid_fen, int level, float &best_move, float *&best_moves, int *counter) {
    const char* startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position fen = libchess::Position(valid_fen);
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);
    return ch_calculate_best_move(start, fen, level, best_move, best_moves, counter);
}

float ch_eval_the_move(const char* sfen, const char* valid_prev_fen, const char* valid_fen) {
    const char* startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);
    libchess::Position fen = libchess::Position(valid_prev_fen);
    libchess::Position mfen = libchess::Position(valid_fen);
    float start_move = (float)ch_evaluate_board(start, fen, fen.turn());
    return ch_evaluate_board(start, mfen, mfen.turn()) - start_move;
}

int ch_is_legal(const char* sfen, const char* fen, int indext) {
    const char *startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);
    libchess::Position pos = libchess::Position(fen);
    auto moves = ch_legal_moves(start, pos);
    return (indext >= 0 && indext < moves.size());
}

char* ch_do_legal(const char* sfen, const char* fen, int indext) {
    const char *startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);
    libchess::Position pos = libchess::Position(fen);
    if (pos.is_terminal() || !pos.valid()) return nullptr;
    auto moves = ch_legal_moves(start, pos);
    if (moves.size() == 0) return nullptr;
    pos.makemove(moves[indext]);
    std::string next_fen = pos.get_fen();
    char* valid_fen = (char *) CALLOC(next_fen.length() + 1, sizeof(char));
    strcpy(valid_fen, next_fen.c_str());
    return valid_fen;
}

int ch_fopen(char *jsonf, char *&sessionId, char *&fen, char *&move, char *&level, char *&sfen) {
    usleep(250);

    std::ifstream r = std::ifstream(jsonf);
    if (!r.good()) return 0;

    json data = json::parse(r);

    std::string sessionIdr;
    if (data.contains("sessionId") && !data["sessionId"].is_null()) {
        sessionIdr = std::string(data["sessionId"]);
        sessionId = (char *) CALLOC(sessionIdr.length() + 1, sizeof(char));
        strcpy(sessionId, sessionIdr.c_str());
    }
    std::string pgnr;
    if (data.contains("fen") && !data["fen"].is_null()) {
        pgnr = std::string(data["fen"]);
        fen = (char *) CALLOC(pgnr.length() + 1, sizeof(char));
        strcpy(fen, pgnr.c_str());
    }
    std::string mover;
    if (data.contains("pgn") && !data["pgn"].is_null()) {
        mover = std::string(data["pgn"]);
        move = (char *) CALLOC(mover.length() + 1, sizeof(char));
        strcpy(move, mover.c_str());
    }
    std::string levelr;
    if (data.contains("level") && !data["level"].is_null()) {
        levelr = std::string(data["level"]);
        level = (char *) CALLOC(levelr.length() + 1, sizeof(char));
        strcpy(level, levelr.c_str());
    } else {
        level = nullptr;
    }

    if (data.contains("sfen") && !data["sfen"].is_null() && std::string(data["sfen"]).length() != 2) {
        auto sfenr = std::string(data["sfen"]);
        sfen = (char *) CALLOC(sfenr.length() + 1, sizeof(char));
        strcpy(sfen, sfenr.c_str());
    } else {
        sfen = nullptr;
    }

    r.close();

    return 1;
}

int ch_fsave(char *jsonf, char *sessionId, char *fen, char *move, char *level, char *sfen, int solver) {
    usleep(250);

    if (fen == nullptr || move == nullptr) {
        std::ofstream o = std::ofstream(jsonf);
        if (!o.good()) return 0;

        json j;

        j["sessionId"] = std::string(sessionId);
        j["fen"] = std::string("");
        j["pgn"] = std::string("");

        o << std::setw(4) << j << std::endl;

        o.close();
        return 0;
    }

    std::ofstream o = std::ofstream(jsonf);
    if (!o.good()) return 0;

    json j;

    j["sessionId"] = std::string(sessionId);
    j["fen"] = std::string(fen);
    j["pgn"] = to_lowercase(std::string(move));

    if (level == nullptr) {
        j["level"] = std::string("1");
    } else {
        j["level"] = std::string(level);
    }

    if (solver == 0) j["sfen"] = std::string("ER");
    if (solver == 1) j["sfen"] = std::string("AI");
    if (solver == 2) j["sfen"] = std::string("ML");

    o << std::setw(4) << j << std::endl;

    o.close();

    return 1;
}

int ch_cmp_move(char *m1, char *m2) {
    auto ms1 = std::string(m1);
    auto ms2 = std::string(m2);
    return ms1.compare(m2);
}

int ch_get_all_valid_moves(char* sfen, char *fen, char *&valid_fen, char **&valid_moves, int &valid_moves_count) {
    if (valid_fen != nullptr || valid_moves != nullptr || valid_moves_count != 0) {
        return 0;
    }
    const char *startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);

    libchess::Position pos = libchess::Position(fen);

    if (!pos.valid()) return 0;

    std::vector<libchess::Move> moves = ch_legal_moves(start, pos);

    if (moves.empty()) return 0;

    valid_moves = (char **) CALLOC(moves.size(), sizeof(char *));

    int i = 0;
    for (libchess::Move move : moves) {
        std::string smove = to_lowercase(std::string(move));
        char* imove = (char *) CALLOC(smove.length() + 1, sizeof(char));
        strcpy(imove, smove.c_str());
        valid_moves[i] = imove;
        i++;
    }

    valid_fen = (char *) CALLOC(strlen(fen) + 1, sizeof(char));
    strcpy(valid_fen, fen);

    valid_moves_count = (int)moves.size();

    return 1;
}

int ch_get_all_valid_moves_after(char* sfen, char *fen, char *valid_move, char *&valid_fen, char **&valid_moves, int &valid_moves_count) {
    if (valid_fen != nullptr || valid_moves != nullptr || valid_moves_count != 0) {
        return 0;
    }
    const char *startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);

    libchess::Position pos = libchess::Position(fen);

    if (!pos.valid()) return 0;

    auto moves_before = ch_legal_moves(start, pos);

    for(auto move : moves_before) {
        if (to_lowercase(std::string(move)) == std::string(valid_move)) {
            pos.makemove(move);
            break;
        }
    }

    std::vector<libchess::Move> moves = ch_legal_moves(start, pos);

    if (moves.empty()) return 0;

    valid_moves = (char **) CALLOC(moves.size(), sizeof(char *));

    int i = 0;
    for (libchess::Move move : moves) {
        std::string smove = to_lowercase(std::string(move));
        char* imove = (char *) CALLOC(smove.length() + 1, sizeof(char));
        strcpy(imove, smove.c_str());
        valid_moves[i] = imove;
        i++;
    }

    std::string next_fen = pos.get_fen();
    valid_fen = (char *) CALLOC(next_fen.length() + 1, sizeof(char));
    strcpy(valid_fen, next_fen.c_str());

    valid_moves_count = (int)moves.size();

    return 1;
}

void ch_split(const std::string& str, std::vector<std::string>& v) {
    std::stringstream ss(str);
    ss >> std::noskipws;
    std::string field;
    char ws_delim;
    while(true) {
        if( ss >> field )
            v.emplace_back(field);
        else if (ss.eof())
            break;
        else
            v.emplace_back("");
        ss.clear();
        ss >> ws_delim;
    }
}

char *ch_analyze_pos(char* sfen, char* positions, char *&pos_move_sfen, char *&pos_move_fen, char **&pos_moves, char *&pos_move, int &pos_moves_count) {
    const char* startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    pos_moves_count = 0;

    std::string line = std::string(positions);
    libchess::Position pos = libchess::Position(sfen[0] != '\0'  ? sfen : startfen);

    if (line.substr(0, 13) == "position fen ") {
        line = line.substr(13, line.length()-13);
        pos_move_sfen = (char *) CALLOC(line.length()+1, sizeof(char));
        pos_move_fen = (char *) CALLOC(line.length()+1, sizeof(char));
        auto pos_move_n = line.find(" moves ");
        if (std::string::npos != pos_move_n) {
            strcpy(pos_move_sfen, line.substr(0, pos_move_n).c_str());
        } else {
            strcpy(pos_move_sfen, line.substr(0, line.length()).c_str());
        }
        if (std::string::npos != pos_move_n) {
            strcpy(pos_move_fen, line.substr(0, pos_move_n).c_str());
        } else {
            strcpy(pos_move_fen, line.substr(0, line.length()).c_str());
        }
        pos_move = (char *) CALLOC(1, sizeof(char));
        strcpy(pos_move, "");
        pos = libchess::Position(pos_move_sfen);
        if (std::string::npos != pos_move_n) line = line.substr(pos_move_n + 1);
    }
    else if (line.substr(0,18) == "position startpos " ) {
        pos_move_sfen = (char *) CALLOC(strlen(startfen)+1, sizeof(char));
        strcpy(pos_move_sfen, startfen);
        line = line.substr(18, line.length()-18);
        pos = libchess::Position(startfen);
    }
    else if (line.substr(0,17) == "position startpos" ) {
        pos_move_sfen = (char *) CALLOC(strlen(startfen)+1, sizeof(char));
        strcpy(pos_move_sfen, startfen);
        pos_move_fen = (char *) CALLOC(strlen(startfen)+1, sizeof(char));
        strcpy(pos_move_fen, startfen);
        pos = libchess::Position(startfen);
        pos_move = (char *) CALLOC(1, sizeof(char));
        strcpy(pos_move, "");
        std::string next_fen = pos.get_fen();
        char* valid_fen = (char *) CALLOC(next_fen.length()+1, sizeof(char));
        strcpy(valid_fen, next_fen.c_str());
        return valid_fen;
    }

    if (line.substr(0,6) == "moves " ) {

        line = line.substr(6, line.length()-6);
        std::vector<std::string> moves;
        ch_split(line, moves);

        std::string pick_move;

        pos_moves = (char **) CALLOC(moves.size(), sizeof(char*));

        for(int i = 0; i < moves.size()-1; ++i) {
            pick_move = moves[i];
            pos_moves[i] = (char *) CALLOC(pick_move.length()+1, sizeof(char));
            pos.makemove(moves[i]);
            strcpy(pos_moves[i], pick_move.c_str());
            pos_moves_count++;
        }
        pick_move = moves[moves.size()-1];

        if (pos_move_fen == nullptr) {
            auto next_pre_fen = pos.get_fen();
            const char *next_pre_sfen = next_pre_fen.c_str();
            pos_move_fen = (char *) CALLOC(strlen(next_pre_sfen) + 1, sizeof(char));
            strcpy(pos_move_fen, next_pre_sfen);
        }

        pos_move = (char *) CALLOC(pick_move.length()+1, sizeof(char));
        strcpy(pos_move, pick_move.c_str());

        pos_moves[moves.size()-1] = (char *) CALLOC(pick_move.length()+1, sizeof(char));
        pos.makemove(moves[moves.size()-1]);
        strcpy(pos_moves[moves.size()-1], pick_move.c_str());
        pos_moves_count++;
    }

    std::string next_fen = pos.get_fen();
    char* valid_fen = (char *) CALLOC(next_fen.length()+1, sizeof(char));
    strcpy(valid_fen, next_fen.c_str());

    return valid_fen;
}

int ch_print_board(const char *valid_fen) {
    libchess::Position pos = libchess::Position(valid_fen);
    std::cerr << pos.get_fen() << std::endl;
    std::cerr << pos << std::endl;
    return 1;
}

int ch_is_checkmate_move(const char* sfen, const char *valid_fen, int idx) {
    const char *startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);
    libchess::Position pos = libchess::Position(valid_fen);
    libchess::Move move = ch_legal_moves(start, pos)[idx];
    pos.makemove(move);
    return pos.is_checkmate();
}

int ch_is_take_move(const char* sfen, const char *valid_fen, int idx) {
    const char *startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);
    libchess::Position pos = libchess::Position(valid_fen);
    libchess::Move move = ch_legal_moves(start, pos)[idx];
    pos.makemove(move);
    return !move.is_capturing() || pos.square_attacked(move.to(), pos.turn());
}

int ch_is_queen_attacked_move(const char* sfen, const char *valid_fen, int idx) {
    const char *startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);
    libchess::Position pos = libchess::Position(valid_fen);
    libchess::Move move = ch_legal_moves(start, pos)[idx];
    pos.makemove(move);
    return move.piece() == libchess::Queen &&
           (pos.square_attacked(move.from(), pos.turn()) && !pos.square_attacked(move.to(), pos.turn()));
}

int ch_is_checkmate(const char *valid_fen) {
    libchess::Position pos = libchess::Position(valid_fen);
    return pos.is_checkmate();
}

int ch_count_draw(const libchess::Position& sfen, libchess::Position& pos) {
    auto moves = ch_legal_moves(sfen, pos);

    if (moves.empty()) return 1;

    int i;
    int score = 0;
    int count = 0;

    for (i = 0; i < 64; i++) {
        const auto sq = libchess::Square(i);
        const auto bb = libchess::Bitboard{sq};

        if (pos.pieces(libchess::Side::White, libchess::Piece::Pawn) & bb) {
            count += 1;
            score += 1; // << " ♟ ";
        } else if (pos.pieces(libchess::Side::White, libchess::Piece::Knight) & bb) {
            count += 1;
            score += 3; // << " ♞ ";
        } else if (pos.pieces(libchess::Side::White, libchess::Piece::Bishop) & bb) {
            count += 1;
            score += 3; // << " ♝ ";
        } else if (pos.pieces(libchess::Side::White, libchess::Piece::Rook) & bb) {
            count += 1;
            score += 5; // << " ♜ ";
        } else if (pos.pieces(libchess::Side::White, libchess::Piece::Queen) & bb) {
            count += 1;
            score += 9; // << " ♛ ";
        } else if (pos.pieces(libchess::Side::White, libchess::Piece::King) & bb) {
            count += 1;
            score += 0; // << " ♚ ";
        } else if (pos.pieces(libchess::Side::Black, libchess::Piece::Pawn) & bb) {
            count += 1;
            score -= 1; // << " ♙ ";
        } else if (pos.pieces(libchess::Side::Black, libchess::Piece::Knight) & bb) {
            count += 1;
            score -= 3; // << " ♘ ";
        } else if (pos.pieces(libchess::Side::Black, libchess::Piece::Bishop) & bb) {
            count += 1;
            score -= 3;  // << " ♗ ";
        } else if (pos.pieces(libchess::Side::Black, libchess::Piece::Rook) & bb) {
            count += 1;
            score -= 5;  // << " ♖ ";
        } else if (pos.pieces(libchess::Side::Black, libchess::Piece::Queen) & bb) {
            count += 1;
            score -= 9; // << " ♕ ";
        } else if (pos.pieces(libchess::Side::Black, libchess::Piece::King) & bb) {
            count += 1;
            score -= 0;  // << " ♔ ";
        } else {
            // << " · ";
        }
    }
    return ((count == 3 && (score == 3 || score -3)) || (count == 2 && score == 0) || (count == 3 && (score == -4 || score == 4)) || (count == 4 && (score == -8 || score == 8)));
}

int ch_is_end(const char* sfen, const char *valid_fen, int idx) {
    const char *startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);

    libchess::Position pos = libchess::Position(valid_fen);

    if (pos.is_checkmate() || pos.is_stalemate() || pos.is_draw() || ch_count_draw(start, pos)) return 1;

    auto moves = ch_legal_moves(start, pos);
    if (idx > 0 && idx < moves.size()) pos.makemove(moves[idx]);

    if (pos.is_checkmate() || pos.is_stalemate() || pos.is_draw() || ch_count_draw(start, pos)) return 1;

    return 0;
}

int ch_board_after_move(const char* sfen, const char *valid_fen, char *valid_move, char *&valid_fen_next, int &valid_move_idx, int &valid_move_cnt) {
    //if (!valid_fen || valid_fen[0] == '\0' || !valid_move || valid_move[0] == '\0') return 1;
    const char *startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);
    libchess::Position pos = libchess::Position(valid_fen);
    valid_move_idx = -1;
    auto moves = ch_legal_moves(start, pos);
    valid_move_cnt = (int)moves.size();
    for(int i = 0; i < valid_move_cnt; ++i) {
        if (strcmp(std::string(moves[i]).c_str(), valid_move) == 0) {
            pos.makemove(moves[i]);
            valid_move_idx = i;
            break;
        }
    }
    std::string next_fen = pos.get_fen();
    valid_fen_next = (char *) CALLOC(next_fen.length() + 1, sizeof(char));
    strcpy(valid_fen_next, next_fen.c_str());
    return 1;
}


bool is_in_range(float p, float exact, float diff) {
    return ((exact - diff) <= p && p <= (exact + diff));
}

char *ch_board_to_fen(const float *board) {
    const float radius = 0.001f;
    std::string fen;
    int emptyCount = 0;

    for (int rank = 7; rank >= 0; rank--) {
        for (int file = 0; file < 8; file++) {
            int index = rank * 8 + file;
            float pieceValue = board[index];

            if (is_in_range(pieceValue, EMPTY, radius)) {
                emptyCount++;
            } else {
                if (emptyCount > 0) {
                    fen += std::to_string(emptyCount);
                    emptyCount = 0;
                }

                char pieceChar = '.';
                     if (is_in_range(pieceValue, BLACK_PAWN,   radius)) pieceChar = 'p';
                else if (is_in_range(pieceValue, BLACK_KNIGHT, radius)) pieceChar = 'n';
                else if (is_in_range(pieceValue, BLACK_BISHOP, radius)) pieceChar = 'b';
                else if (is_in_range(pieceValue, BLACK_ROOK,   radius)) pieceChar = 'r';
                else if (is_in_range(pieceValue, BLACK_QUEEN,  radius)) pieceChar = 'q';
                else if (is_in_range(pieceValue, BLACK_KING,   radius)) pieceChar = 'k';
                else if (is_in_range(pieceValue, WHITE_PAWN,   radius)) pieceChar = 'P';
                else if (is_in_range(pieceValue, WHITE_KNIGHT, radius)) pieceChar = 'N';
                else if (is_in_range(pieceValue, WHITE_BISHOP, radius)) pieceChar = 'B';
                else if (is_in_range(pieceValue, WHITE_ROOK,   radius)) pieceChar = 'R';
                else if (is_in_range(pieceValue, WHITE_QUEEN,  radius)) pieceChar = 'Q';
                else if (is_in_range(pieceValue, WHITE_KING,   radius)) pieceChar = 'K';

                if (pieceChar != '.') fen += pieceChar;
            }
        }
        if (rank > 0) {
            if (emptyCount > 0) {
                fen += std::to_string(emptyCount);
                emptyCount = 0;
            }
            fen += "/";
        }
    }
    if (emptyCount > 0) {
        fen += std::to_string(emptyCount);
        emptyCount = 0;
    }

    fen += board[8*8+0] == 0.f ? " w " : " b ";

    std::string castlingSymbolsU = "KQABCDEFGH";
    std::string castlingSymbolsL = "kqabcdefgh";

    if (board[8*8+1] > 0.f) fen += castlingSymbolsU[(int)(board[8*8+1] * 10.f) - 1];
    if (board[8*8+2] > 0.f) fen += castlingSymbolsU[(int)(board[8*8+2] * 10.f) - 1];
    if (board[8*8+3] > 0.f) fen += castlingSymbolsL[(int)(board[8*8+3] * 10.f) - 1];
    if (board[8*8+4] > 0.f) fen += castlingSymbolsL[(int)(board[8*8+4] * 10.f) - 1];

    if (board[8*8+1] > 0.f || board[8*8+2] > 0.f || board[8*8+3] > 0.f || board[8*8+4] > 0.f) {
        fen += " ";
    }
    else {
        fen += "- ";
    }
    
    std::string ep = std::to_string((int)(board[8*8+5] * 100.f));
    int epi = ep == "-" ? 0 : ((int)atoi(ep.c_str()));
    if (epi != 0) {
        std::string eps = "a1 ";
        eps[0] += (char)(epi / 8);
        eps[1] += (char)(epi % 8);
        fen += eps;
    } else {
        fen += "- ";
    }
    fen += std::to_string((int)board[8*8+6]);
    fen += " ";
    fen += std::to_string((int)board[8*8+7]);

    std::string next_fen = fen;
    char *valid_fen = (char *) CALLOC(next_fen.length() + 1, sizeof(char));
    strcpy(valid_fen, next_fen.c_str());

    return valid_fen;
}

float* ch_fen_to_board(char *valid_fen, int track_alloc = 0) {
    if (valid_fen == nullptr || valid_fen[0] == '\0') return nullptr;

    float* board = track_alloc == 0 ? ((float *) calloc(8 * 8 + 8, sizeof(float))) : ((float *) CALLOC(8 * 8 + 8, sizeof(float)));
    memset(board, 0.f, (8*8+8)*sizeof(float));

    std::string fen = std::string(valid_fen);

    libchess::Position pos = libchess::Position(valid_fen); // TUDO!!

    int i = 56;
    while (i >= 0) {
        const auto sq = libchess::Square(i);
        const auto bb = libchess::Bitboard{sq};
        if (pos.pieces(libchess::Side::Black, libchess::Piece::Pawn) & bb) {
            board[i] = BLACK_PAWN; // << " ♙ ";
        } else if (pos.pieces(libchess::Side::Black, libchess::Piece::Knight) & bb) {
            board[i] = BLACK_KNIGHT; // << " ♘ ";
        } else if (pos.pieces(libchess::Side::Black, libchess::Piece::Bishop) & bb) {
            board[i] = BLACK_BISHOP; // << " ♗ ";
        } else if (pos.pieces(libchess::Side::Black, libchess::Piece::Rook) & bb) {
            board[i] = BLACK_ROOK; // << " ♖ ";
        } else if (pos.pieces(libchess::Side::Black, libchess::Piece::Queen) & bb) {
            board[i] = BLACK_QUEEN; // << " ♕ ";
        } else if (pos.pieces(libchess::Side::Black, libchess::Piece::King) & bb) {
            board[i] = BLACK_KING; // << " ♔ ";
        } else if (pos.pieces(libchess::Side::White, libchess::Piece::Pawn) & bb) {
            board[i] = WHITE_PAWN; // << " ♟ ";
        } else if (pos.pieces(libchess::Side::White, libchess::Piece::Knight) & bb) {
            board[i] = WHITE_KNIGHT; // << " ♞ ";
        } else if (pos.pieces(libchess::Side::White, libchess::Piece::Bishop) & bb) {
            board[i] = WHITE_BISHOP; // << " ♝ ";
        } else if (pos.pieces(libchess::Side::White, libchess::Piece::Rook) & bb) {
            board[i] = WHITE_ROOK; // << " ♜ ";
        } else if (pos.pieces(libchess::Side::White, libchess::Piece::Queen) & bb) {
            board[i] = WHITE_QUEEN; // << " ♛ ";
        } else if (pos.pieces(libchess::Side::White, libchess::Piece::King) & bb) {
            board[i] = WHITE_KING; // << " ♚ ";
        } else {
            board[i] = EMPTY; // << " · ";
        }

        if (i % 8 == 7) {
            i -= 16;
        }

        i++;
    }

    std::vector<std::string> fen_split = std::vector<std::string>();
    ch_split(fen, fen_split);

    int wpos = fen_split[1] == "w" ? 0 : 1;
    std::string castlingS = fen_split[fen_split.size()-4];
    const char* castling = castlingS == "-" ? "" : castlingS.c_str();
    board[8*8+0] = wpos == 0 ? 0.f : 1.f;

    const char *castlingSymbolsU = "KQABCDEFGH";
    const char *castlingSymbolsL = "kqabcdefgh";

    float sK = 0.f, sQ = 0.f, sk = 0.f, sq = 0.f;

    for (int ci = 0; castling[ci] != '\0'; ++ci) {
        const char c = castling[ci];

        const char *posU = strchr(castlingSymbolsU, c);
        if (posU) {
            if (sK == 0.f) sK = (float)(posU - castlingSymbolsU + 1) / 10.f;
            else if (sQ == 0.f) sQ = (float)(posU - castlingSymbolsU + 1) / 10.f;
        }

        const char *posL = strchr(castlingSymbolsL, c);
        if (posL) {
            if (sk == 0.f) sk = (float)(posL - castlingSymbolsL + 1) / 10.f;
            else if (sq == 0.f) sq = (float)(posL - castlingSymbolsL + 1) / 10.f;
        }
    }

    board[8 * 8 + 1] = sK;
    board[8 * 8 + 2] = sQ;
    board[8 * 8 + 3] = sk;
    board[8 * 8 + 4] = sq;

    const char* ep = fen_split[fen_split.size()-3].c_str();
    board[8*8+5] = ep[0] == '-' ? 0 : ((float)(ep[0] - 'a') * 8 + (float)(ep[1] - '1')) / 100.f;
    board[8*8+6] = (float)atoi(fen_split[fen_split.size()-2].c_str());
    board[8*8+7] = (float)atoi(fen_split[fen_split.size()-1].c_str());

    return board;
}

char* ch_get_fen_960() {
    srandom(time(0));
    int seed = ((int)random() % 960) + 1;
    std::string r960 = ch_generate_deterministic_960(seed);
    std::string l960 = r960;
    std::transform(l960.begin(), l960.end(), l960.begin(), ::tolower);
    size_t k = r960.find('K');
    size_t r1 = r960.find('R');
    size_t r2 = r960.rfind('R');
    std::string castlingRights;
    std::string castlingSymbolsU = "ABCDEFGH";
    std::string castlingSymbolsL = "abcdefgh";
    if (r1 < k) castlingRights += castlingSymbolsU[r2];
    if (r2 > k) castlingRights += castlingSymbolsU[r1];
    if (r1 < k) castlingRights += castlingSymbolsL[r2];
    if (r2 > k) castlingRights += castlingSymbolsL[r1];
    if (castlingRights.empty()) castlingRights = "-";
    std::string fen960 = l960 + "/pppppppp/8/8/8/8/PPPPPPPP/" + r960 + " w " + castlingRights + " - 0 1";
    char* fen = (char *) CALLOC((fen960.length() + 1), sizeof(char));
    strcpy(fen, fen960.c_str());
    return fen;
}

float ch_eval_the_board(const char* sfen, float* board, float* powW, float* powB) {
    const char* startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);
    char* valid_fen = ch_board_to_fen(board);
    libchess::Position fen = libchess::Position(valid_fen);
    FREE(valid_fen);
    return (float)ch_evaluate_board(start, fen, fen.turn(), powW, powB);
}

float* ch_eval_the_board_moves(const char* sfen, float* board, float *&best_values, int* best_value_index) {
    const char *startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);
    char *valid_fen = ch_board_to_fen(board);
    libchess::Position pos = libchess::Position(valid_fen);
    auto moves = ch_legal_moves(start, pos);
    auto evaluates = (float *) CALLOC(moves.size(), sizeof(float));
    float evaluates_max = -FLT_MAX;
    float best_value = 0;
    int counter = 0;
    int best_index = ch_eval_best_trivial_move(sfen, valid_fen, 3, best_value, best_values, &counter);
    for (int i = 0; i < moves.size(); ++i) {
        pos.makemove(moves[i]);
        evaluates[i] = (float)ch_evaluate_board(start, pos, pos.turn());
        if (i == best_index) evaluates[i] += 2.7f;
        pos.undomove();
        if (evaluates[i] > evaluates_max) {
            best_values[i] = evaluates[i];
            *best_value_index = i;
        }
    }
    FREE(valid_fen);
    return evaluates;
}

int ch_moves_index(char *sfen, char* valid_fen, char* valid_fen_move) {
    const char *startfen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
    libchess::Position start = sfen == nullptr || sfen[0] == '\0' ? libchess::Position(startfen) : libchess::Position(sfen);
    libchess::Position pos = libchess::Position(valid_fen);
    auto moves = ch_legal_moves(start, pos);
    for(int i = 0; i < moves.size(); ++i) {
        if (std::string(moves[i]) == std::string(valid_fen_move)) return i;
    }
    return -1;
}

}
