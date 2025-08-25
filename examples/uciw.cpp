#include "uciw.hpp"

#include <iostream>
#include <algorithm>

#include "position.hpp"
#include "move.hpp"

UCIEngine::UCIEngine(const std::string& path) {
#ifdef _WIN32
    engine_ = _popen(path.c_str(), "w+");
#else
    engine_ = popen(path.c_str(), "w+");
#endif
    if (!engine_) {
        throw std::runtime_error("Failed to start engine: " + path);
    }
    running_ = true;
    reader_ = std::thread([this] { readLoop(); });

    sendCommand("uci");
}

UCIEngine::~UCIEngine() {
    try {
        quit();
    } catch (...) {}
#ifdef _WIN32
    if (engine_) _pclose(engine_);
#else
    if (engine_) pclose(engine_);
#endif
    if (reader_.joinable()) reader_.join();
}

void UCIEngine::sendCommand(const std::string& cmd) {
    std::lock_guard<std::mutex> lk(io_mtx_);
    if (!engine_) return;
    std::string out = cmd + "\n";
    fputs(out.c_str(), engine_);
    fflush(engine_);
}

std::string UCIEngine::getResponse() {
    std::unique_lock<std::mutex> lk(q_mtx_);
    q_cv_.wait(lk, [this]{ return !q_.empty() || !running_; });
    if (q_.empty()) return {};
    auto s = std::move(q_.front());
    q_.pop();
    return s;
}

bool UCIEngine::tryGetResponse(std::string& line) {
    std::lock_guard<std::mutex> lk(q_mtx_);
    if (q_.empty()) return false;
    line = std::move(q_.front());
    q_.pop();
    return true;
}

void UCIEngine::handshake() {
    for (;;) {
        auto line = getResponse();
        if (line.find("uciok") != std::string::npos) break;
    }
    sendCommand("isready");
    for (;;) {
        auto line = getResponse();
        if (line.find("readyok") != std::string::npos) break;
    }
}

void UCIEngine::setOption(const std::string& name, const std::string& value) {
    sendCommand(std::string("setoption name ") + name + " value " + value);
}

void UCIEngine::stop() { sendCommand("stop"); }

void UCIEngine::quit() {
    if (running_) {
        sendCommand("quit");
        running_ = false;
    }
}

void UCIEngine::readLoop() {
    char buf[2048];
    while (running_ && engine_ && fgets(buf, sizeof(buf), engine_)) {
        std::string line(buf);
        {
            std::lock_guard<std::mutex> lk(q_mtx_);
            q_.push(line);
        }
        q_cv_.notify_one();
    }
    running_ = false;
    q_cv_.notify_all();
}

UCIvsUCI::UCIvsUCI(const std::string& engine_white_path,
                   const std::string& engine_black_path)
: white_(engine_white_path), black_(engine_black_path) {
    // nothing else; we will handshake in init()
}

void UCIvsUCI::setConfig(const MatchConfig& cfg) { cfg_ = cfg; }

void UCIvsUCI::setWhiteOption(const std::string& name, const std::string& value) {
    white_.setOption(name, value);
}

void UCIvsUCI::setBlackOption(const std::string& name, const std::string& value) {
    black_.setOption(name, value);
}

void UCIvsUCI::init() {
    white_.handshake();
    black_.handshake();
}

static double resultToScore(const std::string& res, bool for_white) {
    if (res == "1-0") return for_white ? 1.0 : 0.0;
    if (res == "0-1") return for_white ? 0.0 : 1.0;
    return 0.5;
}

UCIvsUCI::SeriesResult UCIvsUCI::playSeries(int games) {
    SeriesResult sr;
    sr.games = games;
    bool swap_colors = false;

    for (int g = 0; g < games; ++g) {
        MatchConfig gcfg = cfg_;
        if (swap_colors) {
            std::swap(gcfg.white_name, gcfg.black_name);
        }
        setConfig(gcfg);

        auto res = playOne();
        sr.game_results.push_back(res);

        sr.total_score += swap_colors ? resultToScore(res.result_str, false)
                                      : resultToScore(res.result_str, true);

        sr.white_score += resultToScore(res.result_str, true);
        sr.black_score += resultToScore(res.result_str, false);

        swap_colors = !swap_colors;
    }

    return sr;
}

static std::string resultFromLibchess(const libchess::Position& pos) {
    return pos.is_checkmate() ? "1-0" : pos.is_terminal() ? "1/2-1/2" : "0-1";
}

GameResult UCIvsUCI::playOne() {
    GameResult gr;
    libchess::Position pos;

    if (!cfg_.start_fen.empty()) {
        pos = libchess::Position(cfg_.start_fen);
    }

    int white_time = cfg_.initial_time_ms;
    int black_time = cfg_.initial_time_ms;

    std::vector<int> eval_cp_hist;
    bool white_to_move = pos.turn() == libchess::White;

    int plies = 0;

    while (!pos.is_terminal() && (cfg_.max_plies <= 0 || plies < cfg_.max_plies)) {
        std::string best;
        bool ok = getEngineBestMove(pos, white_to_move, best, white_time, black_time, eval_cp_hist);
        if (!ok) {
            if (white_time <= 0 || black_time <= 0) {
                gr.termination = "time";
                gr.result_str = white_time <= 0 ? "0-1" : "1-0";
            } else {
                gr.termination = "unknown";
                gr.result_str = resultFromLibchess(pos);
            }
            break;
        }

        if (best == "(none)" || best.empty()) {
            break;
        }

        auto legal = pos.legal_moves();
        auto it = std::find_if(legal.begin(), legal.end(),
            [&](const libchess::Move& m){ return std::string(m) == std::string(best); });
        if (it == legal.end()) {
            std::cerr << "Illegal move from engine: " << best << "\n";
            gr.termination = "illegal";
            gr.result_str = white_to_move ? "0-1" : "1-0";
            break;
        }

        pos.makemove(*it);
        gr.uci_moves.push_back(best);

        ++plies;
        white_to_move = !white_to_move;
    }

    gr.plies = plies;
    gr.white_time_left_ms = white_time;
    gr.black_time_left_ms = black_time;

    if (pos.is_terminal()) {
        gr.result_str = resultFromLibchess(pos);
        gr.termination = "rules";
    } else if (gr.result_str.empty()) {
        // Safe default if loop exited by max plies
        gr.result_str = "1/2-1/2";
        gr.termination = "maxplies";
    }

    gr.pgn = buildPGN(pos, cfg_, gr.uci_moves, gr.result_str);
    return gr;
}

bool UCIvsUCI::getEngineBestMove(libchess::Position& pos,
                                 bool white_to_move,
                                 std::string& out_move,
                                 int& white_time_ms,
                                 int& black_time_ms,
                                 std::vector<int>& eval_cp_history) {
    UCIEngine& eng = white_to_move ? white_ : black_;

    auto fen = pos.get_fen();
    eng.sendCommand(positionCmdFromFen(fen));

    std::ostringstream go;
    go << "go "
       << "wtime " << white_time_ms << " "
       << "btime " << black_time_ms;
    if (cfg_.movestogo_hint > 0) {
        go << " movestogo " << cfg_.movestogo_hint;
    }
    if (cfg_.per_move_cap_ms > 0) {
        go << " movetime " << cfg_.per_move_cap_ms;
    }
    eng.sendCommand(go.str());

    auto t0 = std::chrono::steady_clock::now();

    std::string best;

    std::optional<int> last_cp;
    for (;;) {
        auto line = eng.getResponse();
        if (line.empty()) break;

        if (line.rfind("info ", 0) == 0) {
            auto cp = parseInfoCp(line);
            if (cp.has_value()) last_cp = cp.value();
        }

        if (line.rfind("bestmove", 0) == 0) {
            std::istringstream iss(line);
            std::string tag;
            iss >> tag >> best;
            break;
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    int spent = (int)std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

    if (white_to_move) {
        white_time_ms -= spent;
        white_time_ms += cfg_.increment_ms;
    } else {
        black_time_ms -= spent;
        black_time_ms += cfg_.increment_ms;
    }

    if (white_time_ms <= 0 || black_time_ms <= 0) {
        out_move.clear();
        return false;
    }

    if (cfg_.resign_cp > 0 && cfg_.resign_moves > 0 && last_cp.has_value()) {
        eval_cp_history.push_back(last_cp.value());
        if ((int)eval_cp_history.size() >= cfg_.resign_moves) {
            bool losing = true;
            for (int i = (int)eval_cp_history.size() - cfg_.resign_moves; i < (int)eval_cp_history.size(); ++i) {
                int cp = eval_cp_history[i];
                if (white_to_move) {
                    if (cp > -cfg_.resign_cp) { losing = false; break; }
                } else {
                    if (cp < cfg_.resign_cp) { losing = false; break; }
                }
            }
            if (losing) {
                out_move = "(none)";
                return true;
            }
        }
    }

    out_move = best;
    return true;
}

std::string UCIvsUCI::positionCmdFromFen(const std::string& fen) {
    std::ostringstream oss;
    oss << "position fen " << fen;
    return oss.str();
}

std::string UCIvsUCI::buildPGN(const libchess::Position& final_pos,
                               const MatchConfig& cfg,
                               const std::vector<std::string>& uci_moves,
                               const std::string& result_str) {
    std::ostringstream oss;
    oss << "[Event \"UCI vs UCI\"]\n";
    oss << "[White \"" << cfg.white_name << "\"]\n";
    oss << "[Black \"" << cfg.black_name << "\"]\n";
    if (!cfg.start_fen.empty()) {
        oss << "[FEN \"" << cfg.start_fen << "\"]\n";
        oss << "[SetUp \"1\"]\n";
    }
    oss << "[TimeControl \"" << cfg.initial_time_ms/1000 << "+" << cfg.increment_ms/1000 << "\"]\n";
    oss << "[Result \"" << result_str << "\"]\n\n";

    int move_no = 1;
    bool white = true;
    for (const auto& m : uci_moves) {
        if (white) oss << move_no++ << ". ";
        oss << m << " ";
        white = !white;
    }
    oss << result_str << "\n";
    return oss.str();
}

std::optional<int> UCIvsUCI::parseInfoCp(const std::string& line) {
    auto pos = line.find(" score ");
    if (pos == std::string::npos) return std::nullopt;
    auto rest = line.substr(pos + 7);
    auto cp_pos = rest.find("cp ");
    if (cp_pos != std::string::npos) {
        cp_pos += 3;
        std::istringstream iss(rest.substr(cp_pos));
        int cp;
        if (iss >> cp) return cp;
    }
    auto m_pos = rest.find("mate ");
    if (m_pos != std::string::npos) {
        m_pos += 5;
        std::istringstream iss(rest.substr(m_pos));
        int mate_ply;
        if (iss >> mate_ply) {
            int sign = (mate_ply > 0) ? 1 : -1;
            return 32000 * sign;
        }
    }
    return std::nullopt;
}

#ifdef UCIW_BUILD_MAIN

#include <cstdlib>

int main(int argc, char** argv) {
    std::string e1 = argc > 1 ? argv[1] : "stockfish";
    std::string e2 = argc > 2 ? argv[2] : "stockfish";

    UCIvsUCI arena(e1, e2);
    MatchConfig cfg;
    cfg.initial_time_ms = 3 * 60 * 1000; // 3+2
    cfg.increment_ms = 2000;
    cfg.movestogo_hint = 40;
    cfg.max_plies = 300;
    cfg.white_name = "EngineA";
    cfg.black_name = "EngineB";
    arena.setConfig(cfg);

    arena.init();

    auto series = arena.playSeries(2);
    double score = series.total_score;
    std::cout << "Series finished. Score (Engine1 perspective): " << score << "/" << series.games << "\n";

    for (size_t i = 0; i < series.game_results.size(); ++i) {
        std::cout << "Game " << (i+1) << " result: " << series.game_results[i].result_str << "\n";
        std::cout << series.game_results[i].pgn << "\n";
    }

    return 0;
}

#endif
