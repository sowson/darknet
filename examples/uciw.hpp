
#pragma once

#include <string>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>
#include <optional>
#include <functional>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <sstream>

namespace libchess {
    class Position;
    class Move;
}

class UCIEngine {
public:
    using Callback = std::function<void(const std::string&)>;

    explicit UCIEngine(const std::string& path = "stockfish");
    ~UCIEngine();

    UCIEngine(const UCIEngine&) = delete;
    UCIEngine& operator=(const UCIEngine&) = delete;

    void sendCommand(const std::string& cmd);

    std::string getResponse();

    bool tryGetResponse(std::string& line);

    void handshake();

    void setOption(const std::string& name, const std::string& value);

    void stop();

    void quit();

private:
    FILE* engine_{nullptr};
    std::thread reader_;
    std::mutex io_mtx_;
    std::mutex q_mtx_;
    std::condition_variable q_cv_;
    std::queue<std::string> q_;
    std::atomic<bool> running_{false};

    void readLoop();
};


struct MatchConfig {
    int initial_time_ms = 180000; // 3+2 default
    int increment_ms = 2000;
    int per_move_cap_ms = 0;
    int max_plies = 300;
    int movestogo_hint = 40;
    int resign_cp = 0;
    int resign_moves = 0;
    std::string start_fen;
    std::string white_name = "Engine1";
    std::string black_name = "Engine2";
};

struct GameResult {
    std::string result_str;
    std::vector<std::string> uci_moves;
    int plies = 0;
    int white_time_left_ms = 0;
    int black_time_left_ms = 0;
    std::string termination;
    std::string pgn;
};

class UCIvsUCI {
public:
    UCIvsUCI(const std::string& engine_white_path,
             const std::string& engine_black_path);

    void setConfig(const MatchConfig& cfg);

    void setWhiteOption(const std::string& name, const std::string& value);
    void setBlackOption(const std::string& name, const std::string& value);

    void init();

    GameResult playOne();

    struct SeriesResult {
        int games = 0;
        double white_score = 0.0;
        double black_score = 0.0;
        double total_score = 0.0;
        std::vector<GameResult> game_results;
    };
    SeriesResult playSeries(int games);

private:
    UCIEngine white_;
    UCIEngine black_;
    MatchConfig cfg_;
    std::vector<std::pair<std::string,std::string>> pending_options_;

    bool getEngineBestMove(libchess::Position& pos,
                           bool white_to_move,
                           std::string& out_move,
                           int& white_time_ms,
                           int& black_time_ms,
                           std::vector<int>& eval_cp_history);

    static std::string positionCmdFromFen(const std::string& fen);

    static std::string buildPGN(const libchess::Position& final_pos,
                                const MatchConfig& cfg,
                                const std::vector<std::string>& uci_moves,
                                const std::string& result_str);

    static std::optional<int> parseInfoCp(const std::string& line);
};
