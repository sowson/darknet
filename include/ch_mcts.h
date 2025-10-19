
#ifndef CH_MCTS_H
#define CH_MCTS_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef uint64_t ch_mv;
    typedef struct {
        char fen[128];
    } ch_board;
typedef struct ch_network ch_network;
typedef struct {
    int  (*gen_legal)(const ch_board* board, ch_mv* moves, int max_moves);
    int  (*apply_move)(const ch_board* board, ch_mv mv, ch_board* out);
    int  (*terminal_result)(const ch_board* board);
    int  (*nn_eval)(network* net, const ch_board* board, const ch_mv* moves, int moves_count, float* out_policy, float* out_value);
} ch_api;
typedef struct {
    float cpuct;
    float dirichlet_alpha;
    float root_noise_frac;
    int   max_children;
    int   use_virtual_loss;
    float virtual_loss;
} ch_mcts_config;
typedef struct ch_mcts_node {
    struct ch_mcts_node* parent;
    struct ch_mcts_node** children;
    int child_count;
    int is_expanded;
    int is_terminal;
    int to_move;
    ch_mv* moves;
    float* priors;
    float* value_sum;
    int*   visit_count;
    int    total_visits;
    ch_mv move_from_parent;
    ch_board* state;
} ch_mcts_node;
typedef struct {
    ch_api api;
    ch_mcts_config cfg;
    network* net;
    void* arena;
    size_t arena_size;
    size_t arena_used;
} ch_mcts;
int ch_mcts_init(ch_mcts* mcts, ch_api api, ch_mcts_config cfg, network* net, void* arena, size_t arena_size);
ch_mcts_node* ch_mcts_create_root(ch_mcts* mcts, const ch_board* start, int to_move);
void ch_mcts_run(ch_mcts* mcts, ch_mcts_node* root, int simulations);
ch_mv ch_mcts_pick_move(const ch_mcts* mcts, const ch_mcts_node* root, float tau);
void ch_mcts_free_all(ch_mcts* mcts);
int ch_mcts_root_visits(const ch_mcts_node* root, int* out_counts, int max);
float ch_mcts_child_q(const ch_mcts_node* node, int i);
float ch_mcts_child_u(const ch_mcts* mcts, const ch_mcts_node* node, int i);
void ch_mcts_play_and_log(ch_mcts* m, ch_mcts_node* root, int seconds, char** moves);
#ifdef __cplusplus
}
#endif
#endif
