#include "darknet.h"
#include "ch_mcts.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <stdint.h>
#include <sys/time.h>
#include "system.h"
#include <stdio.h>

#define CH_MCTS_CACHE_SIZE (1<<15)
#define CH_MCTS_MAX_CHILDREN_LIMIT 512
typedef struct { uint64_t key; int n; float value; ch_mv mv[CH_MCTS_MAX_CHILDREN_LIMIT]; float prob[CH_MCTS_MAX_CHILDREN_LIMIT]; int valid; } ch_mcts_cache_entry;
static ch_mcts_cache_entry ch_mcts_cache[CH_MCTS_CACHE_SIZE];
static inline uint64_t fnv1a_64(const char* s){ uint64_t h=1469598103934665603ULL; for(;*s;s++){ h^=(unsigned char)*s; h*=1099511628211ULL;} return h; }
static inline int cache_lookup(uint64_t k, int max_children, float* out_value, ch_mv* out_mv, float* out_prob, int* out_n){ uint32_t idx=(uint32_t)(k)&(CH_MCTS_CACHE_SIZE-1); for(int i=0;i<256;i++){ ch_mcts_cache_entry* e=&ch_mcts_cache[(idx+i)&(CH_MCTS_CACHE_SIZE-1)]; if(e->valid && e->key==k){ int n=e->n; if(n>max_children) n=max_children; for(int j=0;j<n;j++){ out_mv[j]=e->mv[j]; out_prob[j]=e->prob[j]; } *out_value=e->value; *out_n=n; return 1; } if(!e->valid) break; } return 0; }
static inline void cache_store(uint64_t k, int n, float value, const ch_mv* mv, const float* prob){ uint32_t idx=(uint32_t)(k)&(CH_MCTS_CACHE_SIZE-1); for(int i=0;i<256;i++){ ch_mcts_cache_entry* e=&ch_mcts_cache[(idx+i)&(CH_MCTS_CACHE_SIZE-1)]; if(!e->valid || e->key==k){ e->key=k; e->n=n>CH_MCTS_MAX_CHILDREN_LIMIT?CH_MCTS_MAX_CHILDREN_LIMIT:n; e->value=value; e->valid=1; for(int j=0;j<e->n;j++){ e->mv[j]=mv[j]; e->prob[j]=prob[j]; } return; } } }


static void* arena_alloc(ch_mcts* m, size_t bytes, size_t align) {
    size_t base = (size_t)m->arena + m->arena_used;
    size_t aligned = (base + (align - 1)) & ~(align - 1);
    size_t new_used = (aligned - (size_t)m->arena) + bytes;
    if (new_used > m->arena_size) return NULL;
    m->arena_used = new_used;
    return (void*)aligned;
}

static inline unsigned seed_from_ptr(const void* p) { return (unsigned)((uintptr_t)p ^ (uintptr_t)time(NULL)); }

static inline void fisher_yates_shuffle(int* idx, int n, unsigned* seed) {
    for (int i = n - 1; i > 0; --i) {
        int j = rand_r(seed) % (i + 1);
        int t = idx[i]; idx[i] = idx[j]; idx[j] = t;
    }
}

static ch_mcts_node* alloc_node(ch_mcts* m) {
    ch_mcts_node* n = (ch_mcts_node*)arena_alloc(m, sizeof(ch_mcts_node), 8);
    if (!n) return NULL;
    memset(n, 0, sizeof(*n));
    n->children    = (ch_mcts_node**)arena_alloc(m, sizeof(ch_mcts_node*) * m->cfg.max_children, 8);
    n->moves       = (ch_mv*)arena_alloc(m, sizeof(ch_mv) * m->cfg.max_children, 8);
    n->priors      = (float*)arena_alloc(m, sizeof(float) * m->cfg.max_children, 8);
    n->value_sum   = (float*)arena_alloc(m, sizeof(float) * m->cfg.max_children, 8);
    n->visit_count = (int*)arena_alloc(m, sizeof(int) * m->cfg.max_children, 8);
    n->state       = (ch_board*)arena_alloc(m, sizeof(ch_board), 8);
    if (!n->children || !n->moves || !n->priors || !n->value_sum || !n->visit_count || !n->state) return NULL;
    return n;
}

int ch_mcts_init(ch_mcts* m, ch_api api, ch_mcts_config cfg, network* net, void* arena, size_t arena_size) {
    if (!m || !arena || arena_size == 0) return -1;
    m->api = api; m->cfg = cfg; m->net = net; m->arena = arena; m->arena_size = arena_size; m->arena_used = 0; return 0;
}

ch_mcts_node* ch_mcts_create_root(ch_mcts* m, const ch_board* start, int to_move) {
    ch_mcts_node* r = alloc_node(m); if (!r) return NULL;
    memcpy(r->state, start, sizeof(ch_board));
    r->parent = NULL; r->to_move = to_move; r->is_expanded = 0; r->is_terminal = 0; r->total_visits = 0; return r;
}

float ch_mcts_child_q(const ch_mcts_node* node, int i) { int n = node->visit_count[i]; if (n <= 0) return 0.0f; return node->value_sum[i] / (float)n; }

float ch_mcts_child_u(const ch_mcts* m, const ch_mcts_node* node, int i) {
    float prior = node->priors[i];
    float parent_visits = (float)fmax(1, node->total_visits);
    return m->cfg.cpuct * prior * sqrtf(parent_visits) / (1.0f + (float)node->visit_count[i]);
}

static ch_mcts_node* select_leaf(ch_mcts* m, ch_mcts_node* node) {
    while (node->is_expanded && !node->is_terminal && node->child_count > 0) {
        int best = -1; float best_score = -FLT_MAX; int N = node->child_count;
        int* order = (int*)alloca(N*sizeof(int)); for (int i = 0; i < N; ++i) order[i] = i;
        unsigned seed = seed_from_ptr(node); fisher_yates_shuffle(order, N, &seed);
        for (int t = 0; t < N; ++t) {
            int i = order[t];
            float q = ch_mcts_child_q(node, i);
            float u = ch_mcts_child_u(m, node, i);
            float score = q + u;
            if (score > best_score) { best_score = score; best = i; }
        }
        ch_mcts_node* child = node->children[best];
        if (m->cfg.use_virtual_loss) {
            node->visit_count[best] += 1;
            node->value_sum[best]   -= m->cfg.virtual_loss;
            node->total_visits      += 1;
        }
        node = child;
    }
    return node;
}


static int expand_and_eval(ch_mcts* m, ch_mcts_node* node, float* out_value) {
    int term = m->api.terminal_result(node->state);
    if (term != 2) { node->is_terminal = 1; node->is_expanded = 1; node->child_count = 0; *out_value = (float)term; return 0; }
    int cap = m->cfg.max_children;
    if (cap > CH_MCTS_MAX_CHILDREN_LIMIT) cap = CH_MCTS_MAX_CHILDREN_LIMIT;
    ch_mv tmp_moves[CH_MCTS_MAX_CHILDREN_LIMIT];
    int genN = m->api.gen_legal(node->state, tmp_moves, cap);
    if (genN <= 0) { node->is_terminal = 1; node->is_expanded = 1; node->child_count = 0; *out_value = 0.0f; return 0; }
    uint64_t key = fnv1a_64(node->state->fen);
    float val = 0.0f;
    float pri[CH_MCTS_MAX_CHILDREN_LIMIT];
    int from_cache = 0; int cachedN = 0;
    from_cache = cache_lookup(key, genN, &val, tmp_moves, pri, &cachedN);
    if (!from_cache) {
        if (m->api.nn_eval(m->net, node->state, tmp_moves, genN, pri, &val) != 0) return -1;
        double s = 0.0; for (int i = 0; i < genN; ++i) { if (pri[i] < 0) pri[i] = 0; s += pri[i]; }
        if (s <= 0) { for (int i = 0; i < genN; ++i) pri[i] = 1.0f / (float)genN; }
        else { float inv = 1.0f / (float)s; for (int i = 0; i < genN; ++i) pri[i] *= inv; }
        if (!node->parent && m->cfg.dirichlet_alpha > 0 && m->cfg.root_noise_frac > 0) {
            double gsum = 0.0; double gbuf[CH_MCTS_MAX_CHILDREN_LIMIT]; unsigned seed = seed_from_ptr(node);
            for (int i = 0; i < genN; ++i) { int k = (int)fmax(1.0, m->cfg.dirichlet_alpha * 10.0); double acc = 0.0; for (int j = 0; j < k; ++j) { double u = (rand_r(&seed)+1.0) / (RAND_MAX+2.0); acc += -log(u); } gbuf[i] = acc; gsum += acc; }
            if (gsum > 0) { float eps = m->cfg.root_noise_frac; for (int i = 0; i < genN; ++i) { float g = (float)(gbuf[i] / gsum); pri[i] = (1.0f - eps) * pri[i] + eps * g; } }
        }
        cache_store(key, genN, val, tmp_moves, pri);
    } else {
        genN = cachedN;
    }
    node->child_count = genN;
    for (int i = 0; i < genN; ++i) {
        node->moves[i] = tmp_moves[i];
        node->priors[i] = pri[i];
        node->value_sum[i] = 0.0f;
        node->visit_count[i] = 0;
        ch_mcts_node* ch = alloc_node(m); if (!ch) return -1;
        node->children[i] = ch;
        ch->parent = node;
        ch->move_from_parent = node->moves[i];
        ch->to_move = -node->to_move;
        if (m->api.apply_move(node->state, node->moves[i], ch->state) != 0) return -1;
    }
    node->is_expanded = 1; node->is_terminal = 0; *out_value = val; return 0;
}


static void backpropagate(ch_mcts* m, ch_mcts_node* leaf, float value, int used_virtual_loss) {
    ch_mcts_node* node = leaf; int first = 1;
    while (node->parent) {
        ch_mcts_node* parent = node->parent;
        int idx = -1; for (int i = 0; i < parent->child_count; ++i) { if (parent->children[i] == node) { idx = i; break; } }
        if (idx < 0) return;
        if (used_virtual_loss && first && m->cfg.use_virtual_loss) { parent->visit_count[idx] -= 1; parent->value_sum[idx] += m->cfg.virtual_loss; parent->total_visits -= 1; }
        parent->visit_count[idx] += 1;
        parent->value_sum[idx]   += value;
        parent->total_visits     += 1;
        value = -value; node = parent; first = 0;
    }
}


static void run_simulation(ch_mcts* m, ch_mcts_node* root) {
    ch_mcts_node* leaf = select_leaf(m, root);
    float value = 0.0f; int rc = 0;
    if (!leaf->is_expanded) { rc = expand_and_eval(m, leaf, &value); if (rc != 0) return; }
    else { int term = m->api.terminal_result(leaf->state); value = (float)term; }
    backpropagate(m, leaf, -value, 1);
}

static double now_sec() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1e6;
}

void ch_mcts_run(ch_mcts* m, ch_mcts_node* root, int seconds) {
    double t0 = now_sec();
    while ((now_sec() - t0) < seconds) {
        run_simulation(m, root);
    }
}
/*
void ch_mcts_run(ch_mcts* m, ch_mcts_node* root, int simulations) {
    for (int i = 0; i < simulations; ++i) {
        run_simulation(m, root);
    }
}
*/
static int sample_from_visits(const ch_mcts_node* root, float tau, int* out_idx) {
    int N = root->child_count; if (N <= 0) return -1;
    if (tau <= 1e-6f) { int best = 0, bestv = root->visit_count[0]; for (int i = 1; i < N; ++i) { if (root->visit_count[i] > bestv) { bestv = root->visit_count[i]; best = i; } } *out_idx = best; return 0; }
    double sum = 0.0; double* w = (double*)alloca(N*sizeof(double)); double invtau = 1.0 / (double)tau;
    for (int i = 0; i < N; ++i) { double p = pow(fmax(1.0, (double)root->visit_count[i]), invtau); w[i] = p; sum += p; }
    if (sum <= 0) return -1; double r = ((double)rand() / (double)RAND_MAX) * sum; double acc = 0.0;
    for (int i = 0; i < N; ++i) { acc += w[i]; if (r <= acc) { *out_idx = i; return 0; } } *out_idx = N - 1; return 0;
}

ch_mv ch_mcts_pick_move(const ch_mcts* m, const ch_mcts_node* root, float tau) { (void)m; int idx = 0; if (sample_from_visits(root, tau, &idx) != 0) return (ch_mv)0; return root->moves[idx]; }

void ch_mcts_free_all(ch_mcts* m) { free(m); }

int ch_mcts_root_visits(const ch_mcts_node* root, int* out_counts, int max) { int n = root->child_count; if (max < n) n = max; for (int i = 0; i < n; ++i) out_counts[i] = root->visit_count[i]; return n; }

static void print_top_children(const ch_mcts_node* root, char** moves, int topN) {
    int N = root->child_count;
    if (N <= 0) { fprintf(stderr, "(no children)\n"); return; }
    if (topN > N) topN = N;
    int idx[topN];
    float score[topN];
    for (int i = 0; i < topN; ++i) { idx[i] = i; score[i] = (float)root->visit_count[i]; }
    for (int i = 0; i < topN; ++i) {
        for (int j = i + 1; j < topN; ++j) {
            if (score[j] > score[i]) {
                float ts = score[i]; score[i] = score[j]; score[j] = ts;
                int ti = idx[i]; idx[i] = idx[j]; idx[j] = ti;
            }
        }
    }
    fprintf(stderr, "top %d candidates:\n", topN);
    for (int k = 0; k < topN; ++k) {
        int i = idx[k];
        float q = ch_mcts_child_q(root, i);
        fprintf(stderr, "  #%d move=%s visits=%d Q=%.3f prior=%.3f\n",
               k + 1, moves[(int)root->moves[i]], root->visit_count[i], q, root->priors[i]);
    }
}

void ch_mcts_play_and_log(ch_mcts* m, ch_mcts_node* root, int seconds, char** moves) {
    double t0 = now_sec();
    ch_mcts_run(m, root, seconds);
    double t1 = now_sec();
    ch_mv mv = ch_mcts_pick_move(m, root, 0.0f);
    fprintf(stderr, "MCTS:\n");
    fprintf(stderr, "time: %.2fs\n", t1 - t0);
    fprintf(stderr, "total visits: %d\n", root->total_visits);
    fprintf(stderr, "chosen move: %s\n", moves[(int)mv]);
    print_top_children(root, moves, 5);
    m->arena_used = 0;
}
