#include "darknet.h"
#include "ch_mcts.h"
#include "system.h"
#include "image.h"
#include "time.h"
#include "opencl.h"
#include <string.h>

#include <stdlib.h>

#include <stdio.h>
#ifdef WIN32
#include "unistd\dirent.h"
#else
#include <dirent.h>
#endif

#ifdef WIN32
#include "unistd\unistd.h"
#else
#include <unistd.h>
#endif

#include <sys/stat.h>
#include <assert.h>

#define class temp

#include <execinfo.h>

void ASSERT(int condition) {
    if (!condition) {
        void *buffer[1000];
        int size = backtrace(buffer, 1000);
        char **symbols = backtrace_symbols(buffer, size);
        if (symbols) {
            fprintf(stderr, "call Stack:\n");
            for (int i = 0; i < size; ++i) {
                fprintf(stderr, "%s\n", symbols[i]);
            }
            FREE(symbols);
        } else {
            fprintf(stderr, "failed to retrieve call stack symbols!\n");
        }
    }
}

static int ch_test_tchess_count = -1;

typedef struct {
    char* fen;
    char **moves;
    int n;
} ch_moves;

typedef struct ch_constant_memory_queue {
    void** data;
    void *tree;
    int capacity;
    int count;
    int put_id;
    int get_id;
    int peak_id;
    int total_count;
    int index;
    float value;
    float power;
} ch_constant_memory_queue;

ch_constant_memory_queue* ch_create_constant_memory_queue(int capacity) {
    ch_constant_memory_queue* queue = (ch_constant_memory_queue*) CALLOC(1, sizeof(ch_constant_memory_queue));
    queue->data = (void**) CALLOC(capacity, sizeof(void*));
    queue->capacity = capacity;
    for(int i = 0; i < queue->capacity; ++i) queue->data[i] = NULL;
    queue->count = 0;
    queue->put_id = -1;
    queue->get_id = -1;
    queue->peak_id = -1;
    queue->index = -1;
    return queue;
}

void ch_clean_constant_memory_queue(ch_constant_memory_queue* queue){
    for(int i = 0; i < queue->count; ++i) if (queue->data[i] != NULL) { FREE(queue->data[i]); queue->data[i] = NULL; }
    queue->count = 0;
    queue->peak_id = -1;
    queue->get_id = -1;
    queue->put_id = -1;
}

int ch_get_next_put_id(ch_constant_memory_queue* queue) {
    int put_id = queue->capacity > queue->put_id + 1 ? ++queue->put_id : 0;
    return put_id;
}

void ch_enqueue(ch_constant_memory_queue* queue, void* item) {
    int put_id = ch_get_next_put_id(queue);
    queue->data[put_id] = item;
    queue->count = queue->count < queue->capacity ? queue->count + 1 : queue->capacity;
}

int ch_get_prev_get_id(ch_constant_memory_queue* queue) {
    int get_id = 0 < queue->get_id - 1 ? --queue->get_id : queue->capacity;
    return get_id;
}

int ch_get_next_get_id(ch_constant_memory_queue* queue) {
    int get_id = queue->capacity > queue->get_id + 1 ? ++queue->get_id : 0;
    return get_id;
}

void* ch_rollback(ch_constant_memory_queue* queue) {
    int get_id = ch_get_prev_get_id(queue);
    queue->count = queue->count > 0 ? queue->count - 1 : 0;
    return queue->data[get_id];
}

void* ch_dequeue(ch_constant_memory_queue* queue) {
    int get_id = ch_get_next_get_id(queue);
    queue->count = queue->count > 0 ? queue->count - 1 : 0;
    return queue->data[get_id];
}

int ch_queue_count(ch_constant_memory_queue* queue) {
    if (queue == NULL) return 0;
    return queue->count;
}

int ch_is_empty(ch_constant_memory_queue* queue) {
    if (queue == NULL) return 0;
    return queue->count == 0;
}

void ch_constant_memory_queue_peek_init(ch_constant_memory_queue* queue) {
    queue->peak_id = queue->get_id;
}

void* ch_constant_memory_queue_peek(ch_constant_memory_queue* queue) {
    int peak_id = queue->capacity > queue->peak_id + 1 ? ++queue->peak_id : 0;
    void* item = queue->data[peak_id];
    return item;
}

void ch_constant_memory_queue_replace(ch_constant_memory_queue* queue, void* item) {
    int peak_id = queue->put_id;
    queue->data[peak_id] = item;
}

void ch_destroy_constant_memory_queue(ch_constant_memory_queue* queue) {
    if (queue == NULL) return;
    for(int i = 0; i < queue->count; ++i) if (queue->data[i] != NULL) { FREE(queue->data[i]); queue->data[i] = NULL; }
    queue->capacity = 0;
    queue->count = 0;
    queue->peak_id = -1;
    queue->get_id = -1;
    queue->put_id = -1;
    FREE(queue->data);
    queue->data = NULL;queue->tree = NULL;
    FREE(queue);
    queue = NULL;
}

char *start_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

static float* empty_board = NULL;
static float* start_board = NULL;

typedef struct ch_dict_t {
    char *key;
    void *value;
    struct ch_dict_t *next;
    time_t stamp;
} ch_dict;

ch_dict **ch_dict_alloc() {
    return CALLOC(1, sizeof(ch_dict));
}

void ch_dict_dealloc(ch_dict **dict) {
    FREE(dict);
}

void* ch_dict_get(ch_dict *dict, char *key) {
    ch_dict *ptr;
    for (ptr = dict; ptr != NULL; ptr = ptr->next) {
        if (strcmp(ptr->key, key) == 0) {
            return ptr->value;
        }
    }
    return NULL;
}

void ch_dict_del(ch_dict **dict, char *key) {
    ch_dict *ptr, *prev;
    for (ptr = *dict, prev = NULL; ptr != NULL; prev = ptr, ptr = ptr->next) {
        if (strcmp(ptr->key, key) == 0) {
            if (ptr->next != NULL) {
                if (prev == NULL) {
                    *dict = ptr->next;
                } else {
                    prev->next = ptr->next;
                }
            } else if (prev != NULL) {
                prev->next = NULL;
            } else {
                *dict = NULL;
            }
            FREE(ptr->key);
            ptr->key = NULL;
            /*
            if (ptr->value != NULL) {
                FREE(ptr->value);
                ptr->value = NULL;
            }
            */
            FREE(ptr);
            ptr = NULL;
            return;
        }
    }
}

void ch_dict_exp(ch_dict **dict) {
    ch_dict *ptr, *prev;
    for (ptr = *dict, prev = NULL; ptr != NULL; prev = ptr, ptr = ptr->next) {
        double datetime_diff_ms = difftime(time(0), ptr->stamp) * 1000.;
        if (datetime_diff_ms > (60. * 60. * 1000.)) {
            if (ptr->next != NULL) {
                if (prev == NULL) {
                    *dict = ptr->next;
                } else {
                    prev->next = ptr->next;
                }
            } else if (prev != NULL) {
                prev->next = NULL;
            } else {
                *dict = NULL;
            }
            FREE(ptr->key);
            ptr->key = NULL;
            if (ptr->value != NULL) {
                FREE(ptr->value);
                ptr->value = NULL;
            }
            FREE(ptr);
            ptr = NULL;
            return;
        }
    }
}

void ch_dict_add(ch_dict **dict, char *key, void *value) {
    ch_dict_exp(dict);
    ch_dict *gd = ch_dict_get(*dict, key);
    if (gd != NULL)
    {
        gd->stamp = time(0);
        *dict = gd;
        return;
    }
    ch_dict *d = (ch_dict *) CALLOC(1, sizeof(ch_dict));
    d->key = (char*) CALLOC(strlen(key)+1, sizeof(char));
    strcpy(d->key, key);
    d->value = value;
    d->next = *dict;
    d->stamp = time(0);
    *dict = d;
}

static ch_dict* moves_history = NULL;
static ch_dict* moves_history_positions = NULL;

#define BOARD_SIZE (8*8+8)

typedef struct ch_learn_state_t {
    float board[BOARD_SIZE];
    int index;
    int player;
} ch_learn_state;

static ch_learn_state* empty_item = NULL;

void ch_init_game_history(char* sessionId) {
    if (empty_board == NULL) {
        empty_board = (float*) CALLOC(BOARD_SIZE, sizeof(float));
        for (int i = 0; i < BOARD_SIZE; ++i) {
            empty_board[i] = 0;
        }
    }
    if (start_board == NULL) {
        start_board = ch_fen_to_board(start_fen, 0);
    }
    if (empty_item == NULL) {
        empty_item = (ch_learn_state*) CALLOC(1, sizeof(ch_learn_state));
        empty_item->index = -1;
        memcpy(empty_item->board, empty_board, (1)*(BOARD_SIZE)*sizeof(float));
        empty_item->player = -1;
    }
    if (moves_history == NULL) {
        moves_history = *ch_dict_alloc();
    }
    if (ch_dict_get(moves_history, sessionId) == NULL) {
        ch_dict_add(&moves_history, sessionId, ch_create_constant_memory_queue(512));
    }
    if (moves_history_positions == NULL) {
        moves_history_positions = *ch_dict_alloc();
    }
    if (ch_dict_get(moves_history_positions, sessionId) == NULL) {
        ch_dict_add(&moves_history_positions, sessionId, ch_create_constant_memory_queue(512));
    }
}

void ch_put_back(char* sessionId, ch_learn_state* item) {
    ch_init_game_history(sessionId);
    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    ch_enqueue(q, item);
}

int trivial_player = -1;

void ch_clean_history(char* sessionId, int init) {
    ch_constant_memory_queue *q1 = ch_dict_get(moves_history, sessionId);
    ch_destroy_constant_memory_queue(q1);
    ch_dict_del(&moves_history, sessionId);
    ch_constant_memory_queue *q2 = ch_dict_get(moves_history_positions, sessionId);
    ch_destroy_constant_memory_queue(q2);
    ch_dict_del(&moves_history_positions, sessionId);
    if (init) {
        ch_learn_state *to_learn = (ch_learn_state *) CALLOC(1, sizeof(ch_learn_state));
        memcpy(to_learn->board, start_board, (1)*(BOARD_SIZE)*sizeof(float));
        to_learn->index = 0;
        to_learn->player = 0;
        ch_put_back(sessionId, to_learn);
    }
    if (trivial_player != -1) trivial_player = trivial_player == 0 ? 1 : 0;
}

ch_learn_state* ch_pick(char* sessionId) {
    ch_init_game_history(sessionId);
    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    ch_learn_state* queued = (ch_learn_state*)ch_rollback(q);
    return queued;
}

ch_learn_state* ch_pick_back(char* sessionId) {
    ch_init_game_history(sessionId);
    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    ch_learn_state* queued = (ch_learn_state*)ch_dequeue(q);
    return queued;
}

ch_moves ch_load_moves(char* valid_fen, char** valid_moves, int valid_moves_count)
{
    ch_moves m = {0};
    m.fen = valid_fen;
    m.moves = valid_moves;
    m.n = valid_moves_count;
    return m;
}

float* ch_network_predict(network *net, float *input, int player)
{
    memcpy(net->input, input, net->inputs*net->batch*sizeof(float));
    net->train = 0;
    forward_network(net);
    float* results = CALLOC(2, sizeof(float));
    results[0] = net->output[(0)];
    results[1] = net->output[(1)];
    return results;
}

typedef struct ch_position {
    char fen_board[128];
    int move_count;
} ch_position;

char* ch_parse_fen(char* fen) {
    int i = 0;
    while (fen[i] != ' ' && fen[i] != '\0') {
        i++;
    }
    char* fen_board = (char*) CALLOC(i+1, sizeof(char));
    strncpy(fen_board, fen, i);
    fen_board[i] = '\0';
    return fen_board;
}

float* ch_fen_to_board_with_history(char* sessionId, char* valid_fen, int n) {
    if (moves_history == NULL) {
        moves_history = *ch_dict_alloc();
    }
    if(ch_dict_get(moves_history, sessionId) == NULL) {
        ch_dict_add(&moves_history, sessionId, ch_create_constant_memory_queue(512));
    }

    float* value = (float*) CALLOC(n*(BOARD_SIZE), sizeof(float));

    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    int count = ch_queue_count(q);
    ch_constant_memory_queue_peek_init(q);

    float* valid_board = ch_fen_to_board(valid_fen, 1);

    for(int i = 0; i < n; ++i) {
        if (count < i) {
            ch_learn_state *current_position = ch_constant_memory_queue_peek(q);
            for (int k = 0; k < 8 * 8; ++k) value[i * 8 * 8 + k] = current_position->board[k];
        } else {
            for (int j = i; j < n; ++j) {
                for (int k = 0; k < 8 * 8; ++k) value[i * 8 * 8 + k] = valid_board[k];
            }
            break;
        }
    }

    FREE(valid_board);

    return value;
}

int ch_is_three_fold_repetition(char* sessionId, char* fen) {
    if (moves_history_positions == NULL) {
        moves_history_positions = *ch_dict_alloc();
    }

    if(ch_dict_get(moves_history_positions, sessionId) == NULL) {
        ch_dict_add(&moves_history_positions, sessionId, ch_create_constant_memory_queue(512));
    }

    if (fen == NULL || fen[0] == '\0') {
        ch_constant_memory_queue *q = ch_dict_get(moves_history_positions, sessionId);
        ch_destroy_constant_memory_queue(q);
        ch_dict_del(&moves_history_positions, sessionId);
        q = NULL;
        return 0;
    }

    ch_constant_memory_queue *q = ch_dict_get(moves_history_positions, sessionId);

    if (q == NULL) {
        q = ch_create_constant_memory_queue(512);
        ch_dict_add(&moves_history_positions, sessionId, q);
    }

    int count = ch_queue_count(q);

    if (count > 150) {
        count = ch_queue_count(q);
        for (int i = 0; i < count; ++i) {
            ch_position *peek_position = ch_dequeue(q);
        }
        return 1;
    }

    ch_constant_memory_queue_peek_init(q);

    char* position_fen = ch_parse_fen(fen);

    int fault = 0;
    for (int i = 0; i < count; ++i) {
        ch_position *current_position = ch_constant_memory_queue_peek(q);
        if (current_position != NULL && strlen(current_position->fen_board) != 0) {
            if (strcmp(position_fen, current_position->fen_board) == 0) {
                current_position->move_count += 1;
                if (current_position->move_count >= 3) {
                    fault = 1;
                    break;
                }
            }
        }
    }

    if (fault == 1) {
        count = ch_queue_count(q);
        for (int i = 0; i < count; ++i) {
            ch_position *peek_position = ch_dequeue(q);
        }
        FREE(position_fen);
        return fault;
    }

    ch_position* position = (ch_position*) CALLOC(1, sizeof(ch_position));
    strcpy(position->fen_board, position_fen);
    position->move_count = 1;
    ch_enqueue(q, position);

    FREE(position_fen);
    return fault;
}

int ch_end_move(char* sessionId, char *sfen, char *fen, char* move, int idx)
{
    if (idx == -1) return 0;
    if (fen == NULL || move == NULL) return 0;
    if (fen[0] == '\0' || move[0] == '\0') return 0;
    int is_draw_in_c = ch_is_three_fold_repetition(sessionId, fen);
    return is_draw_in_c || ch_is_end(sfen, fen, idx);
}

int ch_mate_move(char* sfen, char *ko, int idx)
{
    return ch_is_checkmate_move(sfen, ko, idx);
}

float ch_train_network_datum(network *net, int player)
{
    *net->seen += net->batch;
    net->train = 1;

    forward_network(net);
    backward_network(net);

    float error = *net->cost;

    update_network(net);

    return error;
}

float ch_train_network(network *net, float* truth, float* input, int player)
{
    memcpy(net->input, input, net->inputs*net->batch*sizeof(float));
    //memcpy(net->truth, truth, net->truths*net->batch*sizeof(float));
    //memcpy(net->output, truth, net->outputs*net->batch*sizeof(float));
    net->truth[0] = truth[0];
    net->truth[1] = truth[1];
    //net->output[0] = truth[0];
    //net->output[1] = truth[1];
    float loss = ch_train_network_datum(net, player);
    return loss;
}

float *ch_move(char* sfen, float *board, int indext) {
    char* pfen = ch_board_to_fen(board);
    char* mfen = ch_do_legal(sfen, pfen, indext);
    if (mfen == NULL) {
        FREE(pfen);
        fprintf(stderr, "try (%i) on %s\n", indext, pfen);
        return NULL;
    }
    float* mboard = ch_fen_to_board(mfen, 1);
    FREE(mfen);
    FREE(pfen);
    return mboard;
}

char *ch_move_fen(char* sfen, char* fen, int indext) {
    char* mfen = ch_do_legal(sfen, fen, indext);
    if (mfen == NULL) {
        fprintf(stderr, "try (%i) on %s\n", indext, fen);
    }
    return mfen;
}

void ch_self_study_after_pick_the_move(char* sessionId, network *net, int player, int level, int idx, char* valid_move, float *x, float *y) {
    float loss = ch_train_network(net, x,y, player);
    fprintf(stderr, "train step %ld(%s): sub-step: (%i) step-level: (%i) train: (%i) rate: (%.8g) loss: (%.8g)\n",
            net->nsteps, player == 0 ? "w" : "b", ch_queue_count(ch_dict_get(moves_history, sessionId)) + 1, level, idx,
            net->learning_rate, loss);
    if (loss != loss) {
        fprintf(stderr, "\nNaN LOSS detected! No possible to continue!\n");
        exit(4);
    }
}

void ch_self_study_train_self_step(char*sessionId, char *sfen, char *valid_fen, char *valid_move, network *net, int level, int idx, float pow, float val) {
    int player = strstr(valid_fen, " w ") ? 0 : 1;
    if (ch_is_checkmate(valid_fen)) return;
    if (valid_fen == NULL || valid_move == NULL) return;
    if (valid_fen[0] == '\0' || valid_move[0] == '\0') return;
    float *prev = ch_fen_to_board(valid_fen, 1);
    float *next = ch_move(sfen, prev, idx);
    if (next == NULL) {
        FREE(prev);
        return;
    }
    float *x = (float *) CALLOC((2)*(BOARD_SIZE), sizeof(float));
    float *y = (float *) CALLOC((2), sizeof(float));
    for (int i = 0; i < (1)*(BOARD_SIZE); ++i) {
        x[i] = prev[i];
    }
    for (int i = 0; i < (1)*(BOARD_SIZE); ++i) {
        x[(1)*(BOARD_SIZE) + i] = next[i];
    }
    float *pnext = ch_move(sfen, prev, idx);
    float powW = 0;
    float powB = 0;
    float power = ch_eval_the_board(sfen, pnext, &powW, &powB);
    float value = (player == 0 ? powW - powB : powB - powW);
    y[0] = power < pow ? pow : power;
    y[1] = value < val ? val : value;
    FREE(pnext);
    fprintf(stderr, "output[0]: %2.8f, output[1]: %2.8f\n", y[0], y[1]);
    ch_self_study_after_pick_the_move(sessionId, net, player, level, idx, valid_move, x, y);
    FREE(y);
    FREE(x);
    FREE(next);
    FREE(prev);
}

float *ch_copy_board(float *board)
{
    float *next = (float*) CALLOC(BOARD_SIZE, sizeof(float));
    memcpy(next, board, (BOARD_SIZE)*sizeof(float));
    return next;
}

void ch_swap(float* b, int i, int j) {
    float swap = b[i];
    b[i] = b[j];
    b[j] = swap;
}

void ch_flip_board(float *board)
{
    int i;
    for(i = 0; i < 8*8; ++i) {
        board[i] = -1 * board[i];
    }
    board[8*8] = board[8*8] == 0 ? 1 : 0;
    ch_swap(board, 8*8+1, 8*8+3);
    ch_swap(board, 8*8+2, 8*8+4);
    board[8*8+5] = board[8*8+5] == 0 ? 0.f : (float)(64 - (int)board[8*8+5]);
    ch_swap(board, 8*8+6, 8*8+7);
}
// disabled!
void ch_train_till_end_winner(char *sessionId, char* sfen, char *valid_move, network *net, int player) {
    return; // TODO !!
    fprintf(stderr, "train winner till move: %s\n", valid_move);
    int step = 0;
    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    ch_constant_memory_queue_peek_init(q);
    int count = ch_queue_count(q);
    while ((count -= 2) >= 2) {
        ch_learn_state *history_move = ch_constant_memory_queue_peek(q);
        if (history_move != NULL && history_move->player == player) {
            ch_learn_state *history_move_next = ch_constant_memory_queue_peek(q);
            if (history_move_next != NULL && history_move_next->player == !player) {
                ++step;
                float *prev = history_move->board;
                float *next = history_move_next->board;
                float *x = (float *) CALLOC((0)+(2)*(BOARD_SIZE), sizeof(float));
                float *y = (float *) CALLOC((2), sizeof(float));
                for (int i = 0; i < (1)*(BOARD_SIZE); ++i) {
                    x[2 * i] = prev[i];
                    x[2 * i + 1] = next[i];
                }
                float power = ch_eval_the_board(sfen, next, &y[0], &y[1]);
                if (player == 0) y[0] += 1.0f;
                if (player == 1) y[1] += 1.0f;
                char* valid_fen_next = ch_board_to_fen(history_move->board);
                char* valid_fen = ch_board_to_fen(history_move_next->board);
                float loss = ch_train_network(net, x, y, player);
                fprintf(stderr, "train: step: %ld(%s) sub-step: (%i) train: (%i) rate: (%.8g) loss: (%.8g)\n",
                        net->nsteps, player == 0 ? "w" : "b", step, history_move_next->index, net->learning_rate,
                        loss);
                if (loss != loss) {
                    fprintf(stderr, "\nNaN LOSS detected! No possible to continue!\n");
                    exit(2);
                }
                FREE(valid_fen);
                FREE(valid_fen_next);
            }
        }
    }
}
// disabled!
void ch_train_till_end_looser(char *sessionId, char* sfen, char *valid_move, network *net, int player) {
    return; // TODO !!
    fprintf(stderr, "train looser till move: %s\n", valid_move);
    int step = 0;
    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    ch_constant_memory_queue_peek_init(q);
    ch_learn_state *history_move_first_to_drop = ch_constant_memory_queue_peek(q);
    int count = ch_queue_count(q);
    while ((count -= 2) >= 2) {
        ch_learn_state *history_move = ch_constant_memory_queue_peek(q);
        if (history_move != NULL && history_move->player == !player) {
            ch_learn_state *history_move_next = ch_constant_memory_queue_peek(q);
            if (history_move_next != NULL && history_move_next->player == player) {
                ++step;
                float *prev = history_move->board;
                float *next = history_move_next->board;
                float *x = (float *) CALLOC((0)+(2)*(BOARD_SIZE), sizeof(float));
                float *y = (float *) CALLOC((2), sizeof(float));
                for (int i = 0; i < (1)*(BOARD_SIZE); ++i) {
                    x[2 * i] = prev[i];
                    x[2 * i + 1] = next[i];
                }
                y[0] = player == 0 ? +0.0000000001f : -0.0000000001f;
                y[1] = player == 1 ? +0.0000000001f : -0.0000000001f;
                char* valid_fen_next = ch_board_to_fen(history_move->board);
                char* valid_fen = ch_board_to_fen(history_move_next->board);
                float loss = ch_train_network(net, x, y, player);
                fprintf(stderr, "train: step: %ld(%s) sub-step: (%i) train: (%i) rate: (%.8g) loss: (%.8g)\n",
                        net->nsteps, player == 0 ? "w" : "b", step, history_move_next->index, net->learning_rate,
                        loss);
                if (loss != loss) {
                    fprintf(stderr, "\nNaN LOSS detected! No possible to continue!\n");
                    exit(2);
                }
                FREE(valid_fen);
                FREE(valid_fen_next);
            }
        }
    }
}

void ch_put_into_game_queue(char *sessionId, char *valid_fen, char* valid_move, int indext, network* net, char* sfen) {
    int player = strstr(valid_fen, " w ") ? 0 : 1;
    ch_constant_memory_queue *q = ch_dict_get(moves_history, sessionId);
    ch_learn_state* to_learn = (ch_learn_state *) CALLOC(1, sizeof(ch_learn_state));
    float* fen_board = ch_fen_to_board(valid_fen, 1);
    memcpy(to_learn->board, fen_board, (1)*(BOARD_SIZE)*sizeof(float));
    FREE(fen_board);
    to_learn->index = indext;
    to_learn->player = player;
    ch_put_back(sessionId, to_learn);
}

#define BOARD_SIZE (8*8+8)

typedef struct {
    network* net;
    char* sfen;
} ch_adapter_ctx;

static ch_adapter_ctx gctx;

static int api_gen_legal(const ch_board* board, ch_mv* moves, int max_moves) {
    char* valid_fen = NULL;
    char** valid_moves = NULL;
    int cnt = 0;
    int ok = ch_get_all_valid_moves(gctx.sfen, (char*)board->fen, &valid_fen, &valid_moves, &cnt);
    if (!ok || cnt <= 0) {
        if (valid_moves) { for (int i = 0; i < cnt; ++i) FREE(valid_moves[i]); FREE(valid_moves); }
        if (valid_fen) FREE(valid_fen);
        return 0;
    }
    int n = cnt < max_moves ? cnt : max_moves;
    for (int i = 0; i < n; ++i) moves[i] = (ch_mv)i;
    for (int i = 0; i < cnt; ++i) FREE(valid_moves[i]);
    FREE(valid_moves);
    FREE(valid_fen);
    return n;
}

static int api_apply_move(const ch_board* board, ch_mv mv, ch_board* out) {
    char* mfen = ch_move_fen(gctx.sfen, (char*)board->fen, (int)mv);
    if (!mfen) return -1;
    strncpy(out->fen, mfen, sizeof(out->fen)-1);
    out->fen[sizeof(out->fen)-1] = '\0';
    FREE(mfen);
    return 0;
}

static int api_terminal_result(const ch_board* board) {
    if (ch_is_checkmate((char*)board->fen)) return -1;
    return 2;
}

static int api_nn_eval(network* net, const ch_board* board, const ch_mv* moves, int moves_count, float* out_policy, float* out_value) {
    float* prev = ch_fen_to_board((char*)board->fen, 1);
    int player = strstr((char*)board->fen, " w ") ? 0 : 1;
    float* x = (float*)calloc((2)*(BOARD_SIZE), sizeof(float));
    for (int i = 0; i < (BOARD_SIZE); ++i) x[i] = prev[i];
    float sumP = 0.f;
    float maxP = 0.f;
    for (int i = 0; i < moves_count; ++i) {
        char* mfen = ch_move_fen(gctx.sfen, (char*)board->fen, (int)moves[i]);
        float* next = ch_fen_to_board(mfen, 1);
        for (int k = 0; k < (1)*(BOARD_SIZE); ++k) x[(1)*(BOARD_SIZE) + k] = next[k];
        float* y = ch_network_predict(net, x, player);
        out_policy[i] = fmaxf(0.f, y[0]);
        if (out_policy[i] > maxP) maxP = out_policy[i];
        sumP += out_policy[i];
        FREE(y);
        FREE(next);
        FREE(mfen);
    }
    if (sumP <= 0.f) {
        for (int i = 0; i < moves_count; ++i) out_policy[i] = 1.f / (float)moves_count;
    } else {
        float inv = 1.f / sumP;
        for (int i = 0; i < moves_count; ++i) out_policy[i] *= inv;
    }
    float powW = 0.f, powB = 0.f;
    float v = ch_eval_the_board(gctx.sfen, prev, &powW, &powB);
    if (player == 0) *out_value = powW - powB; else *out_value = powB - powW;
    FREE(x);
    FREE(prev);
    return 0;
}

static ch_api make_api() {
    ch_api api;
    api.gen_legal = (int(*)(const ch_board*, ch_mv*, int))api_gen_legal;
    api.apply_move = (int(*)(const ch_board*, ch_mv, ch_board*))api_apply_move;
    api.terminal_result = (int(*)(const ch_board*))api_terminal_result;
    api.nn_eval = (int(*)(network*, const ch_board*, const ch_mv*, int, float*, float*))api_nn_eval;
    return api;
}

int ch_pick_move_mcts(char* sessionId, char* sfen, char* valid_fen, char** valid_moves, ch_moves ch_m, network *net, int n, int level, int *solver, float *pow, float *val)
{
    gctx.net = net;
    gctx.sfen = sfen;
    ch_mcts_config cfg;
    cfg.cpuct = 0.72f;
    cfg.dirichlet_alpha = 0.3f;
    cfg.root_noise_frac = 0.25f;
    cfg.max_children = 32;
    cfg.use_virtual_loss = 0;
    cfg.virtual_loss = 1.0f;
    static unsigned char arena[8 * 1024 * 1024];
    ch_mcts m;
    ch_api api = make_api();
    ch_mcts_init(&m, api, cfg, net, arena, sizeof(arena));
    ch_board rootb;
    strncpy(rootb.fen, valid_fen, sizeof(rootb.fen)-1);
    rootb.fen[sizeof(rootb.fen)-1] = '\0';
    int to_move = strstr(valid_fen, " w ") ? +1 : -1;
    ch_mcts_node* root = ch_mcts_create_root(&m, &rootb, to_move);
    int sims = (level < 0 ? 0 : level) + 1;
    ch_print_board(valid_fen);
    ch_mcts_play_and_log(&m, root, level + 1, valid_moves);
    ch_mv mv = ch_mcts_pick_move(&m, root, 0.0f);
    int idx = (int)mv;
    if (idx < 0 || idx >= n) idx = 0;
    *solver = 2;
    float* prev = ch_fen_to_board(valid_fen, 1);
    float* next = ch_move(sfen, prev, idx);
    float powW=0.f, powB=0.f;
    ch_eval_the_board(sfen, next, &powW, &powB);
    if (strstr(valid_fen, " w ")) { *pow = powW; *val = powW - powB; } else { *pow = powB; *val = powB - powW; }
    FREE(next);
    FREE(prev);
    return idx;
}

static char *ch_lin_in_dir;
static char *ch_lin_out_dir;
static network  *ch_lin_net;
static char* ch_lin_weight_file;

#ifndef __linux__

int ch_process_file(char *file_name) {

    network* net  = ch_lin_net;
    char* ch_weight_file = ch_lin_weight_file;
    if (++net->nsteps % 1000 == 0) save_weights(net, ch_weight_file);

    char* in_dir  = ch_lin_in_dir;
    char* out_dir = ch_lin_out_dir;

    fprintf(stderr, "fn: %s\n", file_name);

    char fname[1024];
    char ffiname[1024];
    char ffoname[1024];

    strcpy(fname, file_name);

    strcpy(ffiname, ch_lin_in_dir);
    strcat(ffiname, "/");
    strcat(ffiname, fname);
    fname[strlen(ch_lin_in_dir) + strlen("/") + strlen(fname) + 1] = '\0';

    strcpy(ffoname, ch_lin_out_dir);
    strcat(ffoname, "/");
    strcat(ffoname, fname);
    fname[strlen(ch_lin_out_dir) + strlen("/") + strlen(fname) + 1] = '\0';

    struct stat ch_st = {0};
    off_t size = 0;
    off_t offs = 0;
    do {
        offs = size;
        stat(ffiname, &ch_st);
        size = ch_st.st_size;
        if (offs != size) usleep(250); else break;
    } while (1);

    char* pw = " w ";
    char* pb = " b ";

    char* sessionId = NULL;
    char* fen = NULL;
    char* fen_move = NULL;
    char* level = NULL;
    char* sfen = NULL;
    int mlevel = -1;

    int solver = 0;

    if (ch_fopen(ffiname, &sessionId, &fen, &fen_move, &level, &sfen)) {

        if (fen == NULL || fen[0] == '\0') {
            if (fen) FREE(fen);
            fen = CALLOC(strlen(sfen) + 1, sizeof(char));
            strcpy(fen, sfen);
        }

        fprintf(stderr, "sessionId: %s\n", sessionId);
        fprintf(stderr, "level: %s\n", level);
        fprintf(stderr, "sfen: %s\n", sfen);
        fprintf(stderr, "fen: %s\n", fen);

        ch_init_game_history(sessionId);

        if (fen_move != NULL && strcmp(fen_move, "") != 0) {
            fprintf(stderr, "pgn: %s\n", fen_move);
            char *fen_next = NULL;
            int fen_next_idx = 0;
            int fen_next_cnt = 0;
            if (ch_board_after_move(sfen, fen, fen_move, &fen_next, &fen_next_idx, &fen_next_cnt)) {
                if (fen_next == NULL) {
                    ch_fsave(ffoname, sessionId, NULL, NULL, level, sfen, solver);
                    return 1;
                }
                ch_print_board(fen_next);
                FREE(fen_next);
            }
        }
        mlevel = level != NULL ? (int)atoi(level) : 3;

        int valid_moves_count = 0;
        char **valid_moves = NULL;
        char* valid_fen = NULL;

        if (fen != NULL && fen[0] != '\0' && ch_get_all_valid_moves(sfen, fen, &valid_fen, &valid_moves, &valid_moves_count)) {

            int indext = -1;
            int player = strstr(valid_fen, " w ") ? 0 : 1;

            net->nsteps++;

            if (indext == -1 && fen_move != NULL) {
                for (int i = 0; i < valid_moves_count; ++i) {
                    if (strcmp(valid_moves[i], fen_move) == 0) {
                        indext = i;
                        break;
                    }
                }
            }

            if (ch_end_move(sessionId, sfen, fen, valid_moves[indext], indext)) {
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);
                ch_fsave(ffoname, sessionId, valid_fen, "", level, sfen, solver);
                ch_clean_history(sessionId, 1);
                for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
                FREE(valid_moves);
                FREE(valid_fen);
                return 0;
            }

            if (indext == -1) {
                ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
                float pow = 0.f;
                float value = 0.f;
                indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, net, valid_moves_count, mlevel, &solver, &pow, &value);
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);
                ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], net, mlevel, indext, pow, value);
                ch_fsave(ffoname, sessionId, valid_fen, valid_moves[indext], level, sfen, solver);
                for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
                FREE(valid_moves);
                FREE(valid_fen);
                return 0;
            }

            //ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);

            strcpy(fen, valid_fen);
            if (fen_move != NULL) strcpy(fen_move, valid_moves[indext]);

            FREE(valid_fen); valid_fen = NULL;
            for (int i = 0; i < valid_moves_count; ++i) {FREE(valid_moves[i]); valid_moves[i] = NULL; } valid_moves_count = 0;
            FREE(valid_moves); valid_moves = NULL;

            valid_moves_count = 0;
            valid_moves = NULL;
            valid_fen = NULL;

            if (ch_get_all_valid_moves_after(sfen, fen, fen_move, &valid_fen, &valid_moves, &valid_moves_count)) {

                indext = -1;
                player = strstr(valid_fen, " w ") ? 0 : 1;

                ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
                float pow = 0.f;
                float value = 0.f;
                indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, net, valid_moves_count, mlevel, &solver, &pow, &value);
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);
                ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], net, mlevel, indext, pow, value);

                if (valid_moves[indext] != NULL && strcmp(valid_moves[indext], "") != 0) {
                    char *fen_next = NULL;
                    int fen_next_idx = 0;
                    int fen_next_cnt = 0;
                    if (ch_board_after_move(sfen, valid_fen, valid_moves[indext], &fen_next, &fen_next_idx,&fen_next_cnt)) {
                        FREE(fen_next);
                    }
                }

                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);
                ch_fsave(ffoname, sessionId, valid_fen, valid_moves[indext], level, sfen, solver);

                FREE(valid_fen);
                for (int i = 0; i < valid_moves_count; ++i) FREE(valid_moves[i]);
                FREE(valid_moves);

            } else {
                ch_fsave(ffoname, sessionId, NULL, NULL, level, sfen, solver);
            }
        }

        FREE(sfen);
        FREE(fen_move);
        FREE(fen);
        FREE(sessionId);
    }

    remove(ffiname);

    if (net->nsteps % 10000 == 0) save_weights(net, ch_weight_file);

    return 0;
}

void test_dchess(int argc, char **argv, char *cfgfile, char *weight_file, char *in_dir, char *out_dir) {
    srandom(time(0));

    network *net;

    if (!exists(weight_file)) {
        net = parse_network_cfg(cfgfile);
        net->nsteps = 0;
        save_weights(net, weight_file);
    }

    ch_init_game_history("13a25e80-ece3-4a4b-9347-e6df74386d02");

    if (exists(weight_file)) {
        if (ngpusg > 1) {
            opencl_set_device(gpusg[1]);
        }
        else {
            opencl_set_device(0);
        }
        net = load_network(cfgfile, weight_file, 0);
        if (ngpusg > 1) {
            net->gpu_index = gpusg[1];
        }
        else {
            net->gpu_index = 0;
        }
    }

    set_batch_network(net, 1);
    char *ch_weight_file = weight_file;

    ch_lin_net = net;
    ch_lin_weight_file = weight_file;

    ch_lin_in_dir = in_dir;
    ch_lin_out_dir = out_dir;

    const char* patterns[] = {"*.json"};
    while (!init_notified_file_name(in_dir, patterns, ch_process_file));
}

#else

int ch_process_file(char *file_name) {
    network* net  = ch_lin_net;
    char* ch_weight_file = ch_lin_weight_file;
    if (++net->nsteps % 1000 == 0) save_weights(net, ch_weight_file);

    char* in_dir  = ch_lin_in_dir;
    char* out_dir = ch_lin_out_dir;

    fprintf(stderr, "fn: %s\n", file_name);

    char fname[1024];
    char ffiname[1024];
    char ffoname[1024];

    strcpy(fname, file_name);

    strcpy(ffiname, ch_lin_in_dir);
    strcat(ffiname, "/");
    strcat(ffiname, fname);
    fname[strlen(ch_lin_in_dir) + strlen("/") + strlen(fname) + 1] = '\0';

    strcpy(ffoname, ch_lin_out_dir);
    strcat(ffoname, "/");
    strcat(ffoname, fname);
    fname[strlen(ch_lin_out_dir) + strlen("/") + strlen(fname) + 1] = '\0';

    struct stat ch_st = {0};
    off_t size = 0;
    off_t offs = 0;
    do {
        offs = size;
        stat(ffiname, &ch_st);
        size = ch_st.st_size;
        if (offs != size) usleep(500); else break;
    } while (1);

    char* pw = " w ";
    char* pb = " b ";

    char* sessionId = NULL;
    char* fen = NULL;
    char* fen_move = NULL;
    char* level = NULL;
    char* sfen = NULL;
    int mlevel = -1;

    int solver = 0;

    if (ch_fopen(ffiname, &sessionId, &fen, &fen_move, &level, &sfen)) {

        if (fen == NULL || fen[0] == '\0') {
            if (fen) FREE(fen);
            fen = CALLOC(strlen(sfen) + 1, sizeof(char));
            strcpy(fen, sfen);
        }

        fprintf(stderr, "sessionId: %s\n", sessionId);
        fprintf(stderr, "level: %s\n", level);
        fprintf(stderr, "sfen: %s\n", sfen);
        fprintf(stderr, "fen: %s\n", fen);

        ch_init_game_history(sessionId);

        if (fen_move != NULL && strcmp(fen_move, "") != 0) {
            fprintf(stderr, "pgn: %s\n", fen_move);
            char *fen_next = NULL;
            int fen_next_idx = 0;
            int fen_next_cnt = 0;
            if (ch_board_after_move(sfen, fen, fen_move, &fen_next, &fen_next_idx, &fen_next_cnt)) {
                if (fen_next == NULL) {
                    ch_fsave(ffoname, sessionId, NULL, NULL, level, sfen, solver);
                    return 1;
                }
                ch_print_board(fen_next);
                FREE(fen_next);
            }
        }

        if (level != NULL) {
            mlevel = atoi(level);
        } else {
            mlevel = 3;
        }

        int valid_moves_count = 0;
        char **valid_moves = NULL;
        char* valid_fen = NULL;

        if (fen != NULL && fen[0] != '\0' && ch_get_all_valid_moves(sfen, fen, &valid_fen, &valid_moves, &valid_moves_count)) {

            int indext = -1;
            int player = strstr(valid_fen, " w ") ? 0 : 1;

            net->nsteps++;

            if (indext == -1 && fen_move != NULL) {
                for (int i = 0; i < valid_moves_count; ++i) {
                    if (strcmp(valid_moves[i], fen_move) == 0) {
                        indext = i;
                        break;
                    }
                }
            }

            if (ch_end_move(sfen, sessionId, fen, valid_moves[indext], indext)) {
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);
                ch_fsave(ffoname, sessionId, valid_fen, "", level, sfen, solver);
                ch_clean_history(sessionId, 1);
                for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
                FREE(valid_moves);
                FREE(valid_fen);
                return 0;
            }

            if (indext == -1) {
                ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
                float pow = 0.f;
                float value = 0.f;
                indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, net, valid_moves_count, mlevel, &solver, &pow, &value);
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);
                ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], net, mlevel, indext, pow, value);
                ch_fsave(ffoname, sessionId, valid_fen, valid_moves[indext], level, sfen, solver);
                for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
                FREE(valid_moves);
                FREE(valid_fen);
                return 0;
            }

            //ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);

            strcpy(fen, valid_fen);
            if (fen_move != NULL) strcpy(fen_move, valid_moves[indext]);

            FREE(valid_fen); valid_fen = NULL;
            for (int i = 0; i < valid_moves_count; ++i) {FREE(valid_moves[i]); valid_moves[i] = NULL; } valid_moves_count = 0;
            FREE(valid_moves); valid_moves = NULL;

            valid_moves_count = 0;
            valid_moves = NULL;
            valid_fen = NULL;

            if (ch_get_all_valid_moves_after(sfen, fen, fen_move, &valid_fen, &valid_moves, &valid_moves_count)) {

                indext = -1;
                player = strstr(valid_fen, " w ") ? 0 : 1;

                ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
                float pow = 0.f;
                float value = 0.f;
                indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, net, valid_moves_count, mlevel, &solver, &pow, &value);
                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);
                ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], net, mlevel, indext, pow, value);

                if (valid_moves[indext] != NULL && strcmp(valid_moves[indext], "") != 0) {
                    char *fen_next = NULL;
                    int fen_next_idx = 0;
                    int fen_next_cnt = 0;
                    if (ch_board_after_move(sfen, valid_fen, valid_moves[indext], &fen_next, &fen_next_idx, &fen_next_cnt)) {
                        FREE(fen_next);
                    }
                }

                ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);
                ch_fsave(ffoname, sessionId, valid_fen, valid_moves[indext], level, sfen, solver);

                FREE(valid_fen);
                for (int i = 0; i < valid_moves_count; ++i) FREE(valid_moves[i]); valid_moves_count = 0;
                FREE(valid_moves);

            } else {
                ch_fsave(ffoname, sessionId, NULL, NULL, level, sfen, solver);
            }
        }

        FREE(sfen);
        FREE(fen_move);
        FREE(fen);
        FREE(sessionId);
    }

    remove(ffiname);

    return 0;
}

void test_dchess(int argc, char **argv, char *cfgfile, char *weight_file, char *in_dir, char *out_dir) {
    srandom(time(0));

    network *net;

    if (!exists(weight_file)) {
        net = parse_network_cfg(cfgfile);
        net->nsteps = 0;
        save_weights(net, weight_file);
    }

    ch_init_game_history("13a25e80-ece3-4a4b-9347-e6df74386d02");

    if (exists(weight_file)) {
        if (ngpusg > 1) {
            opencl_set_device(gpusg[1]);
        }
        else {
            opencl_set_device(0);
        }
        net = load_network(cfgfile, weight_file, 0);
        if (ngpusg > 1) {
            net->gpu_index = gpusg[1];
        }
        else {
            net->gpu_index = 0;
        }
    }
    set_batch_network(net, 1);
    char *ch_weight_file = weight_file;

    ch_lin_net = net;
    ch_lin_weight_file = weight_file;

    ch_lin_in_dir = in_dir;
    ch_lin_out_dir = out_dir;

    const char* patterns[] = {"*.json"};
    while (!init_notified_file_name(in_dir, patterns, ch_process_file));
}

#endif

typedef struct ch_board_state {
    char fen[128];
    char move[16];
    int final;
    int indext;
} ch_board_state;

ch_board_state ch_self_learn_step(char* sessionId, char* sfen, int level, network *net, char *fen, char *fen_move, int learn) {

    int player = strstr(fen, " w ") ? 0 : 1;

    ch_board_state return_value = {0};
    return_value.final = 0;

    int valid_moves_count = 0;
    char **valid_moves = NULL;
    char* valid_fen = NULL;

    if (ch_get_all_valid_moves(sfen, fen, &valid_fen, &valid_moves, &valid_moves_count)) {

        int indext = -1;

        if (indext == -1 && fen_move != NULL && fen_move[0] != '\0') {
            for (int i = 0; i < valid_moves_count; ++i) {
                if (strcmp(valid_moves[i], fen_move) == 0) {
                    indext = i;
                    break;
                }
            }
        }

        if (ch_end_move(sessionId, sfen, fen, valid_moves[indext], indext)) {
            ch_put_into_game_queue(sessionId, fen, valid_moves[indext], indext, net, sfen);
            strcpy(return_value.fen, valid_fen);
            strcpy(return_value.move, valid_moves[indext]);
            return_value.indext = indext;
            for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
            FREE(valid_moves);
            FREE(valid_fen);
            return_value.final = 1;
            return return_value;
        }

        if (indext == -1) {
            ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
            int solver = 0;
            float pow = 0.f;
            float value = 0.f;
            indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, net, valid_moves_count, level, &solver, &pow, &value);
            ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);
            if (learn == 1 && trivial_player == player) ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], net, level, indext, pow, value);
            strcpy(return_value.fen, valid_fen);
            strcpy(return_value.move, valid_moves[indext]);
            return_value.indext = indext;
            for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
            FREE(valid_moves);
            FREE(valid_fen);
            return return_value;
        }

        strcpy(fen, valid_fen);
        strcpy(fen_move, valid_moves[indext]);

        for (int j = 0; j < valid_moves_count; ++j) FREE(valid_moves[j]);
        FREE(valid_moves);
        FREE(valid_fen);

        valid_moves_count = 0;
        valid_moves = NULL;
        valid_fen = NULL;

        if (ch_get_all_valid_moves_after(sfen, fen, fen_move, &valid_fen, &valid_moves, &valid_moves_count)) {

            indext = -1;
            player = strstr(valid_fen, " w ") ? 0 : 1;

            ch_moves ch_m = ch_load_moves(valid_fen, valid_moves, valid_moves_count);
            int solver = 0;
            float pow = 0.f;
            float value = 0.f;
            indext = ch_pick_move_mcts(sessionId, sfen, valid_fen, valid_moves, ch_m, net, valid_moves_count, level, &solver, &pow, &value);
            ch_put_into_game_queue(sessionId, valid_fen, valid_moves[indext], indext, net, sfen);
            if (learn == 1 && trivial_player == player) ch_self_study_train_self_step(sessionId, sfen, valid_fen, valid_moves[indext], net, level, indext, pow, value);
            strcpy(return_value.fen, valid_fen);
            strcpy(return_value.move, valid_moves[indext]);
            return_value.indext = indext;
            FREE(valid_fen); valid_fen = NULL;
            for (int i = 0; i < valid_moves_count; ++i) {FREE(valid_moves[i]); valid_moves[i] = NULL; }
            FREE(valid_moves); valid_moves = NULL;
            return return_value;
        }

        strcpy(return_value.fen, valid_fen);
        strcpy(return_value.move, valid_moves[indext]);
        return_value.indext = indext;

        FREE(valid_fen); valid_fen = NULL;
        for (int i = 0; i < valid_moves_count; ++i) {FREE(valid_moves[i]); valid_moves[i] = NULL; }
        FREE(valid_moves); valid_moves = NULL;
    }

    return return_value;
}

void test_tchess(int argc, char **argv, char *cfgfile, char *weight_file) {
    srandom(time(0));

    network *net;

    if (!exists(weight_file)) {
        net = parse_network_cfg(cfgfile);
        net->nsteps = 0;
        save_weights(net, weight_file);
    }

    char* sessionId = "13a25e80-ece3-4a4b-9347-e6df74386d02";
    ch_init_game_history(sessionId);

    if (exists(weight_file)) {
        if (ngpusg > 1) {
            opencl_set_device(gpusg[1]);
        } else {
            opencl_set_device(0);
        }
        net = load_network(cfgfile, weight_file, 0);
        if (ngpusg > 1) {
            net->gpu_index = gpusg[1];
        } else {
            net->gpu_index = 0;
        }
    }
    set_batch_network(net, 1);

    char *ch_weight_file = weight_file;

    char valid_fen[128];
    char valid_fen_move[8];
    char sfen[128];

    ch_board_state move_state = {0};

    int mcount = 0;
    int mlevel = 4;

    char* startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    if (0) {
        startpos = "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3";
        float* fboard = ch_fen_to_board(startpos, 1);
        char* cfen = ch_board_to_fen(fboard);
        if (strcmp(cfen, startpos) != 0) {
            fprintf(stderr, "%s\n", startpos);
            ch_print_board(startpos);
            float *board = ch_fen_to_board(startpos, 1);
            char *fen = ch_board_to_fen(board);
            fprintf(stderr, "%s\n", fen);
            FREE(fen);
            FREE(board);
            exit(1);
        }
        FREE(cfen);
        FREE(fboard);
    }

    if (0) {
        startpos = "rnbqkbnr/pp2pppp/8/2ppP3/8/8/PPPP1PPP/RNBQKBNR w HAha d6 0 3";
        float* fboard = ch_fen_to_board(startpos, 1);
        char* cfen = ch_board_to_fen(fboard);
        if (strcmp(cfen, startpos) != 0) {
            fprintf(stderr, "%s\n", startpos);
            ch_print_board(startpos);
            float *board = ch_fen_to_board(startpos, 1);
            char *fen = ch_board_to_fen(board);
            fprintf(stderr, "%s\n", fen);
            FREE(fen);
            FREE(board);
            exit(1);
        }
        FREE(cfen);
        FREE(fboard);
    }

    if (0) {
        char* test_fen = "1kr5/3n4/q3p2p/p2n2p1/PppB1P2/5BP1/1P2Q2P/3R2K1 w - - 0 1";
        move_state = ch_self_learn_step(sessionId, sfen, 24, net, test_fen, "", 0);
        assert(strcmp("f4f5", move_state.move) == 0);
    }

    int player = 0;
    trivial_player = 0;
    one_more_time:
    player = 0;
    ch_clean_history(sessionId, 1);
    strcpy(valid_fen, startpos);
    strcpy(valid_fen_move, "");
    if (++ch_test_tchess_count % 27 != 0) {
        char* fen960 = ch_get_fen_960();
        strcpy(valid_fen, fen960);
        FREE(fen960);
    }
    mlevel = net->nsteps < net->burn_in ? 1 :  4;
    strcpy(sfen, valid_fen);
    while (net->nsteps < net->max_batches) {
        move_state = ch_self_learn_step(sessionId, sfen, mlevel, net, valid_fen, valid_fen_move, 1);
        strcpy(valid_fen, move_state.fen);
        strcpy(valid_fen_move, move_state.move);
        if (ch_end_move(sessionId, sfen, valid_fen, valid_fen_move, move_state.indext)) {
            ch_put_into_game_queue(sessionId, valid_fen, valid_fen_move, move_state.indext, net, sfen);
            save_weights(net, ch_weight_file);
            move_state.final = 0;
            player = player == 0 ? 1 : 0;
            if (trivial_player != -1) trivial_player = trivial_player == 0 ? 1 : 0;
            goto one_more_time;
        }
        float* board = ch_fen_to_board(valid_fen, 1);
        float* next = ch_move(sfen, board, move_state.indext);
        if (next) {
            char* next_fen = ch_board_to_fen(next);
            strcpy(move_state.fen, next_fen);
            FREE(next_fen);
        }
        FREE(next);
        FREE(board);
        if (++net->nsteps % 10000 == 0) save_weights(net, ch_weight_file);
        if (move_state.final == 1) {
            move_state.final = 0;
            player = player == 0 ? 1 : 0;
            if (trivial_player != -1) trivial_player = trivial_player == 0 ? 1 : 0;
            goto one_more_time;
        }
    }
    free_network(net);
}

void ch_train_possible_checkmate(char *sessionId, ch_board_state move_state, char *sfen, network *net, int level, int deep) {
    if (deep > 0) {
        int player = strstr(move_state.fen, " w ") ? 0 : 1;
        char *valid_maybe_fen = NULL;
        char **valid_maybe_moves = NULL;
        int valid_maybe_moves_count = 0;
        if (ch_get_all_valid_moves(sfen, move_state.fen, &valid_maybe_fen, &valid_maybe_moves, &valid_maybe_moves_count) && valid_maybe_moves_count > 0) {
            for (int i = 0; i < valid_maybe_moves_count; ++i) {
                if (ch_mate_move(sfen, valid_maybe_fen, i)) {
                    ch_learn_state *to_maybe_learn = (ch_learn_state *) CALLOC(1, sizeof(ch_learn_state));
                    float *fen_maybe_board = ch_fen_to_board(valid_maybe_fen, 1);
                    memcpy(to_maybe_learn->board, fen_maybe_board, (1)*(BOARD_SIZE)*sizeof(float));
                    to_maybe_learn->index = i;
                    to_maybe_learn->player = player;
                    ch_put_back(sessionId, to_maybe_learn);
                    ch_train_till_end_winner(sessionId, sfen, valid_maybe_moves[i], net, +player);
                    ch_train_till_end_looser(sessionId, sfen, valid_maybe_moves[i], net, !player);
                    ch_pick_back(sessionId);
                    FREE(fen_maybe_board);
                } else {
                    ch_board_state move_maybe_next_state = {0};
                    strcpy(move_maybe_next_state.fen, valid_maybe_fen);
                    strcpy(move_maybe_next_state.move, valid_maybe_moves[i]);
                    move_maybe_next_state.indext = i;
                    ch_learn_state *to_maybe_learn = (ch_learn_state *) CALLOC(1, sizeof(ch_learn_state));
                    float *fen_maybe_board = ch_fen_to_board(valid_maybe_fen, 1);
                    memcpy(to_maybe_learn->board, fen_maybe_board, (1)*(BOARD_SIZE)*sizeof(float));
                    to_maybe_learn->index = i;
                    to_maybe_learn->player = player == 0 ? 1 : 0;
                    ch_put_back(sessionId, to_maybe_learn);
                    ch_train_possible_checkmate(sessionId, move_state, sfen, net, level, --deep);
                    ch_pick_back(sessionId);
                    FREE(fen_maybe_board);
                }
            }
            for (int i = 0; i < valid_maybe_moves_count; ++i) FREE(valid_maybe_moves[i]);
            FREE(valid_maybe_moves);
            FREE(valid_maybe_fen);
        }
    }
}

void test_echess(int argc, char** argv, char *cfgfile, char *weight_file) {
    char valid_fen[128];
    char valid_fen_move[8];
    char sfen[128];
    char valid_fen_next[128];
    char valid_fen_last[128];

    ch_board_state move_state = {0};

    int print = 0;

    char* startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    int level = 4;

    network *net;
    char *ch_weight_file = weight_file;

    if (!exists(weight_file)) {
        net = parse_network_cfg(cfgfile);
        net->nsteps = 0;
        save_weights(net, weight_file);
    }

    char* sessionId = "13a25e80-ece3-4a4b-9347-e6df74386d02";
    ch_init_game_history(sessionId);

    if (exists(weight_file)) {
        net = load_network(cfgfile, weight_file, 0);
    } else {
        net = parse_network_cfg(cfgfile);
        net->nsteps = 0;
        save_weights(net, weight_file);
    }
    set_batch_network(net, 1);

    FILE* log = NULL;
    if (print) log = fopen("log.txt", "a+");

    fprintf(stdout, "iChess.io by Piotr Sowa v7.27\n");

    strcpy(valid_fen, startpos);
    strcpy(valid_fen_move, "");
    //ch_get_fen_960(valid_fen);
    strcpy(sfen, valid_fen);

    int player = 0;
    ch_clean_history(sessionId, 1);

    char *buff = NULL;
    size_t len = 0;
    size_t nread = 0;
    while ((nread = getline(&buff, &len, stdin)) > 0) {
        if (buff == NULL) continue;
        buff[strcspn(buff, "\r\n")] = 0;

        if (print) {
            fprintf(log, "%s\n", buff);
            fflush(log);
        }
        if (strncmp(buff, "ucinewgame", 10) == 0) {
            strcpy(move_state.fen, valid_fen);
            strcpy(move_state.move, valid_fen_move);
            strcpy(valid_fen, sfen[0] != '\0' ? sfen : startpos);
            strcpy(valid_fen_move, "");
            strcpy(sfen, "");
            player = 0;
            ch_clean_history(sessionId, 1);
			save_weights(net, ch_weight_file);
            fprintf(stdout, "%s\n", "uciok");
            fflush(stdout);

        }
        else if (strncmp(buff, "uci", 3) == 0) {
            strcpy(valid_fen, startpos);
            strcpy(valid_fen_move, "");
            fprintf(stdout, "%s\n", "id name iChess.io 7.27");
            fprintf(stdout, "%s\n", "id author Piotr Sowa");
            fprintf(stdout, "%s\n", "option name UCI_Chess960 type check default false");
            fprintf(stdout, "%s\n", "option name BackendOptions type string default");
            fprintf(stdout, "%s\n", "option name Ponder type check default false");
            fprintf(stdout, "%s\n", "option name MultiPV type spin default 1 min 1 max 500");
            fprintf(stdout, "%s\n", "uciok");
            fflush(stdout);

        }
        else if (strncmp(buff, "isready", 7) == 0) {
            fprintf(stdout,"readyok\n");
            fflush(stdout);

        }
        else if (strncmp(buff, "stop", 4) == 0) {
            // ;-)
        }
        else if (strncmp(buff, "quit", 4) == 0) {
            break;
        }
        else if (strncmp(buff, "position ", 9) == 0){
            char** moves = NULL;
            char* move = NULL;
            char* sfenn = NULL;
            char* mfenn = NULL;

            int count = 0;
            char* fen = ch_analyze_pos(sfen, buff, &sfenn, &mfenn, &moves, &move, &count);

            strcpy(sfen, sfenn != NULL ? sfenn : "");
            strcpy(valid_fen, fen != NULL ? fen : "");
            strcpy(valid_fen_move, move != NULL ? move : "");
            strcpy(valid_fen_next, sfenn != NULL ? sfenn : "");
            strcpy(valid_fen_last, mfenn != NULL ? mfenn : "");

            char *mfen_next = NULL;
            int mfen_next_idx = 0;
            int mfen_next_cnt = 0;

            int qcount = ch_queue_count(ch_dict_get(moves_history,sessionId));
            int exists = qcount > 1;

            player = qcount % 2 == 0 ? 1 : 0;

            int addon = (exists ? count : qcount);

            if (exists) {

                if (ch_board_after_move(sfen, valid_fen_last, moves[count - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                    strcpy(move_state.fen, valid_fen_last);
                    strcpy(move_state.move, moves[count - 1]);
                    move_state.indext = mfen_next_idx;

                    //ch_put_into_game_queue(sessionId, valid_fen_last, moves[count - 1], mfen_next_idx, net, sfen);

                    float *board0 = ch_fen_to_board(valid_fen_last, 1);
                    float *board_next0 = ch_fen_to_board(mfen_next, 1);
                    float pow0 = ch_eval_the_move(sfen, valid_fen_last, mfen_next);
                    float powW0 = 0;
                    float powB0 = 0;
                    ch_eval_the_board(sfen, board0, &powW0, &powB0);
                    float value0 = player == 0 ? powW0 : powB0;

                    fprintf(stderr,
                            "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                            net->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow0, mfen_next_idx);

                    ch_self_study_train_self_step(sessionId, sfen, valid_fen_last, moves[count - 1], net, level, mfen_next_idx, pow0, value0);

                    FREE(board0);
                    FREE(board_next0);

                }
				FREE(mfen_next);
            }
            else if (count > 0) {

                if (ch_board_after_move(sfen, valid_fen_next, moves[addon - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                    strcpy(move_state.fen, valid_fen_next);
                    strcpy(move_state.move, moves[addon - 1]);
                    move_state.indext = mfen_next_idx;

                    ch_put_into_game_queue(sessionId, valid_fen_next, moves[addon - 1], mfen_next_idx, net, sfen);

                    if (++net->nsteps % 10000 == 0) save_weights(net, ch_weight_file);

                    float *board1 = ch_fen_to_board(valid_fen_next, 1);
                    float *board_next1 = ch_fen_to_board(mfen_next, 1);
                    float pow1 = ch_eval_the_move(sfen, valid_fen_next, mfen_next);
                    float powW1 = 0;
                    float powB1 = 0;
                    ch_eval_the_board(sfen, board1, &powW1, &powB1);
                    float value1 = player == 0 ? powW1 : powB1;

                    fprintf(stderr,
                            "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                            net->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow1, mfen_next_idx);

                    ch_self_study_train_self_step(sessionId, sfen, valid_fen_next, moves[addon - 1], net, level, mfen_next_idx, pow1, value1);

                    strcpy(valid_fen_next, mfen_next);
                    strcpy(valid_fen_last, mfen_next);

                    strcpy(valid_fen, valid_fen_next);
                    strcpy(valid_fen_move, moves[addon - 1]);

                    addon = (exists ? count : ch_queue_count(ch_dict_get(moves_history, sessionId)));

                    for (int i = addon - 1; i < count; ++i) {

                        if (ch_board_after_move(sfen, valid_fen_next, moves[addon - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                            strcpy(move_state.fen, valid_fen_next);
                            strcpy(move_state.move, moves[addon - 1]);
                            move_state.indext = mfen_next_idx;

                            ch_put_into_game_queue(sessionId, valid_fen_next, moves[addon - 1], mfen_next_idx, net , sfen);
                            if (++net->nsteps % 10000 == 0) save_weights(net, ch_weight_file);

                            float *board2 = ch_fen_to_board(valid_fen_next, 1);
                            float *board_next2 = ch_fen_to_board(mfen_next, 1);
                            float pow2 = ch_eval_the_move(sfen, valid_fen_next, mfen_next);
                            float powW2 = 0;
                            float powB2 = 0;
                            ch_eval_the_board(sfen, board2, &powW2, &powB2);
                            float value2 = player == 0 ? powW2 : powB2;

                            fprintf(stderr,
                                    "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                                    net->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow2, mfen_next_idx);

                            ch_self_study_train_self_step(sessionId, sfen, valid_fen_next, moves[addon - 1], net, level,mfen_next_idx, pow2, value2);

                            strcpy(valid_fen_next, mfen_next);
                            strcpy(valid_fen_last, mfen_next);

                            strcpy(valid_fen, valid_fen_next);
                            strcpy(valid_fen_move, moves[addon - 1]);

                            addon = (exists ? count : ch_queue_count(ch_dict_get(moves_history, sessionId)));

                            player = player == 0 ? 1 : 0;

                            FREE(board2);
                            FREE(board_next2);
                        }
                    }

                    FREE(board1);
                    FREE(board_next1);
                }
				FREE(mfen_next);
            }

            ch_train_possible_checkmate(sessionId, move_state, sfen, net, level, 2);

            if (sfenn != NULL) FREE(sfenn);
            if (mfenn != NULL) FREE(mfenn);
            if (moves != NULL) FREE(moves);
            if (move != NULL) FREE(move);
            if (fen != NULL) FREE(fen);
            if (print) ch_print_board(valid_fen);
        }
        else if (strncmp(buff, "go", 2) == 0 || strncmp(buff, "go infinite", 11) == 0) {

            int qcount = ch_queue_count(ch_dict_get(moves_history,sessionId));

            player = qcount % 2 == 0 ? 0 : 1;

            move_state = ch_self_learn_step(sessionId, sfen, level, net, valid_fen, valid_fen_move, 1);

            ch_train_possible_checkmate(sessionId, move_state, sfen, net, level, 2);

            if (++net->nsteps % 10000 == 0) save_weights(net, ch_weight_file);

            strcpy(valid_fen, move_state.fen);
            strcpy(valid_fen_move, move_state.move);

            fprintf(stdout, "info depth %i pv %s\n", 1, valid_fen_move);

            if (strcmp(valid_fen_move, "") == 0) {
                fprintf(stdout, "bestmove\n");
            } else {
                fprintf(stdout, "bestmove %s ponder %s\n", valid_fen_move, valid_fen_move);
            }

            fflush(stdout);
            if (print) ch_print_board(valid_fen);
        }
        else if (strncmp(buff, "register", 8) == 0) {
            fprintf(stdout,"registration ok\n");
            fflush(stdout);
        }

        FREE(buff);
        buff = NULL;
    }

    save_weights(net, ch_weight_file);

    free_network(net);

    ch_clean_history(sessionId, 1);

    if (print) fclose(log);
}

void test_mchess(int argc, char** argv, char *cfgfile, char *weight_file) {
    char valid_fen[128];
    char valid_fen_move[8];
    char sfen[128];
    char valid_fen_next[128];
    char valid_fen_last[128];

    ch_board_state move_state = {0};

    int print = 0;

    char* startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

    int level = 4;

    network *net;
    char *ch_weight_file = weight_file;

    if (!exists(weight_file)) {
        net = parse_network_cfg(cfgfile);
        net->nsteps = 0;
        save_weights(net, weight_file);
    }

    char* sessionId = "13a25e80-ece3-4a4b-9347-e6df74386d02";
    ch_init_game_history(sessionId);

    if (exists(weight_file)) {
        net = load_network(cfgfile, weight_file, 0);
    } else {
        net = parse_network_cfg(cfgfile);
        net->nsteps = 0;
        save_weights(net, weight_file);
    }
    set_batch_network(net, 1);

    FILE* log = NULL;
    if (print) log = fopen("log.txt", "a+");

    fprintf(stdout, "iChess.io by Piotr Sowa v7.27\n");

    strcpy(valid_fen, startpos);
    strcpy(valid_fen_move, "");
    strcpy(sfen, valid_fen);

    int player = 0;
    ch_clean_history(sessionId, 1);

    FILE *sf = popen("/usr/local/Cellar/stockfish/17.1/bin/stockfish", "r+");
    if (!sf) {
        fprintf(stderr, "Failed to start Stockfish\n");
        exit(1);
    }

    fprintf(sf, "uci\n"); fflush(sf);

    char *buff = NULL;
    size_t len = 0;
    size_t nread = 0;
    while ((nread = getline(&buff, &len, stdin)) > 0) {
        if (buff == NULL) continue;
        buff[strcspn(buff, "\r\n")] = 0;

        if (print) {
            fprintf(log, "%s\n", buff);
            fflush(log);
        }
        if (strncmp(buff, "ucinewgame", 10) == 0) {
            strcpy(move_state.fen, valid_fen);
            strcpy(move_state.move, valid_fen_move);
            strcpy(valid_fen, sfen[0] != '\0' ? sfen : startpos);
            strcpy(valid_fen_move, "");
            strcpy(sfen, "");
            player = 0;
            ch_clean_history(sessionId, 1);
            save_weights(net, ch_weight_file);
            fprintf(stdout, "%s\n", "uciok");
            fflush(stdout);
            fprintf(sf, "%s\n", buff); fflush(sf);
        }
        else if (strncmp(buff, "uci", 3) == 0) {
            strcpy(valid_fen, startpos);
            strcpy(valid_fen_move, "");
            fprintf(stdout, "%s\n", "id name iChess.io 7.27");
            fprintf(stdout, "%s\n", "id author Piotr Sowa");
            fprintf(stdout, "%s\n", "option name UCI_Chess960 type check default false");
            fprintf(stdout, "%s\n", "option name BackendOptions type string default");
            fprintf(stdout, "%s\n", "option name Ponder type check default false");
            fprintf(stdout, "%s\n", "option name MultiPV type spin default 1 min 1 max 500");
            fprintf(stdout, "%s\n", "uciok");
            fflush(stdout);
            fprintf(sf, "%s\n", buff); fflush(sf);
        }
        else if (strncmp(buff, "isready", 7) == 0) {
            fprintf(stdout,"readyok\n");
            fflush(stdout);
            fprintf(sf, "%s\n", buff); fflush(sf);
        }
        else if (strncmp(buff, "stop", 4) == 0) {
            fprintf(sf, "%s\n", buff); fflush(sf);
        }
        else if (strncmp(buff, "quit", 4) == 0) {
            fprintf(sf, "%s\n", buff); fflush(sf);
            break;
        }
        else if (strncmp(buff, "position ", 9) == 0){
            fprintf(sf, "%s\n", buff); fflush(sf);

            char** moves = NULL;
            char* move = NULL;
            char* sfenn = NULL;
            char* mfenn = NULL;

            int count = 0;
            char* fen = ch_analyze_pos(sfen, buff, &sfenn, &mfenn, &moves, &move, &count);

            strcpy(sfen, sfenn != NULL ? sfenn : "");
            strcpy(valid_fen, fen != NULL ? fen : "");
            strcpy(valid_fen_move, move != NULL ? move : "");
            strcpy(valid_fen_next, sfenn != NULL ? sfenn : "");
            strcpy(valid_fen_last, mfenn != NULL ? mfenn : "");

            char *mfen_next = NULL;
            int mfen_next_idx = 0;
            int mfen_next_cnt = 0;

            int qcount = ch_queue_count(ch_dict_get(moves_history,sessionId));
            int exists = qcount > 1;

            player = qcount % 2 == 0 ? 1 : 0;

            int addon = (exists ? count : qcount);

            if (exists) {

                if (ch_board_after_move(sfen, valid_fen_last, moves[count - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                    strcpy(move_state.fen, valid_fen_last);
                    strcpy(move_state.move, moves[count - 1]);
                    move_state.indext = mfen_next_idx;

                    //ch_put_into_game_queue(sessionId, valid_fen_last, moves[count - 1], mfen_next_idx, net, sfen);

                    float *board0 = ch_fen_to_board(valid_fen_last, 1);
                    float *board_next0 = ch_fen_to_board(mfen_next, 1);
                    float pow0 = ch_eval_the_move(sfen, valid_fen_last, mfen_next);
                    float powW0 = 0;
                    float powB0 = 0;
                    ch_eval_the_board(sfen, board0, &powW0, &powB0);
                    float value0 = player == 0 ? powW0 : powB0;

                    fprintf(stderr,
                            "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                            net->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow0, mfen_next_idx);

                    ch_self_study_train_self_step(sessionId, sfen, valid_fen_last, moves[count - 1], net, level, mfen_next_idx, pow0, value0);

                    FREE(board0);
                    FREE(board_next0);
                }
                FREE(mfen_next);
            }
            else if (count > 0) {

                if (ch_board_after_move(sfen, valid_fen_next, moves[addon - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                    strcpy(move_state.fen, valid_fen_next);
                    strcpy(move_state.move, moves[addon - 1]);
                    move_state.indext = mfen_next_idx;

                    ch_put_into_game_queue(sessionId, valid_fen_next, moves[addon - 1], mfen_next_idx, net, sfen);
                    if (++net->nsteps % 10000 == 0) save_weights(net, ch_weight_file);

                    float *board1 = ch_fen_to_board(valid_fen_next, 1);
                    float *board_next1 = ch_fen_to_board(mfen_next, 1);
                    float pow1 = ch_eval_the_move(sfen, valid_fen_next, mfen_next);
                    float powW1 = 0;
                    float powB1 = 0;
                    ch_eval_the_board(sfen, board1, &powW1, &powB1);
                    float value1 = player == 0 ? powW1 : powB1;

                    fprintf(stderr,
                            "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                            net->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow1, mfen_next_idx);

                    ch_self_study_train_self_step(sessionId, sfen, valid_fen_next, moves[addon - 1], net, level, mfen_next_idx, pow1, value1);

                    strcpy(valid_fen_next, mfen_next);
                    strcpy(valid_fen_last, mfen_next);

                    strcpy(valid_fen, valid_fen_next);
                    strcpy(valid_fen_move, moves[addon - 1]);

                    addon = (exists ? count : ch_queue_count(ch_dict_get(moves_history, sessionId)));

                    for (int i = addon - 1; i < count; ++i) {

                        if (ch_board_after_move(sfen, valid_fen_next, moves[addon - 1], &mfen_next, &mfen_next_idx, &mfen_next_cnt)) {

                            strcpy(move_state.fen, valid_fen_next);
                            strcpy(move_state.move, moves[addon - 1]);
                            move_state.indext = mfen_next_idx;

                            ch_put_into_game_queue(sessionId, valid_fen_next, moves[addon - 1], mfen_next_idx, net , sfen);
                            if (++net->nsteps % 10000 == 0) save_weights(net, ch_weight_file);

                            float *board2 = ch_fen_to_board(valid_fen_next, 1);
                            float *board_next2 = ch_fen_to_board(mfen_next, 1);
                            float pow2 = ch_eval_the_move(sfen, valid_fen_next, mfen_next);
                            float powW2 = 0;
                            float powB2 = 0;
                            ch_eval_the_board(sfen, board2, &powW2, &powB2);
                            float value2 = player == 0 ? powW2 : powB2;

                            fprintf(stderr,
                                    "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                                    net->nsteps, player == 0 ? "w" : "b", mfen_next_cnt, 1, pow2, mfen_next_idx);

                            ch_self_study_train_self_step(sessionId, sfen, valid_fen_next, moves[addon - 1], net, level,mfen_next_idx, pow2, value2);

                            strcpy(valid_fen_next, mfen_next);
                            strcpy(valid_fen_last, mfen_next);

                            strcpy(valid_fen, valid_fen_next);
                            strcpy(valid_fen_move, moves[addon - 1]);

                            addon = (exists ? count : ch_queue_count(ch_dict_get(moves_history, sessionId)));

                            player = player == 0 ? 1 : 0;

                            FREE(board2);
                            FREE(board_next2);
                        }
                    }

                    FREE(board1);
                    FREE(board_next1);
                }
                FREE(mfen_next);
            }

            ch_train_possible_checkmate(sessionId, move_state, sfen, net, level, 2);

            if (sfenn != NULL) FREE(sfenn);
            if (mfenn != NULL) FREE(mfenn);
            if (moves != NULL) FREE(moves);
            if (move != NULL) FREE(move);
            if (fen != NULL) FREE(fen);
            if (print) ch_print_board(valid_fen);
        }
        else if (strncmp(buff, "go", 2) == 0 || strncmp(buff, "go infinite", 11) == 0) {
            fprintf(sf, "%s\n", buff); fflush(sf);

            char sf_line[256];
            char sf_move[16] = "";
            while (fgets(sf_line, sizeof(sf_line), sf)) {
                if (strncmp(sf_line, "bestmove", 8) == 0) {
                    sscanf(sf_line, "bestmove %s", sf_move);
                    break;
                }
            }

            if (strcmp(sf_move, "") != 0) {
                strcpy(valid_fen_move, sf_move);
            }

            int qcount = ch_queue_count(ch_dict_get(moves_history,sessionId));

            player = qcount % 2 == 0 ? 0 : 1;

            strcpy(move_state.fen, valid_fen);
            strcpy(move_state.move, valid_fen_move);

            float *board0 = ch_fen_to_board(valid_fen_last, 1);
            float *board_next0 = ch_fen_to_board(valid_fen, 1);
            float pow0 = ch_eval_the_move(sfen, valid_fen_last, valid_fen);
            float powW0 = 0;
            float powB0 = 0;
            ch_eval_the_board(sfen, board0, &powW0, &powB0);
            float value0 = player == 0 ? powW0 : powB0;

            int indext = ch_moves_index(sfen, valid_fen, valid_fen_move);

            fprintf(stderr,
                    "pick: step %ld(%s): count: (%i) checked: (%i) power: (%.7f) index: (%i)\n",
                    net->nsteps, player == 0 ? "w" : "b", 0, 1, pow0, indext);

            ch_self_study_train_self_step(sessionId, sfen, valid_fen_last, valid_fen_move, net, level, indext, pow0, value0);

            if (++net->nsteps % 10000 == 0) save_weights(net, ch_weight_file);

            strcpy(valid_fen, move_state.fen);
            strcpy(valid_fen_move, move_state.move);

            fprintf(stdout, "info depth %i pv %s\n", 1, valid_fen_move);

            if (strcmp(valid_fen_move, "") == 0) {
                fprintf(stdout, "bestmove\n");
            } else {
                fprintf(stdout, "bestmove %s ponder %s\n", valid_fen_move, valid_fen_move);
            }

            fflush(stdout);
            if (print) ch_print_board(valid_fen);

            FREE(board0);
            FREE(board_next0);
        }
        else if (strncmp(buff, "register", 8) == 0) {
            fprintf(sf, "%s\n", buff); fflush(sf);

            fprintf(stdout,"registration ok\n");
            fflush(stdout);
        }

        FREE(buff);
        buff = NULL;
    }

    save_weights(net, ch_weight_file);

    free_network(net);

    ch_clean_history(sessionId, 1);

    pclose(sf);

    if (print) fclose(log);
}
