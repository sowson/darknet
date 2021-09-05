// Oh boy, why am I about to do this....
#ifndef NETWORK_H
#define NETWORK_H
#include "darknet.h"

#include "image.h"
#include "layer.h"
#include "data.h"
#include "tree.h"


#ifdef GPU
void pull_network_output(network *net);
#endif

void compare_networks(network *n1, network *n2, data d);
char *get_layer_string(LAYER_TYPE a);

network *make_network(int n);
void clean_network_moves_gpu(network *net);
void clean_layer_moves_gpu(layer *l);

float *get_network_output_y4(network net);

int get_network_output_size(network net);
int get_network_input_size(network net);

float network_accuracy_multi(network *net, data d, int n);
int get_predicted_class_network(network *net);
void print_network(network *net);
int resize_network(network *net, int w, int h);
void calc_network_cost(network *net);

#endif

