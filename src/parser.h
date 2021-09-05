#ifndef PARSER_H
#define PARSER_H
#include "darknet.h"
#include "network.h"

void save_network(network net, char *filename);
void save_weights_double(network net, char *filename);

network parse_network_cfg_custom(char *filename, int batch, int time_steps);

#endif
