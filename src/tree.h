#ifndef TREE_H
#define TREE_H
#include "darknet.h"

int hierarchy_top_prediction(float *predictions, tree *hier, float thresh, int stride);
void hierarchy_predictions(float *predictions, int n, tree *hier, int only_leaves, int stride);
void change_leaves(tree *t, char *leaf_list);
float get_hierarchy_probability(float *x, tree *hier, int c, int stride);

int hierarchy_top_prediction_y4(float *predictions, tree *hier, float thresh, int stride);
void hierarchy_predictions_y4(float *predictions, int n, tree *hier, int only_leaves);
void change_leaves_y4(tree *t, char *leaf_list);
float get_hierarchy_probability_y4(float *x, tree *hier, int c);

#endif
