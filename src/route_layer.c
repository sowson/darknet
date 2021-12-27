#include "route_layer.h"
#include "utils.h"
#include "opencl.h"
#include "blas.h"
#include <stdio.h>

route_layer make_route_layer(int batch, int n, int *input_layers, int *input_sizes, int groups, int group_id)
{
    fprintf(stderr,"route ");
    route_layer l = {0};
    l.type = ROUTE;
    l.batch = batch;
    l.n = n;
    l.input_layers = input_layers;
    l.input_sizes = input_sizes;
    l.groups = groups;
    l.group_id = group_id;
    int i;
    int outputs = 0;
    for(i = 0; i < n; ++i){
        fprintf(stderr," %d", input_layers[i]);
        outputs += input_sizes[i];
    }
    fprintf(stderr,"\n");
    outputs = outputs / groups;
    l.outputs = outputs;
    l.inputs = outputs;
    //fprintf(stderr, " inputs = %d \t outputs = %d, groups = %d, group_id = %d \n", l.inputs, l.outputs, l.groups, l.group_id);
    l.delta = (float*)calloc(outputs * batch, sizeof(float));
    l.output = (float*)calloc(outputs * batch, sizeof(float));

    l.forward = forward_route_layer;
    l.backward = backward_route_layer;
    #ifdef GPU
    if(gpu_index >= 0) {
        l.forward_gpu = forward_route_layer_gpu;
        l.backward_gpu = backward_route_layer_gpu;
        l.update_gpu = 0;

        l.delta_gpu = opencl_make_array(l.delta, outputs * batch);
        l.output_gpu = opencl_make_array(l.output, outputs * batch);
    }
    #endif
    return l;
}

void resize_route_layer(route_layer *l, network *net)
{
    int i;
    layer first = net->layers[l->input_layers[0]];
    l->out_w = first.out_w;
    l->out_h = first.out_h;
    l->out_c = first.out_c;
    l->outputs = first.outputs;
    l->input_sizes[0] = first.outputs;
    for(i = 1; i < l->n; ++i){
        int index = l->input_layers[i];
        layer next = net->layers[index];
        l->outputs += next.outputs;
        l->input_sizes[i] = next.outputs;
        if(next.out_w == first.out_w && next.out_h == first.out_h){
            l->out_c += next.out_c;
        }else{
            printf("error: different size of input layers: %d x %d, %d x %d\n", next.out_w, next.out_h, first.out_w, first.out_h);
            l->out_h = l->out_w = l->out_c = 0;
            exit(EXIT_FAILURE);
        }
    }
    l->out_c = l->out_c / l->groups;
    l->outputs = l->outputs / l->groups;
    l->inputs = l->outputs;

#ifdef GPU
    if (gpu_index >= 0) {
        opencl_free_gpu_only(l->output_gpu);
        opencl_free_gpu_only(l->delta_gpu);
    }
#endif
    l->output = (float*)realloc(l->output, l->outputs * l->batch * sizeof(float));
    l->delta = (float*)realloc(l->delta, l->outputs * l->batch * sizeof(float));
#ifdef GPU
    if (gpu_index >= 0) {
        l->output_gpu = opencl_make_array(l->output, l->outputs * l->batch);
        l->delta_gpu = opencl_make_array(l->delta, l->outputs * l->batch);
    }
#endif
}

void forward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *input = net.layers[index].output;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            copy_cpu(part_input_size, input + j*input_size + part_input_size*l.group_id, 1, l.output + offset + j*l.outputs, 1);
        }
        offset += part_input_size;
    }
}

void backward_route_layer(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        float *delta = net.layers[index].delta;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            axpy_cpu(part_input_size, 1, l.delta + offset + j*l.outputs, 1, delta + j*input_size + part_input_size*l.group_id, 1);
        }
        offset += part_input_size;
    }
}

#ifdef GPU
void forward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        cl_mem_ext input = net.layers[index].output_gpu;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            copy_offset_gpu(part_input_size, input, j*input_size + part_input_size*l.group_id, 1, l.output_gpu, offset + j*l.outputs, 1);
        }
        offset += part_input_size;
    }
}

void backward_route_layer_gpu(const route_layer l, network net)
{
    int i, j;
    int offset = 0;
    for(i = 0; i < l.n; ++i){
        int index = l.input_layers[i];
        cl_mem_ext delta = net.layers[index].delta_gpu;
        int input_size = l.input_sizes[i];
        int part_input_size = input_size / l.groups;
        for(j = 0; j < l.batch; ++j){
            axpy_offset_gpu(part_input_size, 1, l.delta_gpu, offset + j*l.outputs, 1, delta, j*input_size + part_input_size*l.group_id, 1);
        }
        offset += part_input_size;
    }
}
#endif