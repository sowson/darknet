#include "layer.h"
#include "opencl.h"

#include <stdlib.h>

void free_layer(layer l)
{
    if(l.type == DROPOUT){
        if (gpu_index >= 0) {
            if(l.rand)           free(l.rand);
        }
#ifdef GPU
        if (gpu_index >= 0) {
            if (l.rand_gpu.ptr) opencl_free(l.rand_gpu);
        }
#endif
        return;
    }

    if (gpu_index == -1) {
        if(l.mask)               free(l.mask);
        if(l.cweights)           free(l.cweights);
        if(l.indexes)            free(l.indexes);
        if(l.input_layers)       free(l.input_layers);
        if(l.input_sizes)        free(l.input_sizes);
        if(l.map)                free(l.map);
        if(l.rand)               free(l.rand);
        if(l.cost)               free(l.cost);
        if(l.state)              free(l.state);
        if(l.prev_state)         free(l.prev_state);
        if(l.forgot_state)       free(l.forgot_state);
        if(l.forgot_delta)       free(l.forgot_delta);
        if(l.state_delta)        free(l.state_delta);
        if(l.concat)             free(l.concat);
        if(l.concat_delta)       free(l.concat_delta);
        if(l.binary_weights)     free(l.binary_weights);
        if(l.biases)             free(l.biases);
        if(l.bias_updates)       free(l.bias_updates);
        if(l.scales)             free(l.scales);
        if(l.scale_updates)      free(l.scale_updates);
        if(l.weights)            free(l.weights);
        if(l.weight_updates)     free(l.weight_updates);
        if(l.delta)              free(l.delta);
        if(l.output)             free(l.output);
        if(l.squared)            free(l.squared);
        if(l.norms)              free(l.norms);
        if(l.spatial_mean)       free(l.spatial_mean);
        if(l.mean)               free(l.mean);
        if(l.variance)           free(l.variance);
        if(l.mean_delta)         free(l.mean_delta);
        if(l.variance_delta)     free(l.variance_delta);
        if(l.rolling_mean)       free(l.rolling_mean);
        if(l.rolling_variance)   free(l.rolling_variance);
        if(l.x)                  free(l.x);
        if(l.x_norm)             free(l.x_norm);
        if(l.m)                  free(l.m);
        if(l.v)                  free(l.v);
        if(l.z_cpu)              free(l.z_cpu);
        if(l.r_cpu)              free(l.r_cpu);
        if(l.h_cpu)              free(l.h_cpu);
        if(l.binary_input)       free(l.binary_input);
    }

#ifdef GPU
    if (gpu_index >= 0) {
        if (l.indexes_gpu.ptr) opencl_free(l.indexes_gpu);
        if (l.z_gpu.ptr) opencl_free(l.z_gpu);
        if (l.r_gpu.ptr) opencl_free(l.r_gpu);
        if (l.h_gpu.ptr) opencl_free(l.h_gpu);
        if (l.m_gpu.ptr) opencl_free(l.m_gpu);
        if (l.v_gpu.ptr) opencl_free(l.v_gpu);
        if (l.prev_state_gpu.ptr) opencl_free(l.prev_state_gpu);
        if (l.forgot_state_gpu.ptr) opencl_free(l.forgot_state_gpu);
        if (l.forgot_delta_gpu.ptr) opencl_free(l.forgot_delta_gpu);
        if (l.state_gpu.ptr) opencl_free(l.state_gpu);
        if (l.state_delta_gpu.ptr) opencl_free(l.state_delta_gpu);
        if (l.concat_gpu.ptr) opencl_free(l.concat_gpu);
        if (l.concat_delta_gpu.ptr) opencl_free(l.concat_delta_gpu);
        if (l.binary_input_gpu.ptr) opencl_free(l.binary_input_gpu);
        if (l.binary_weights_gpu.ptr) opencl_free(l.binary_weights_gpu);
        if (l.batch_normalize) {
            if (l.mean_gpu.ptr) opencl_free(l.mean_gpu);
            if (l.variance_gpu.ptr) opencl_free(l.variance_gpu);
            if (l.rolling_mean_gpu.ptr) opencl_free(l.rolling_mean_gpu);
            if (l.rolling_variance_gpu.ptr) opencl_free(l.rolling_variance_gpu);
            if (l.variance_delta_gpu.ptr) opencl_free(l.variance_delta_gpu);
            if (l.mean_delta_gpu.ptr) opencl_free(l.mean_delta_gpu);
            if (l.x_gpu.ptr) opencl_free(l.x_gpu);
            if (l.x_norm_gpu.ptr) opencl_free(l.x_norm_gpu);
        }
        if (l.weights_gpu.ptr) opencl_free(l.weights_gpu);
        if (l.weight_updates_gpu.ptr) opencl_free(l.weight_updates_gpu);
        if (l.biases_gpu.ptr) opencl_free(l.biases_gpu);
        if (l.bias_updates_gpu.ptr) opencl_free(l.bias_updates_gpu);
        if (l.scales_gpu.ptr) opencl_free(l.scales_gpu);
        if (l.scale_updates_gpu.ptr) opencl_free(l.scale_updates_gpu);
        if (l.output_gpu.ptr) opencl_free(l.output_gpu);
        if (l.delta_gpu.ptr) opencl_free(l.delta_gpu);
        if (l.rand_gpu.ptr) opencl_free(l.rand_gpu);
        if (l.squared_gpu.ptr) opencl_free(l.squared_gpu);
        if (l.norms_gpu.ptr) opencl_free(l.norms_gpu);
    }
#endif
}
