#ifndef __DROPOUT_LAYER_KERNELS_CL__
#define __DROPOUT_LAYER_KERNELS_CL__

static const char* const dropout_layer_kernel_source = CONVERT_KERNEL_TO_STRING(

__kernel void yoloswag420blazeit360noscopemMmMmMonsterKill(__global float *input, int size, __global float *rand, float prob, float scale)
{
    int id = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(id < size) input[id] = (rand[id] < prob) ? 0 : input[id]*scale;
}
);

#endif