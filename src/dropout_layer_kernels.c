#ifdef GPU

#include "darknet.h"

#include "dropout_layer.h"
#include "opencl.h"
#include "utils.h"

#include "dropout_layer_kernels.cl"

cl_program* opencl_dropout_layer_program;
cl_kernel* opencl_yoloswag420blazeit360noscopemMmMmMonsterKill;

void dropout_kernel_init(void)
{
    if (opencl_device_id_t == 0) {
        opencl_dropout_layer_program = calloc(opencl_device_ct_t, sizeof(cl_program));
        opencl_yoloswag420blazeit360noscopemMmMmMonsterKill = calloc(opencl_device_ct_t, sizeof(cl_kernel));
    }

    opencl_load_buffer(dropout_layer_kernel_source, strlen(dropout_layer_kernel_source), &opencl_dropout_layer_program[opencl_device_id_t]);

    opencl_create_kernel(&opencl_dropout_layer_program[opencl_device_id_t], "yoloswag420blazeit360noscopemMmMmMonsterKill", &opencl_yoloswag420blazeit360noscopemMmMmMonsterKill[opencl_device_id_t]);
}

void dropout_kernel_release(void)
{
    clReleaseKernel(opencl_yoloswag420blazeit360noscopemMmMmMonsterKill[opencl_device_id_t]);
    opencl_yoloswag420blazeit360noscopemMmMmMonsterKill[opencl_device_id_t] = 0;

    clReleaseProgram(opencl_dropout_layer_program[opencl_device_id_t]);

    if (opencl_device_id_t == opencl_device_ct_t-1) {
        free(opencl_dropout_layer_program);
        free(opencl_yoloswag420blazeit360noscopemMmMmMonsterKill);
    }
}

void forward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if (!net.train) return;
    int size = layer.inputs*layer.batch;
    layer.rand_gpu = opencl_random(layer.rand_gpu, size);

    dim2 dimGrid;
    dimGrid = opencl_gridsize(size);

    opencl_kernel(opencl_yoloswag420blazeit360noscopemMmMmMonsterKill[opencl_device_id_t], dimGrid, 10, &net.input_gpu.mem, sizeof(cl_mem), &size, sizeof(cl_int), &layer.rand_gpu.mem, sizeof(cl_mem), &layer.probability, sizeof(cl_float), &layer.scale, sizeof(cl_float));
}

void backward_dropout_layer_gpu(dropout_layer layer, network net)
{
    if(!net.delta) return;
    int size = layer.inputs*layer.batch;

    dim2 dimGrid;
    dimGrid = opencl_gridsize(size);

    opencl_kernel(opencl_yoloswag420blazeit360noscopemMmMmMonsterKill[opencl_device_id_t], dimGrid, 10, &net.delta_gpu.mem, sizeof(cl_mem), &size, sizeof(cl_int), &layer.rand_gpu.mem, sizeof(cl_mem), &layer.probability, sizeof(cl_float), &layer.scale, sizeof(cl_float));
}

#endif