#ifdef GPU

#define DARKNET_VERBOSE_GPU
//#define DEBUG_KERNELS

#include "opencl.h"
#include <unistd.h>
#include <assert.h>

#ifndef GPU_INDEX
#define GPU_INDEX
int gpu_index = 1;
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <math.h>

#include "blas.h"

int *gpusg;
int ngpusg;
__thread int opencl_device_id_t;
__thread int opencl_device_ct_t;

cl_int *cl_native_double_width_s;
size_t *cl_native_max_group_size_s;
size_t *cl_native_address_bits_s;

cl_context opencl_context;
cl_command_queue* opencl_queues;
cl_device_id* opencl_devices;

cl_context_properties* cl_props;

const char* clGetErrorString(int errorCode) {
    switch (errorCode) {
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";
        case -69: return "CL_INVALID_PIPE_SIZE";
        case -70: return "CL_INVALID_DEVICE_QUEUE";
        case -71: return "CL_INVALID_SPEC_ID";
        case -72: return "CL_MAX_SIZE_RESTRICTION_EXCEEDED";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        case -1006: return "CL_INVALID_D3D11_DEVICE_KHR";
        case -1007: return "CL_INVALID_D3D11_RESOURCE_KHR";
        case -1008: return "CL_D3D11_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1009: return "CL_D3D11_RESOURCE_NOT_ACQUIRED_KHR";
        case -1010: return "CL_INVALID_DX9_MEDIA_ADAPTER_KHR";
        case -1011: return "CL_INVALID_DX9_MEDIA_SURFACE_KHR";
        case -1012: return "CL_DX9_MEDIA_SURFACE_ALREADY_ACQUIRED_KHR";
        case -1013: return "CL_DX9_MEDIA_SURFACE_NOT_ACQUIRED_KHR";
        case -1093: return "CL_INVALID_EGL_OBJECT_KHR";
        case -1092: return "CL_EGL_RESOURCE_NOT_ACQUIRED_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1057: return "CL_DEVICE_PARTITION_FAILED_EXT";
        case -1058: return "CL_INVALID_PARTITION_COUNT_EXT";
        case -1059: return "CL_INVALID_PARTITION_NAME_EXT";
        case -1094: return "CL_INVALID_ACCELERATOR_INTEL";
        case -1095: return "CL_INVALID_ACCELERATOR_TYPE_INTEL";
        case -1096: return "CL_INVALID_ACCELERATOR_DESCRIPTOR_INTEL";
        case -1097: return "CL_ACCELERATOR_TYPE_NOT_SUPPORTED_INTEL";
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1098: return "CL_INVALID_VA_API_MEDIA_ADAPTER_INTEL";
        case -1099: return "CL_INVALID_VA_API_MEDIA_SURFACE_INTEL";
        case -1100: return "CL_VA_API_MEDIA_SURFACE_ALREADY_ACQUIRED_INTEL";
        case -1101: return "CL_VA_API_MEDIA_SURFACE_NOT_ACQUIRED_INTEL";
        default: return "CL_UNKNOWN_ERROR";
    }
}

const char* clCheckError(int errorCode) {
    const char* error = clGetErrorString(errorCode);
    printf("FATAL ERROR: %s\n", error);
    exit(-1);
    assert(0);
    return error;
}

dim2 dim2_create(const int x, const int y)
{
    dim2 ret;

    ret.x = x;
    ret.y = y;

    return ret;
}

dim2 opencl_gridsize(const int n)
{
    dim2 ret = dim2_create(n, 1);

    return ret;
}

void opencl_set_device(int n)
{
    opencl_device_ct_t = ngpusg;
    opencl_device_id_t = n;
}

void opencl_load(const char *fileName, cl_program *output)
{
    FILE *fp;
    size_t lSize, readSize;
    char * sourceBuffer;

    fp = fopen(fileName, "r");

    if (fp == NULL)
    {
        printf("opencl_load: Could not open file: %s\n", fileName);
        fclose(fp);
        return;
    }

    // Determine file size.
    fseek(fp, 0, SEEK_END);
    lSize = ftell(fp);
    rewind(fp);

    sourceBuffer = (char*) malloc(sizeof(char) * lSize);

    if (sourceBuffer == NULL)
    {
        printf("opencl_load: Could not allocate memory for file: %s\n",
               fileName);
        fclose(fp);
        return;
    }

    readSize = fread(sourceBuffer, 1, lSize, fp);
    fclose(fp);

    if (readSize > lSize)
    {
        printf("opencl_load: failed to read file: %s\n", fileName);
        free(sourceBuffer);
        return;
    }

    opencl_load_buffer(sourceBuffer, readSize, output);

    free(sourceBuffer);
}

char* concat(const char *s1, const char *s2)
{
    char *result = calloc(strlen(s1) + strlen(s2) + 1, sizeof(char));
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

void opencl_load_buffer(const char *buffer, const size_t size, cl_program *output)
{
    cl_int clErr;

    *output = clCreateProgramWithSource(opencl_context, 1,
                                        (const char**)&buffer, &size, &clErr);

    if (clErr != CL_SUCCESS)
    {
        printf("opencl_load: could not create program. error: %s\n", clCheckError(clErr));
        exit(-1);
    }

#ifdef ARM
    clErr = clBuildProgram(
            *output,
            1,
            &opencl_devices[opencl_device_id_t],
            NULL, NULL, NULL);
#else
    clErr = clBuildProgram(
            *output,
            1,
            &opencl_devices[opencl_device_id_t],
            "-Werror "
            "-cl-std=CL1.2 "
            "-cl-opt-disable "
          //"-cl-denorms-are-zero "
          //"-cl-fp32-correctly-rounded-divide-sqrt "
            "-cl-no-signed-zeros "
            "-cl-mad-enable "
          //"-cl-fast-relaxed-math "
            , NULL, NULL);
#endif

    if (clErr != CL_SUCCESS)
    {
        printf("opencl_load: could not compile. error: %s\n", clCheckError(clErr));
        size_t len;
        char *ebuffer = (char*)calloc(0x10000000, sizeof(char));
        clGetProgramBuildInfo(*output, opencl_devices[opencl_device_id_t], CL_PROGRAM_BUILD_LOG, 0x10000000 * sizeof(char), ebuffer, &len);
        printf("code:\n%s\n", ebuffer);
        free(ebuffer);
        exit(-1);
    }
}

void opencl_create_kernel(cl_program *program, const char *kernelName,
                          cl_kernel *kernel)
{
    cl_int clErr;

    *kernel = clCreateKernel(*program, kernelName, &clErr);

    if (clErr)
    {
        printf("opencl_create_kernel: Could not create kernel %s.\n",
               kernelName);
    }
}

void opencl_init(int *gpus, int ngpus) {
    opencl_device_ct_t = ngpus;

    cl_native_double_width_s = calloc(ngpus, sizeof(int));
    cl_native_max_group_size_s = calloc(ngpus, sizeof(int));
    cl_native_address_bits_s = calloc(ngpus, sizeof(int));

    opencl_devices = (cl_device_id *) calloc((cl_uint)ngpus, sizeof(cl_device_id));
    opencl_queues = (cl_command_queue *) calloc((cl_uint)ngpus, sizeof(cl_command_queue));

    cl_int clErr;

    // Create OpenCL context from scratch.
    cl_platform_id clPlatform = 0;
    cl_uint clNumPlatforms = 0;
    cl_uint clnumEntries = ngpus;

    cl_props = calloc(3, sizeof(cl_context_properties));
    cl_props[0] = CL_CONTEXT_PLATFORM;
    cl_props[1] = 0;
    cl_props[2] = 0;

    clErr = clGetPlatformIDs(clnumEntries, &clPlatform, &clNumPlatforms);

    if (clErr != CL_SUCCESS) {
        printf("opencl_init: Could not get platform IDs.\n");
        exit(-1);
    }

    cl_uint num = 32;
    cl_uint all = 0;
    cl_device_id devices[num];
    clErr = clGetDeviceIDs(clPlatform, CL_DEVICE_TYPE_GPU, num, devices, &all);

    if (clErr != CL_SUCCESS) {
        printf("opencl_init: Could not get device IDs.\n");
        exit(-1);
    }
    else {
        printf("Device IDs: %d\n", all);
    }

    int i;
    for(i = 0; i < ngpus; ++i)
    {
        opencl_devices[i] = devices[gpus[i]];
    }

    cl_props[1] = (cl_context_properties) clPlatform;

    opencl_context = clCreateContext(cl_props, ngpus, opencl_devices, NULL, NULL, &clErr);

    if (clErr != CL_SUCCESS) {
        printf("opencl_init: Could not create context.\n");
        exit(-1);
    }

    int d;
    for (d = 0; d < ngpus; ++d) {
        opencl_device_id_t = d;

        opencl_queues[opencl_device_id_t] = clCreateCommandQueue(opencl_context,
                                                                 opencl_devices[opencl_device_id_t],
                                                                 CL_FALSE, &clErr);

        if (clErr != CL_SUCCESS) {
            printf("opencl_init: Could not create queue.\n");
            exit(-1);
        }

        cl_native_double_width_s[opencl_device_id_t] = 0;
        cl_native_max_group_size_s[opencl_device_id_t] = 0;
        cl_native_address_bits_s[opencl_device_id_t] = 0;

        clGetDeviceInfo(opencl_devices[opencl_device_id_t], CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), &cl_native_double_width_s[opencl_device_id_t], NULL);
        clGetDeviceInfo(opencl_devices[opencl_device_id_t], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &cl_native_max_group_size_s[opencl_device_id_t], NULL);
        clGetDeviceInfo(opencl_devices[opencl_device_id_t], CL_DEVICE_ADDRESS_BITS, sizeof(size_t), &cl_native_address_bits_s[opencl_device_id_t], NULL);

#if defined(DARKNET_VERBOSE_GPU)
        // Print out usefull information.
        const size_t bufferSize = 2048;
        char *buffer = (char *) calloc(bufferSize, sizeof(char));

        printf("Device ID: %d\n", gpus[opencl_device_id_t]);
        clGetDeviceInfo(opencl_devices[opencl_device_id_t], CL_DEVICE_NAME, bufferSize * sizeof(char), buffer, NULL);
        printf("Device name: %s\n", buffer);
        clGetDeviceInfo(opencl_devices[opencl_device_id_t], CL_DEVICE_VENDOR, bufferSize * sizeof(char), buffer, NULL);
        printf("Device vendor: %s\n", buffer);
        clGetDeviceInfo(opencl_devices[opencl_device_id_t], CL_DEVICE_VERSION, bufferSize * sizeof(char), buffer, NULL);
        printf("Device opencl availability: %s\n", buffer);
        clGetDeviceInfo(opencl_devices[opencl_device_id_t], CL_DRIVER_VERSION, bufferSize * sizeof(char), buffer, NULL);
        printf("Device opencl used: %s\n", buffer);
        printf("Device double precision: %s\n", cl_native_double_width_s[opencl_device_id_t] == 0 ? "NO" : "YES");
        printf("Device max group size: %zu\n", cl_native_max_group_size_s[opencl_device_id_t]);
        printf("Device address bits: %zu\n", cl_native_address_bits_s[opencl_device_id_t]);
        free(buffer);

        sleep(1);
#endif
        activation_kernel_init();
        blas_kernel_init();
        col2im_kernel_init();
        convolutional_kernel_init();
        im2col_kernel_init();
        maxpool_kernel_init();
        gemm_kernel_init();
        avgpool_kernel_init();
#ifndef ARM
        crop_kernel_init();
#endif
        dropout_kernel_init();
    }

    // TODO: TEST CPU & GPU (3)
#if defined(DARKNET_TEST_CPU_AND_GPU)
    opencl_cpu_gpu_test();
    opencl_deinit(gpus, ngpus);
    exit(0);
#endif
#ifdef UNIT_TESTS
    opencl_kernel_unit_tests();
    opencl_deinit(gpus, ngpus);
    exit(0);
#endif
}

void opencl_deinit(int *gpus, int ngpus)
{
    int a;
    int d;
    for (a = 0, d = ngpus-1; a < ngpus; --d, ++a) {
        opencl_device_id_t = a;

        clFinish(opencl_queues[opencl_device_id_t]);

        activation_kernel_release();
        blas_kernel_release();
        col2im_kernel_release();
        convolutional_kernel_release();
        im2col_kernel_release();
        maxpool_kernel_release();
        gemm_kernel_release();
        avgpool_kernel_release();
#ifndef ARM
        crop_kernel_release();
#endif
        dropout_kernel_release();

        clReleaseCommandQueue(opencl_queues[opencl_device_id_t]);
    }

    clReleaseContext(opencl_context);

    //free(cl_props);

    free(opencl_queues);
    free(opencl_devices);

    free(cl_native_double_width_s);
    free(cl_native_max_group_size_s);
    free(cl_native_address_bits_s);

    gpu_index = -1;
}

void opencl_kernel(cl_kernel kernel, const dim2 globalItemSize, const int argc, ...)
{
    cl_int clErr;

    cl_command_queue que = opencl_queues[opencl_device_id_t];

    va_list vl;
    va_start(vl, argc);

    size_t argSize = 0;
    void *argValue = NULL;

    int i, j;
    for (i = 0, j = 0; i < argc; i+=2, ++j)
    {
        argValue = va_arg(vl, void*);
        argSize = va_arg(vl, size_t);

        assert(argValue);

        clErr = clSetKernelArg(kernel, j, argSize, argValue);

        if (clErr != CL_SUCCESS)
        {
            const size_t bufferSize = 2048;
            char *kernelName = (char*) calloc(bufferSize, sizeof(char));

            clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
            printf("opencl_kernel %s could not set kernel argument. error: %s\n", kernelName, clCheckError(clErr));

            free(kernelName);
            exit(-1);
        }
    }

    va_end(vl);

    size_t globalOffset[2], globalItems[2];
    globalOffset[0] = 0;
    globalOffset[1] = 0;

    globalItems[0] = globalItemSize.x;
    globalItems[1] = globalItemSize.y;

#ifdef BENCHMARK
    clock_t t;
    t = clock();
#endif

    clErr = clEnqueueNDRangeKernel(que, kernel, 2, globalOffset, globalItems, NULL, 0, NULL, NULL);

    clFlush(que);

#ifdef BENCHMARK
    t = clock() - t;
    double time_taken = ((double)t);
    const size_t bufferSize = 2048;
    char *kernelName = (char*) calloc(bufferSize, sizeof(char));
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
    printf("%s\t%d\n", kernelName, (int)time_taken);
    free(kernelName);
#endif

    if (clErr != CL_SUCCESS)
    {
        const size_t bufferSize = 2048;
        char *kernelName = (char*) calloc(bufferSize, sizeof(char));

        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
        printf("opencl %s error: %s\n", kernelName, clCheckError(clErr));

        free(kernelName);
        exit(-1);
    }
#ifdef DEBUG_KERNELS
    else {
        const size_t bufferSize = 2048;
        char *kernelName = (char*) calloc(bufferSize, sizeof(char));
        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
        printf("opencl %s : ", kernelName);
        free(kernelName);
        printf("OK \n");
    }
#endif
}

void opencl_kernel_local(cl_kernel kernel, const dim2 globalItemSize, const dim2 localItemSize, const int argc, ...)
{
    cl_int clErr;

    cl_command_queue que = opencl_queues[opencl_device_id_t];

    va_list vl;
    va_start(vl, argc);

    size_t argSize = 0;
    void *argValue = NULL;

    int i, j;
    for (i = 0, j = 0; i < argc; i+=2, ++j)
    {
        argValue = va_arg(vl, void*);
        argSize = va_arg(vl, size_t);

        // I need NULL to __local arrays
        // assert(argValue);

        clErr = clSetKernelArg(kernel, j, argSize, argValue);

        if (clErr != CL_SUCCESS)
        {
            const size_t bufferSize = 2048;
            char *kernelName = (char*) calloc(bufferSize, sizeof(char));

            clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
            printf("opencl_kernel %s could not set kernel argument. error: %s\n", kernelName, clCheckError(clErr));

            free(kernelName);
            exit(-1);
        }
    }

    va_end(vl);

    size_t globalOffset[2];
    globalOffset[0] = 0;
    globalOffset[1] = 0;

    size_t globalItems[2];
    globalItems[0] = globalItemSize.x;
    globalItems[1] = globalItemSize.y;

    size_t localItems[2];
    localItems[0] = localItemSize.x;
    localItems[1] = localItemSize.y;

#ifdef BENCHMARK
    clock_t t;
    t = clock();
#endif

    clErr = clEnqueueNDRangeKernel(que, kernel, 2, globalOffset, globalItems, localItems, 0, NULL, NULL);

    clFlush(que);

#ifdef BENCHMARK
    t = clock() - t;
    double time_taken = ((double)t);
    const size_t bufferSize = 2048;
    char *kernelName = (char*) calloc(bufferSize, sizeof(char));
    clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
    printf("%s\t%d\n", kernelName, (int)time_taken);
    free(kernelName);
#endif

    if (clErr != CL_SUCCESS)
    {
        const size_t bufferSize = 2048;
        char *kernelName = (char*) calloc(bufferSize, sizeof(char));

        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
        printf("opencl %s error: %s\n", kernelName, clCheckError(clErr));

        free(kernelName);
        exit(-1);
    }
#ifdef DEBUG_KERNELS
    else {
        const size_t bufferSize = 2048;
        char *kernelName = (char*) calloc(bufferSize, sizeof(char));
        clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, bufferSize, kernelName, NULL);
        printf("opencl %s : ", kernelName);
        free(kernelName);
        printf("OK \n");
    }
#endif
}

cl_mem_ext opencl_random(cl_mem_ext x_gpu, size_t n)
{
    int i;
    float *m;
    if (!x_gpu.ptr)
    {
        m = calloc(n, sizeof(float));
    } else {
        m = x_gpu.ptr;
    }
    for(i = 0; i < n; ++i){
        m[i] = (float)rand()/RAND_MAX;
    }
    if (!x_gpu.ptr) {
        x_gpu = opencl_make_array(m, n);
    }
    else {
        opencl_push_array(x_gpu, m, n);
    }
    return x_gpu;
}

float opencl_compare(cl_mem_ext x_gpu, float *x, size_t n, char *s)
{
    float *tmp = (float*)calloc(n, sizeof(float));
    opencl_pull_array(x_gpu, tmp, n);

    axpy_cpu(n, -1, x, 1, tmp, 1);
    float err = dot_cpu(n, tmp, 1, tmp, 1);
    printf("Error %s: %f\n", s, sqrtf(err/n));
    free(tmp);
    return err;
}

float opencl_mag_array(cl_mem_ext x_gpu, size_t n)
{
    float *temp = (float*) calloc(n, sizeof(float));
    opencl_pull_array(x_gpu, temp, n);

    float m = mag_array(temp, n);

    if(temp) free(temp);

    return m;
}

void opencl_dump_mem_stat()
{
    size_t used, total;

    clGetDeviceInfo(opencl_devices[opencl_device_id_t], CL_DEVICE_GLOBAL_MEM_SIZE,
                    sizeof(size_t), &total, NULL);

    clGetDeviceInfo(opencl_devices[opencl_device_id_t], CL_DEVICE_LOCAL_MEM_SIZE,
                    sizeof(size_t), &used, NULL);

    printf("OpenCL memory status: Used/Free/Total = [%lu]/[%lu]/[%lu]\n", used, total - used, total);
}

cl_mem_ext opencl_make_array(float *x, size_t n)
{
    assert(x && n);

    cl_mem_ext buf;

    buf.len = n;
    buf.obs = sizeof(cl_float);
    buf.off = 0;
    buf.cnt = 0;

    buf.ptr = x;

    cl_int clErr;

    buf.org = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
                             buf.len * buf.obs, buf.ptr,
                             &clErr);

    if (clErr != CL_SUCCESS)
        printf("could create buffer on device. error: %s\n", clCheckError(clErr));

    buf.mem = buf.org;

    buf.cln = cln;
    buf.inc = inc;
    buf.dec = dec;
    buf.add = add;
    buf.rem = rem;

    buf.que = opencl_queues[opencl_device_id_t];

    buf.idx = opencl_device_id_t;

    return buf;
}

cl_mem_ext opencl_make_int_array(int *x, size_t n)
{
    assert(x && n);

    cl_mem_ext buf;

    buf.len = n;
    buf.obs = sizeof(cl_int);
    buf.off = 0;
    buf.cnt = 0;

    buf.ptr = x;

    cl_int clErr;

    buf.org = clCreateBuffer(opencl_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
                             buf.len * buf.obs, buf.ptr,
                             &clErr);

    if (clErr != CL_SUCCESS)
        printf("could create buffer on device. error: %s\n", clCheckError(clErr));

    buf.mem = buf.org;

    buf.cln = cln;
    buf.inc = inc;
    buf.dec = dec;
    buf.add = add;
    buf.rem = rem;

    buf.que = opencl_queues[opencl_device_id_t];

    buf.idx = opencl_device_id_t;

    return buf;
}

void opencl_push_int_array(cl_mem_ext x_gpu, int *x, size_t n)
{
#ifdef BENCHMARK
    clock_t t;
    t = clock();
#endif
    if (x_gpu.ptr == (void*)x) {
        cl_int clErr = clEnqueueWriteBuffer(x_gpu.que, x_gpu.mem, CL_TRUE, x_gpu.off * x_gpu.obs, (n - x_gpu.off) * x_gpu.obs, x, 0, NULL, NULL);
        if (clErr != CL_SUCCESS)
            printf("could not push array to device. error: %s\n", clCheckError(clErr));
    }
#ifdef BENCHMARK
    t = clock() - t;
    double time_taken = ((double)t);
    printf("%s\t%x\t%d\n", "opencl_push_int_array", x_gpu.ptr, (int)time_taken);
#endif
}

void opencl_pull_int_array(cl_mem_ext x_gpu, int *x, size_t n)
{
#ifdef BENCHMARK
    clock_t t;
    t = clock();
#endif
    if (x_gpu.ptr == (void*)x) {
        cl_int clErr = clEnqueueReadBuffer(x_gpu.que, x_gpu.mem, CL_TRUE, x_gpu.off * x_gpu.obs, (n - x_gpu.off) * x_gpu.obs, x, 0, NULL, NULL);
        if (clErr != CL_SUCCESS)
            printf("could not pull array from device. error: %s\n", clCheckError(clErr));
    }
#ifdef BENCHMARK
    t = clock() - t;
    double time_taken = ((double)t);
    printf("%s\t%x\t%d\n", "opencl_pull_int_array", x_gpu.ptr, (int)time_taken);
#endif
}

void opencl_push_array(cl_mem_ext x_gpu, float *x, size_t n)
{
#ifdef BENCHMARK
    clock_t t;
    t = clock();
#endif
    if (x_gpu.ptr == (void*)x) {
        cl_int clErr = clEnqueueWriteBuffer(x_gpu.que, x_gpu.mem, CL_TRUE, x_gpu.off * x_gpu.obs, (n - x_gpu.off) * x_gpu.obs, x, 0, NULL, NULL);
        if (clErr != CL_SUCCESS)
            printf("could not push array to device. error: %s\n", clCheckError(clErr));
    }
#ifdef BENCHMARK
    t = clock() - t;
    double time_taken = ((double)t);
    printf("%s\t%x\t%d\n", "opencl_push_array", x_gpu.ptr, (int)time_taken);
#endif
}

void opencl_pull_array(cl_mem_ext x_gpu, float *x, size_t n)
{
#ifdef BENCHMARK
    clock_t t;
    t = clock();
#endif
    if (x_gpu.ptr == (void*)x) {
        cl_int clErr = clEnqueueReadBuffer(x_gpu.que, x_gpu.mem, CL_TRUE, x_gpu.off * x_gpu.obs, (n - x_gpu.off) * x_gpu.obs, x, 0, NULL, NULL);
        if (clErr != CL_SUCCESS)
            printf("could not pull array from device. error: %s\n", clCheckError(clErr));
    }
#ifdef BENCHMARK
    t = clock() - t;
    double time_taken = ((double)t);
    printf("%s\t%x\t%d\n", "opencl_pull_array", x_gpu.ptr, (int)time_taken);
#endif
}

void opencl_push_array_map(cl_mem_ext x_gpu, void *x, size_t n)
{
#ifdef BENCHMARK
    clock_t t;
    t = clock();
#endif
    if (x_gpu.obs == sizeof (cl_float)) {
        opencl_push_array(x_gpu, x, n);
    }
    if (x_gpu.obs == sizeof (cl_int)) {
        opencl_push_int_array(x_gpu, x, n);
    }
    /*
    cl_int clErr;
    x_gpu.map = clEnqueueMapBuffer(x_gpu.que, x_gpu.mem, CL_TRUE, CL_MAP_WRITE,
                                   x_gpu.off * x_gpu.obs, (n - x_gpu.off) * x_gpu.obs, 0, NULL, NULL, &clErr);
    if (clErr != CL_SUCCESS) {
        printf("could not map array to device. error: %s\n", clCheckError(clErr));
        exit(1);
    }
    if (x_gpu.obs == sizeof (cl_float)) {
        memcpy((float *)x_gpu.map, (float *)x, (n - x_gpu.off) * x_gpu.obs);
    }
    if (x_gpu.obs == sizeof (cl_int)) {
        memcpy((int *)x_gpu.map, (int *)x, (n - x_gpu.off) * x_gpu.obs);
    }
    clErr = clEnqueueUnmapMemObject(x_gpu.que, x_gpu.mem, x_gpu.map, 0, NULL, NULL);
    if (clErr != CL_SUCCESS)
        printf("could not unmap array from device. error: %s\n", clCheckError(clErr));
    */
#ifdef BENCHMARK
    t = clock() - t;
    double time_taken = ((double)t);
    printf("%s\t%x\t%d\n", "opencl_push_array", x_gpu.ptr, (int)time_taken);
#endif
}

void opencl_pull_array_map(cl_mem_ext x_gpu, void *x, size_t n)
{
#ifdef BENCHMARK
    clock_t t;
    t = clock();
#endif
    if (x_gpu.obs == sizeof (cl_float)) {
        opencl_pull_array(x_gpu, x, n);
    }
    if (x_gpu.obs == sizeof (cl_int)) {
        opencl_pull_int_array(x_gpu, x, n);
    }
    /*
    cl_int clErr;
    x_gpu.map = clEnqueueMapBuffer(x_gpu.que, x_gpu.mem, CL_TRUE, CL_MAP_READ,
                                   x_gpu.off * x_gpu.obs, (n - x_gpu.off) * x_gpu.obs, 0, NULL, NULL, &clErr);
    if (clErr != CL_SUCCESS) {
        printf("could not map array to device. error: %s\n", clCheckError(clErr));
        exit(1);
    }
    if (x_gpu.obs == sizeof(cl_float)) {
        memcpy((float *)x, (float *)x_gpu.map, (n - x_gpu.off) * x_gpu.obs);
    }
    if (x_gpu.obs == sizeof(cl_int)) {
        memcpy((int *)x, (int *)x_gpu.map, (n - x_gpu.off) * x_gpu.obs);
    }
    clErr = clEnqueueUnmapMemObject(x_gpu.que, x_gpu.mem, x_gpu.map, 0, NULL, NULL);
    if (clErr != CL_SUCCESS)
        printf("could not unmap array from device. error: %s\n", clCheckError(clErr));
    */
#ifdef BENCHMARK
    t = clock() - t;
    double time_taken = ((double)t);
    printf("%s\t%x\t%d\n", "opencl_pull_array", x_gpu.ptr, (int)time_taken);
#endif
}

void opencl_free(cl_mem_ext x_gpu)
{
    if(!x_gpu.ptr) return;
    x_gpu.len = 0;
    x_gpu.obs = 0;
    x_gpu.mem = 0;
    x_gpu.off = 0;
    x_gpu.cnt = 0;
    x_gpu.cln = 0;
    x_gpu.inc = 0;
    x_gpu.dec = 0;
    x_gpu.add = 0;
    x_gpu.rem = 0;
    clReleaseMemObject(x_gpu.org);
    x_gpu.org = 0;
    x_gpu.map = 0;
    x_gpu.que = 0;
    if(x_gpu.ptr) free(x_gpu.ptr);
    x_gpu.ptr = 0;
}

void opencl_free_gpu_only(cl_mem_ext x_gpu)
{
    if(!x_gpu.ptr) return;
    x_gpu.len = 0;
    x_gpu.obs = 0;
    x_gpu.mem = 0;
    x_gpu.off = 0;
    x_gpu.cnt = 0;
    x_gpu.cln = 0;
    x_gpu.inc = 0;
    x_gpu.dec = 0;
    x_gpu.add = 0;
    x_gpu.rem = 0;
    clReleaseMemObject(x_gpu.org);
    x_gpu.org = 0;
    x_gpu.map = 0;
    x_gpu.que = 0;
}

cl_mem_ext cln(cl_mem_ext buf) {
    buf.mem = buf.org;
    buf.off = 0;
    buf.cnt = 0;
    return buf;
}

cl_mem_ext inc(cl_mem_ext buf, int inc, size_t len) {
    buf.off += inc;
    buf.cnt += 1;
    return mov(buf, len);
}

cl_mem_ext dec(cl_mem_ext buf, int dec, size_t len) {
    buf.off -= dec;
    buf.cnt -= 1;
    return mov(buf, len);
}

cl_mem_ext mov(cl_mem_ext buf, size_t len) {
    cl_buffer_region region;

    region.origin = buf.off * buf.obs;
    region.size = len != 0 ? len * buf.obs : (buf.len - buf.off) * buf.obs;

    cl_int clErr = 0;
    buf.mem = clCreateSubBuffer(buf.org, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &clErr);

    if (clErr != CL_SUCCESS)
    {
        printf("could not create sub-buffer on device. error: %s\n", clCheckError(clErr));
    }

    return buf;
}

cl_mem_ext add(cl_mem_ext buf, int inc, size_t len) {
    buf.off += inc;
    buf.cnt += 1;
    return upd(buf, len);
}

cl_mem_ext rem(cl_mem_ext buf, int dec, size_t len) {
    buf.off -= dec;
    buf.cnt -= 1;
    return upd(buf, len);
}

cl_mem_ext upd(cl_mem_ext buf, size_t len) {
    cl_mem_ext ret;

    ret.org = buf.org;

    ret.len = buf.len;
    ret.obs = buf.obs;
    ret.org = buf.org;
    ret.off = buf.off;
    ret.cnt = buf.cnt;

    ret.cln = cln;
    ret.inc = inc;
    ret.dec = dec;
    ret.add = add;
    ret.rem = rem;

    cl_buffer_region region;

    region.origin = ret.off * ret.obs;
    region.size = len != 0 ? len * ret.obs : (ret.len - ret.off) * ret.obs;

    cl_int clErr = 0;
    ret.mem = clCreateSubBuffer(ret.org, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &clErr);

    if (clErr != CL_SUCCESS)
    {
        printf("could not create sub-buffer on device. error: %s\n", clCheckError(clErr));
    }

    return ret;
}

#endif // GPU
