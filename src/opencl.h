#ifndef OPENCL_H
#define OPENCL_H

#if defined(_WIN32) && !defined(WIN32)
#define WIN32
#endif

#define CL_TARGET_OPENCL_VERSION 120

//#define GPU_STATS

#ifdef ARM
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#endif

#ifdef GPU
#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/cl.h>
#include <OpenCL/cl_ext.h>
#else
#include <CL/cl.h>
#include <CL/cl_ext.h>
#endif

#include <math.h>

#ifndef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#endif
#ifndef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#endif

extern int *gpusg;
extern int ngpusg;
#ifdef WIN32
extern __declspec(thread) int opencl_device_id_t;
extern __declspec(thread) int opencl_device_ct_t;
#else
extern __thread int opencl_device_id_t;
extern __thread int opencl_device_ct_t;
#endif
extern cl_int *cl_native_double_width_s;
extern size_t *cl_native_max_group_size_s;
extern size_t *cl_native_address_bits_s;
extern cl_context opencl_context;
extern cl_command_queue* opencl_queues;
extern cl_device_id* opencl_devices;

typedef struct _cl_mem_ext cl_mem_ext;

typedef struct _cl_mem_ext {
    cl_mem mem;
    cl_mem org;
    size_t len;
    size_t off;
    size_t obs;
    size_t cnt;
    cl_mem_ext (*cln) (cl_mem_ext buf);
    cl_mem_ext (*inc) (cl_mem_ext buf, int inc, size_t len);
    cl_mem_ext (*dec) (cl_mem_ext buf, int dec, size_t len);
    cl_mem_ext (*add) (cl_mem_ext buf, int add, size_t len);
    cl_mem_ext (*rem) (cl_mem_ext buf, int rem, size_t len);
    void* ptr;
    void* map;
    cl_command_queue que;
    void* chk;
} cl_mem_ext;

cl_mem_ext cln(cl_mem_ext buf);
cl_mem_ext inc(cl_mem_ext buf, int inc, size_t len);
cl_mem_ext dec(cl_mem_ext buf, int dec, size_t len);
cl_mem_ext mov(cl_mem_ext buf, size_t len);
cl_mem_ext add(cl_mem_ext buf, int inc, size_t len);
cl_mem_ext rem(cl_mem_ext buf, int dec, size_t len);
cl_mem_ext upd(cl_mem_ext buf, size_t len);
cl_command_queue que();

void activation_kernel_init(void);
void blas_kernel_init(void);
void col2im_kernel_init(void);
void convolutional_kernel_init(void);
void im2col_kernel_init(void);
void maxpool_kernel_init(void);
void gemm_kernel_init(void);
void avgpool_kernel_init(void);
void crop_kernel_init(void);
void dropout_kernel_init(void);

void activation_kernel_release(void);
void blas_kernel_release(void);
void col2im_kernel_release(void);
void convolutional_kernel_release(void);
void im2col_kernel_release(void);
void maxpool_kernel_release(void);
void gemm_kernel_release(void);
void avgpool_kernel_release(void);
void crop_kernel_release(void);
void dropout_kernel_release(void);

typedef struct _dim2
{
    size_t x;
    size_t y;
} dim2;
dim2 dim2_create(const int x, const int y);
typedef struct _dim3
{
    size_t x;
    size_t y;
    size_t z;
} dim3;
dim3 dim3_create(const int x, const int y, const int z);

#define CONVERT_KERNEL_TO_STRING(...) # __VA_ARGS__

void opencl_load(const char *fileName, cl_program *output);
void opencl_load_buffer(const char *bufferName, const size_t size, cl_program *output);
void opencl_create_kernel(cl_program *program, const char *kernalName, cl_kernel *kernel);
void opencl_init(int *gpus, int ngpus);
void opencl_deinit(int *gpus, int ngpus);
void opencl_kernel(cl_kernel kernel, const dim2 globalItemSize, const int argc, ...);
void opencl_kernel_local(cl_kernel kernel, const dim2 globalItemSize, const dim2 localItemSize, const int argc, ...);
void opencl_kernel_local3(cl_kernel kernel, const dim3 globalItemSize, const dim3 localItemSize, const int argc, ...);

cl_mem_ext opencl_random(cl_mem_ext x_gpu, size_t n);

cl_mem_ext opencl_make_array(float *x, size_t n);
cl_mem_ext opencl_make_int_array(int *x, size_t n);
dim2 opencl_gridsize(const int n);
void opencl_dump_mem_stat();

void opencl_pull_int_array(cl_mem_ext x_gpu, int *x, size_t n);
void opencl_push_int_array(cl_mem_ext x_gpu, int *x, size_t n);

void opencl_pull_array(cl_mem_ext x_gpu, float *x, size_t n);
void opencl_push_array(cl_mem_ext x_gpu, float *x, size_t n);

void opencl_pull_int_array_map(cl_mem_ext x_gpu, int *x, size_t n);
void opencl_push_int_array_map(cl_mem_ext x_gpu, int *x, size_t n);

void opencl_pull_array_map(cl_mem_ext x_gpu, void *x, size_t n);
void opencl_push_array_map(cl_mem_ext x_gpu, void *x, size_t n);

#endif // GPU
#endif // OPENCL_H
