#ifndef __BLAS_KERNELS_1_CL__
#define __BLAS_KERNELS_1_CL__

static const char* const blas_kernel_source_1 = CONVERT_KERNEL_TO_STRING(

/*
static void atomicAdd(volatile __global float *a, float v) {
    //float s = v;
    //float n = 0;
    //float o = 0;
    //do {
    //    n = s + atom_xchg(a, o);
    //    s = o + atom_xchg(a, n);
    //}
    //while (s != o);
    union {
        float f;
        unsigned int i;
    } o;
    o.i = 0;
    union {
        float f;
        unsigned int i;
    } n;
    n.i = 1;
    do {
        o.f = *a;
        n.f = o.f + v;
    } while (atom_cmpxchg((__global unsigned int *)a, o.i, n.i) != o.i);
}
*/


__kernel void test_kernel(int N, __global float *input, __global float *output, __global float *expected)
{
        int index = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

        if (index >= N) return;

        output[index] = sqrt(input[index]);

        index += 1;
        input[index] = output[index-1];
        output[index] = log(input[index]);

        index += 1;
        input[index] = output[index-1];
        output[index] = pow(input[index], output[index-2]);

        index += 1;
        input[index] = output[index-1];
        output[index] = -exp(input[index]);

        index += 1;
        input[index] = output[index-1];
        output[index] = fabs(input[index]);

        index += 1;
        input[index] = output[index-1];
        output[index] = sin(input[index]);

        index += 1;
        input[index] = output[index-1];
        output[index] = cos(input[index]);
}


__kernel void scale_bias_kernel(int N, __global float *output, __global float *biases, int batch, int n, int size)
{
    int index = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

    if (index >= N) return;

    int i = (index % (n*size) / size);

    output[index] *= biases[i];
}


__kernel void backward_scale_kernel(int tuning, __local float* sums, int batch, int n, int size, __global float *x_norm, __global float *delta, __global float *scale_updates)
{
    int t = get_global_id(0);
    if (t > tuning) return;
    int i = get_global_id(1);
    if (i > n) return;

    sums[t] = 0;

    int k,j,s;
    for(j = 0; j < batch; ++j){
        for(k = t; k < size; k += tuning){
            int index = size*i + size*n*j + k + t;
            sums[t] += delta[index]*x_norm[index];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(s = 0; s < tuning; ++s) {
        scale_updates[i] += sums[s];
    }
}


__kernel void add_bias_kernel(int N, __global float *output, __global float *biases, int batch, int n, int size)
{
    int index = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

    if (index >= N) return;

    int i = (index % (n*size) / size);

    output[index] += biases[i];
}


__kernel void backward_bias_kernel(int tuning, __local float* sums, int batch, int n, int size, __global float *bias_updates, __global float *delta)
{
    int t = get_global_id(0);
    if (t > tuning) return;
    int i = get_global_id(1);
    if (i > n) return;

    sums[t] = 0;

    int k,j,s;
    for(j = 0; j < batch; ++j){
        for(k = t; k < size; k += tuning){
            int index = size*i + size*n*j + k + t;
            sums[t] += delta[index];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for(s = 0; s < tuning; ++s) {
        bias_updates[i] += sums[s];
    }
}

__kernel void  mean_kernel(int N, __global float *x, int batch, int filters, int spatial, __global float *mean)
{
    float scale = 1.f/(batch * spatial);

    int id = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (id >= N) return;

    int i = id;
    mean[i] = 0;
    int j, k;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j * filters * spatial + i * spatial + k;
            mean[i] += x[index];
        }
    }
    mean[i] *= scale;
}


__kernel void variance_kernel(int N, __global float *x, __global float *mean, int batch, int filters, int spatial, __global float *variance)
{
    float scale = 1.f/(batch * spatial - 1);

    int id = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (id >= N) return;

    int i = id;
    variance[i] = 0;
    int j,k;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j * filters * spatial + i * spatial + k;
            variance[i] += pow((x[index] - mean[i]), 2);
        }
    }
    variance[i] *= scale;
}


__kernel void mean_delta_kernel(int N, __global float *delta, __global float *variance, int batch, int filters, int spatial, __global float *mean_delta) {
    int id = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (id >= N) return;

    int i = id;
    mean_delta[i] = 0;
    int j, k;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j * filters * spatial + i * spatial + k;
            mean_delta[i] += delta[index];
        }
    }

    mean_delta[i] *= (-1.f/sqrt(variance[i] + .000001f));
}


__kernel void variance_delta_kernel(int N, __global float *x, __global float *delta, __global float *mean, __global float *variance, int batch, int filters, int spatial, __global float *variance_delta) {
    int id = (get_group_id(0) + get_group_id(1) * get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (id >= N) return;

    int i = id;
    variance_delta[i] = 0;
    int j,k;
    for (j = 0; j < batch; ++j) {
        for (k = 0; k < spatial; ++k) {
            int index = j * filters * spatial + i * spatial + k;
            variance_delta[i] += delta[index] * (x[index] - mean[i]);
        }
    }
    variance_delta[i] *= -.5f * pow(variance[i] + .000001f, (float)(-3.f/2.f));
}


__kernel void accumulate_kernel(__global float *x, int n, int groups, __global float *sum)
{
    int k;
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (i >= groups) return;
    sum[i] = 0;
    for(k = 0; k < n; ++k){
        sum[i] += x[k*groups + i];
    }
}


__kernel void fast_mean_kernel(int tuning, __local float *sums, int filters, int batch, int spatial, __global float *x, __global float *mean) {
    int t = get_global_id(0);
    if (t >= tuning) return;
    int i = get_global_id(1);
    if (i >= filters) return;

    sums[t] = 0;

    int j, k, s;
    for (j = 0; j < batch; ++j) {
        for (k = t; k < spatial; k += tuning) {
            int index = j * filters * spatial + i * spatial + k;
            sums[t] += x[index];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    mean[i] = 0;
    for (s = 0; s < tuning; ++s) {
        mean[i] += sums[s];
    }
    mean[i] /= (spatial * batch);
}


__kernel void fast_variance_kernel(int tuning, __local float *sums, int filters, int batch, int spatial, __global float *x, __global float *mean, __global float *variance)
{
    int t = get_global_id(0);
    if (t >= tuning) return;
    int i = get_global_id(1);
    if (i >= filters) return;

    sums[t] = 0;

    int j,k,s;
    for (j = 0; j < batch; ++j) {
        for (k = t; k < spatial; k += tuning) {
            int index = j * filters * spatial + i * spatial + k;
            sums[t] += pow((x[index] - mean[i]), 2);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    variance[i] = 0;
    for (s = 0; s < tuning; ++s) {
        variance[i] += sums[s];
    }
    variance[i] /= (spatial * batch - 1);
}


 __kernel void fast_mean_delta_kernel(int tuning, __local float *sums, int filters, int batch, int spatial, __global float *variance, __global float *delta, __global float *mean_delta)
{
    int t = get_global_id(0);
    if (t >= tuning) return;
    int i = get_global_id(1);
    if (i >= filters) return;

    sums[t] = 0;

    int j,k,s;
    for (j = 0; j < batch; ++j) {
        for (k = t; k < spatial; k += tuning) {
            int index = j * filters * spatial + i * spatial + k;
            sums[t] += delta[index];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    mean_delta[i] = 0;
    for (s = 0; s < tuning; ++s) {
        mean_delta[i] += sums[s];
    }
    mean_delta[i] *= (-1.f/sqrt(variance[i] + .000001f));
}

__kernel void fast_variance_delta_kernel(int tuning, __local float *sums, int filters, int batch, int spatial, __global float *x, __global float *variance, __global float *delta, __global float *mean, __global float *variance_delta)
{
    int t = get_global_id(0);
    if (t >= tuning) return;
    int i = get_global_id(1);
    if (i >= filters) return;

    sums[t] = 0;

    int j,k,s;
    for (j = 0; j < batch; ++j) {
        for (k = t; k < spatial; k += tuning) {
            int index = j * filters * spatial + i * spatial + k;
            sums[t] += delta[index] * (x[index] - mean[i]);
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    variance_delta[i] = 0;
    for (s = 0; s < tuning; ++s) {
        variance_delta[i] += sums[s];
    }
    variance_delta[i] *= -.5f * pow(variance[i] + .000001f, (float)(-3.f/2.f));
}

__kernel void adam_kernel(int N, __global float *x, __global float *m, __global float *v, float B1, float B2, float rate, float eps, int t)
{
    int index = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (index >= N) return;

    x[index] = x[index] + (rate * sqrt(1.f-pow(B2, t)) / (1.f-pow(B1, t)) * m[index] / (sqrt((v[index] + eps))));
}


__kernel void normalize_kernel(int N, __global float *x, __global float *mean, __global float *variance, int batch, int filters, int spatial)
{
    int id = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (id >= N) return;

    int b = (id / (filters*spatial));
    int f = (id % (filters*spatial) / spatial);
    int i = (id % spatial);

    int index = b*filters*spatial + f*spatial + i;
    x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .00001f));
}



__kernel void normalize_delta_kernel(int N, __global float *x, __global float *mean, __global float *variance, __global float *mean_delta, __global float *variance_delta, int batch, int filters, int spatial, __global float *delta)
{
    int id = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (id >= N) return;

    int j = (id / (filters*spatial));
    int f = (id % (filters*spatial) / spatial);
    int k = (id % spatial);

    int index = j*filters*spatial + f*spatial + k;
    delta[index] = delta[index] * 1.f/(sqrt(variance[f] + .000001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
}


__kernel void l2norm_kernel(int N, __global float *x, __global float *dx, int batch, int filters, int spatial)
{
    int index = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (index >= N) return;
    int b = index / spatial;
    int i = index % spatial;
    int f;
    float sum = 0;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        sum += pow(x[index], 2.f);
    }
    sum = sqrt(sum);
    if(sum == 0) sum = 1.f;
    for(f = 0; f < filters; ++f){
        int index = b*filters*spatial + f*spatial + i;
        x[index] /= sum;
        dx[index] = (1 - x[index]) / sum;
    }
}

__kernel void reorg_kernel(int N, __global float *x, int w, int h, int c, int batch, int stride, int forward, __global float *out)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i >= N) return;
    int in_index = i;
    int in_w = i%w;
    i = i/w;
    int in_h = i%h;
    i = i/h;
    int in_c = i%c;
    i = i/c;
    int b = i%batch;

    int out_c = c/(stride*stride);

    int c2 = in_c % out_c;
    int offset = in_c / out_c;
    int w2 = in_w*stride + offset % stride;
    int h2 = in_h*stride + offset / stride;

    int out_index = w2 + w*stride*(h2 + h*stride*(c2 + out_c*b));

    if(forward)
        out[out_index] = x[in_index];
    else
        out[in_index] = x[out_index];
}


__kernel void axpy_kernel(int N, __const float ALPHA, __global float *X, int OFFX, int INCX,  __global float *Y, int OFFY, int INCY)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < N) Y[i*INCY+OFFY] += ALPHA*X[i*INCX+OFFX];
}


__kernel void pow_kernel(int N, __const float ALPHA, __global float *X, int OFFX, int INCX, __global float *Y, int OFFY, int INCY)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < N) Y[i*INCY + OFFY] = pow(X[i*INCX + OFFX], ALPHA);
}


__kernel void const_kernel(int N, __const float ALPHA, __global float *X, int OFFX, int INCX)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < N) X[i*INCX + OFFX] = ALPHA;
}


__kernel void constrain_kernel(int N, __const float ALPHA, __global float *X, int INCX)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < N) X[i*INCX] = min(ALPHA, max(-ALPHA, X[i*INCX]));
}


__kernel void supp_kernel(int N, __const float ALPHA, __global float *X, int INCX)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < N) {
        if((X[i*INCX] * X[i*INCX]) < (ALPHA * ALPHA)) X[i*INCX] = 0;
    }
}


__kernel void add_kernel(int N, __const float ALPHA, __global float *X, int INCX)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < N) X[i*INCX] += ALPHA;
}


__kernel void scal_kernel(int N, __const float ALPHA, __global float *X, int INCX)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < N) X[i*INCX] *= ALPHA;
}


__kernel void fill_kernel(int N, __const float ALPHA, __global float *X, int OFFX, int INCX)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < N) X[i*INCX + OFFX] = ALPHA;
}


__kernel void mask_kernel(int n,  __global float *x, float mask_num, __global float *mask, float val)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n && mask[i] == mask_num) x[i] = val;
}


__kernel void copy_kernel(int N,  __global float *X, int OFFX, int INCX, __global float *Y, int OFFY, int INCY)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < N) Y[i*INCY + OFFY] = X[i*INCX + OFFX];
}


__kernel void mul_kernel(int N, __global float *X, int INCX, __global float *Y, int INCY)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < N) Y[i*INCY] *= X[i*INCX];
}

);
#endif
