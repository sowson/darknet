#ifndef __BLAS_KERNELS_2_CL__
#define __BLAS_KERNELS_2_CL__

static const char* const blas_kernel_source_2 = CONVERT_KERNEL_TO_STRING(

__kernel void flatten_kernel(int N, __global float *x, int spatial, int layers, int batch, int forward, __global float *out)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i >= N) return;
    int in_s = i%spatial;
    i = i/spatial;
    int in_c = i%layers;
    i = i/layers;
    int b = i;

    int i1 = b*layers*spatial + in_c*spatial + in_s;
    int i2 = b*layers*spatial + in_s*layers +  in_c;

    if (forward)
        out[i2] = x[i1];
    else
        out[i1] = x[i2];
}


__kernel void shortcut_kernel(int size, int minw, int minh, int minc, int stride, int sample, int batch, int w1, int h1, int c1, __global float *add, int w2, int h2, int c2, float s1, float s2, __global float *out)
{
    int id = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (id >= size) return;
    int i = id % minw;
    id /= minw;
    int j = id % minh;
    id /= minh;
    int k = id % minc;
    id /= minc;
    int b = id % batch;

    int out_index = i*sample + w2*(j*sample + h2*(k + c2*b));
    int add_index = i*stride + w1*(j*stride + h1*(k + c1*b));
    //out[out_index] += add[add_index];
    out[out_index] = s1*out[out_index] + s2*add[add_index];
}


__kernel void smooth_l1_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n){
        float diff = truth[i] - pred[i];
        float abs_val = fabs(diff);
        if(abs_val < 1) {
            error[i] = diff * diff;
            delta[i] = diff;
        }
        else {
            error[i] = 2*abs_val - 1;
            delta[i] = (diff > 0) ? 1 : -1;
        }
    }
}


__kernel void softmax_x_ent_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n) {
        float t = truth[i];
        float p = pred[i];
        error[i] = (t!=0) ? -log(p) : 0;
        delta[i] = t-p;
    }
}


__kernel void logistic_x_ent_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        error[i] = -t*log(p+.0000001f) - (1.f-t)*log(1.f-p+.0000001f);
        delta[i] = t-p;
    }
}


__kernel void l2_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n){
        float t = truth[i];
        float p = pred[i];
        float diff = t-p;
        error[i] = pow(diff,2);
        delta[i] = diff;
    }
}


__kernel void l1_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n){
        float diff = truth[i] - pred[i];
        error[i] = fabs(diff);
        delta[i] = (diff > 0) ? 1 : -1;
    }
}


__kernel void wgan_kernel(int n, __global float *pred, __global float *truth, __global float *delta, __global float *error)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n){
        error[i] = (truth[i]!=0) ? -pred[i] : pred[i];
        delta[i] = (truth[i] > 0) ? 1 : -1;
    }
}


__kernel void weighted_sum_kernel(int n, __global float *a, __global float *b, __global float *s, __global float *c)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n){
        c[i] = s[i]*a[i] + (1-s[i])*(b ? b[i] : 0);
    }
}

);
#endif
