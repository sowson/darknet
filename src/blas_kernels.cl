#ifndef __BLAS_KERNELS_CL__
#define __BLAS_KERNELS_CL__

static const char* const blas_kernel_source = CONVERT_KERNEL_TO_STRING(

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

    mean_delta[i] *= (-1.f/sqrt(variance[i] + .0001f));
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
    variance_delta[i] *= -.5f * pow(variance[i] + .0001f, (float)(-3.f/2.f));
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

    mean_delta[i] = 0;
    for (s = 0; s < tuning; ++s) {
        mean_delta[i] += sums[s];
    }
    mean_delta[i] *= (-1.f/sqrt(variance[i] + .00001f));
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

    variance_delta[i] = 0;
    for (s = 0; s < tuning; ++s) {
        variance_delta[i] += sums[s];
    }
    variance_delta[i] *= -.5f * pow(variance[i] + .00001f, (float)(-3.f/2.f));
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
    delta[index] = delta[index] * 1.f/(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
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


__kernel void weighted_delta_kernel(int n, __global float *a, __global float *b, __global float *s, __global float *da, __global float *db, __global float *ds, __global float *dc)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n){
        if(da) da[i] += dc[i] * s[i];
        db[i] += dc[i] * (1-s[i]);
        ds[i] += dc[i] * a[i] + dc[i] * -b[i];
    }
}


__kernel void mult_add_into_kernel(int n, __global float *a, __global float *b, __global float *c)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n){
        c[i] += a[i]*b[i];
    }
}


__kernel void deinter_kernel(int NX, __global float *X, int NY, __global float *Y, int B, __global float *OUT)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            if(X) X[b*NX + j] += OUT[i];
        } else {
            if(Y) Y[b*NY + j - NX] += OUT[i];
        }
    }
}


__kernel void inter_kernel(int NX, __global float *X, int NY, __global float *Y, int B, __global float *OUT)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < (NX+NY)*B){
        int b = i / (NX+NY);
        int j = i % (NX+NY);
        if (j < NX){
            OUT[i] = X[b*NX + j];
        } else {
            OUT[i] = Y[b*NY + j - NX];
        }
    }
}


__kernel void softmax_device(__global float *input, int n, float temp, int stride, __global float *output)
{
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        int val = input[i*stride];
        largest = (val>largest) ? val : largest;
    }
    for(i = 0; i < n; ++i){
        float e = exp(input[i*stride]/temp - largest/temp);
        sum += e;
        output[i*stride] = e;
    }
    for(i = 0; i < n; ++i){
        output[i*stride] /= sum;
    }
}


__kernel void softmax_kernel(__global float *input, int offset, int n, int batch, int batch_offset, int groups, int group_offset, int stride, float temp, __global float *output)
{
    int id = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (id >= batch*groups) return;
    int b = id / groups;
    int g = id % groups;
    softmax_device(input + b*batch_offset + g*group_offset + offset, n, temp, stride, output + b*batch_offset + g*group_offset + offset);
}


__kernel void softmax_tree_kernel(__global float *input, int offset, int index, int spatial, int batch, int stride, float temp, __global float *output, int groups, __global float *group_size, __global float *group_offset)
{
    int id = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (id >= spatial*batch*groups) return;
    int s = id % spatial;
    id = id / spatial;
    int g = id % groups;
    int b = id / groups;
    int goff = group_offset[g]*spatial;
    int boff = b*stride;
    softmax_device(input + offset + goff + boff + s, group_size[g], temp, spatial, output + offset + goff + boff + s);
}


__kernel void scale_mask_kernel(int n, __global float *x, float mask_num, __global float *mask, float scale)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n && mask[i] == mask_num) x[i] *= scale;
}


__kernel void dot_kernel(__global float *output, float scale, int batch, int n, int size, __global float *delta)
{
    int index = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);

    int f1 = index / n;
    int f2 = index % n;
    if (f2 <= f1) return;

    float sum = 0;
    float norm1 = 0;
    float norm2 = 0;
    int b, i;
    for(b = 0; b <  batch; ++b){
        for(i = 0; i < size; ++i){
            int i1 = b * size * n + f1 * size + i;
            int i2 = b * size * n + f2 * size + i;
            sum += output[i1] * output[i2];
            norm1 += output[i1] * output[i1];
            norm2 += output[i2] * output[i2];
        }
    }
    norm1 = sqrt(fabs(norm1));
    norm2 = sqrt(fabs(norm2));
    float norm = norm1 * norm2;
    sum = sum / norm;
    for(b = 0; b <  batch; ++b){
        for(i = 0; i < size; ++i){
            int i1 = b * size * n + f1 * size + i;
            int i2 = b * size * n + f2 * size + i;
            delta[i1] += - scale * sum * output[i2] / norm;
            delta[i2] += - scale * sum * output[i1] / norm;
        }
    }
}


__kernel void upsample_kernel(int N, __global float *x, int w, int h, int c, int batch, int stride, int forward, float scale, __global float *out)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i >= N) return;
    int out_index = i;
    int out_w = i%(w*stride);
    i = i/(w*stride);
    int out_h = i%(h*stride);
    i = i/(h*stride);
    int out_c = i%c;
    i = i/c;
    int b = i%batch;

    int in_w = out_w / stride;
    int in_h = out_h / stride;
    int in_c = out_c;

    int in_index = b*w*h*c + in_c*w*h + in_h*w + in_w;

    //if(forward) out[out_index] += scale * x[in_index];
    //else atomicAdd(&x[in_index], scale * out[out_index]);
    if(forward) out[out_index] += scale * x[in_index];
    else x[in_index] += scale * out[out_index];
}


__kernel void gemm_kernel(
        int tuning, __local float* sums,
        int TA, int TB,
        int M, int N, int K,
        __const float ALPHA,
        __global float *A, int offset_A, int lda,
        __global float *B, int offset_B, int ldb,
        __const float BETA,
        __global float *C, int offset_C, int ldc) {

    int td = get_global_id(0);
    if (td > tuning) return;
    int id = get_global_id(1);
    if (id > N*M) return;

    int iM = id / N;
    int jN = id % N;
    int kK = 0;
    int ts = 0;

    C[iM * ldc + jN + offset_C] *= BETA;

    sums[td] = 0;

    for(kK = td; kK < K; kK += tuning) {
        if (TA==0 && TB==0) {
            sums[td] += ALPHA * A[iM * lda + kK + offset_A] * B[kK * ldb + jN + offset_B];
        } else if (TA==1 && TB==0) {
            sums[td] += ALPHA * A[kK * lda + iM + offset_A] * B[kK * ldb + jN + offset_B];
        } else if (TA==0 && TB==1) {
            sums[td] += ALPHA * A[iM * lda + kK + offset_A] * B[jN * ldb + kK + offset_B];
        } else {
            sums[td] += ALPHA * A[iM + kK * lda + offset_A] * B[kK + jN * ldb + offset_B];
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    if (td == tuning-1) {
        for(ts = 0; ts < tuning; ++ts) {
            if (TA==0 && TB==0) {
                C[iM * ldc + jN + offset_C] += sums[ts];
            } else if (TA==1 && TB==0) {
                C[iM * ldc + jN + offset_C] += sums[ts];
            } else if (TA==0 && TB==1) {
                C[iM * ldc + jN + offset_C] += sums[ts];
            } else {
                C[iM * ldc + jN + offset_C] += sums[ts];
            }
        }
    }
}


__kernel void scal_add_kernel(int N, float ALPHA, float BETA, __global float *X, int OFFX, int INCX)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (i < N) {
        X[i*INCX + OFFX] *= ALPHA;
        X[i*INCX + OFFX] += BETA;
    }
}

__kernel void mean_array_kernel(int N, float alpha, __global float *s, __global float *a)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if (i >= N) return;
    a[i] *= (1 - alpha) + s[i];
    a[i] *= alpha;
    s[i] = a[i];
}

);
#endif
