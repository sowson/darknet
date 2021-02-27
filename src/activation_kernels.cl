#ifndef __ACTIVATION_KERNELS_CL__
#define __ACTIVATION_KERNELS_CL__

static const char* const activation_kernels_source = CONVERT_KERNEL_TO_STRING(

float lhtan_activate_kernel(float x);
float lhtan_gradient_kernel(float x);
float hardtan_activate_kernel(float x);
float linear_activate_kernel(float x);
float logistic_activate_kernel(float x);
float loggy_activate_kernel(float x);
float relu_activate_kernel(float x);
float elu_activate_kernel(float x);
float selu_activate_kernel(float x);
float relie_activate_kernel(float x);
float ramp_activate_kernel(float x);
float leaky_activate_kernel(float x);
float tanh_activate_kernel(float x);
float plse_activate_kernel(float x);
float stair_activate_kernel(float x);
float mish_activate_kernel(float x);

float hardtan_gradient_kernel(float x);
float linear_gradient_kernel(float x);
float logistic_gradient_kernel(float x);
float loggy_gradient_kernel(float x);
float relu_gradient_kernel(float x);
float elu_gradient_kernel(float x);
float selu_gradient_kernel(float x);
float relie_gradient_kernel(float x);
float ramp_gradient_kernel(float x);
float leaky_gradient_kernel(float x);
float tanh_gradient_kernel(float x);
float plse_gradient_kernel(float x);
float stair_gradient_kernel(float x);
float mish_gradient_kernel(float x);

float softplus(float x);

typedef enum{
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU, MISH
} ACTIVATION;

float activate_kernel(float x, ACTIVATION a);
float gradient_kernel(float x, ACTIVATION a);

float lhtan_activate_kernel(float x)
{
    if(x < 0) return .001f*x;
    if(x > 1) return .001f*(x-1.f) + 1.f;
    return x;
}
float lhtan_gradient_kernel(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}

float linear_activate_kernel(float x){return x;}
float logistic_activate_kernel(float x){return 1.f/(1.f + exp(-x));}
float loggy_activate_kernel(float x){return 2.f/(1.f + exp(-x)) - 1;}
float relu_activate_kernel(float x){return x*(x>0);}
float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
float selu_activate_kernel(float x){return (x >= 0)*1.0507f*x + (x < 0)*1.0507f*1.6732f*(exp(x)-1);}
float relie_activate_kernel(float x){return (x>0) ? x : .01f*x;}
float ramp_activate_kernel(float x){return x*(x>0)+.1f*x;}
float leaky_activate_kernel(float x){return (x>0) ? x : .1f*x;}
float tanh_activate_kernel(float x){return (2.f/(1 + exp(-2*x)) - 1);}

float plse_activate_kernel(float x)
{
    if(x < -4) return .01f * (x + 4);
    if(x > 4)  return .01f * (x - 4) + 1;
    return .125f*x + .5f;
}

float stair_activate_kernel(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2);
    else return (x - n) + floor(x/2);
}

float hardtan_gradient_kernel(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}

float linear_gradient_kernel(float x){return 1;}
float logistic_gradient_kernel(float x){return (1-x)*x;}

float loggy_gradient_kernel(float x)
{
    float y = (x+1)/2;
    return 2*(1-y)*y;
}

float relu_gradient_kernel(float x){return (x>0);}
float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}
float selu_gradient_kernel(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
float relie_gradient_kernel(float x){return (x>0) ? 1 : .01f;}
float ramp_gradient_kernel(float x){return (x>0)+.1f;}
float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1f;}
float tanh_gradient_kernel(float x){return 1-x*x;}
float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01f : .125f;}

float stair_gradient_kernel(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}

float softplus(float x) {
    float t = 27;
    if (x >  t) return x;
    if (x < -t) return exp(x);
    return log1p(exp(x));
}

float mish_activate_kernel(float x) {
    //https://arxiv.org/abs/1908.08681v1
    float c = softplus(x);
    float a = x * tanh(c);
    return a;
}

float mish_gradient_kernel(float x) {
    //https://arxiv.org/abs/1908.08681v1
    //float d = 2*exp(2*x) + exp(2*x) + 2;
    //float w = 4*(x+1) + 4*exp(2*x) + exp(3*x) + exp(x*(4*x+6));
    //float g = exp(x) * w / pow(d,2);
    //return g;
    float sp = softplus(x);
    float g_sp = -expm1(-sp);
    float tsp = tanh(sp);
    float g_tsp = (1 - tsp*tsp) * g_sp;
    float g = x * g_tsp / tsp;
    return g;
}

float activate_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_activate_kernel(x);
        case LOGISTIC:
            return logistic_activate_kernel(x);
        case LOGGY:
            return loggy_activate_kernel(x);
        case RELU:
            return relu_activate_kernel(x);
        case ELU:
            return elu_activate_kernel(x);
        case SELU:
            return selu_activate_kernel(x);
        case RELIE:
            return relie_activate_kernel(x);
        case RAMP:
            return ramp_activate_kernel(x);
        case LEAKY:
            return leaky_activate_kernel(x);
        case TANH:
            return tanh_activate_kernel(x);
        case PLSE:
            return plse_activate_kernel(x);
        case STAIR:
            return stair_activate_kernel(x);
        case HARDTAN:
            return hardtan_activate_kernel(x);
        case LHTAN:
            return lhtan_activate_kernel(x);
        case MISH:
            return mish_activate_kernel(x);
        default: relu_activate_kernel(x);
    }
    return 0;
}

float gradient_kernel(float x, ACTIVATION a)
{
    switch(a){
        case LINEAR:
            return linear_gradient_kernel(x);
        case LOGISTIC:
            return logistic_gradient_kernel(x);
        case LOGGY:
            return loggy_gradient_kernel(x);
        case RELU:
            return relu_gradient_kernel(x);
        case ELU:
            return elu_gradient_kernel(x);
        case SELU:
            return selu_gradient_kernel(x);
        case RELIE:
            return relie_gradient_kernel(x);
        case RAMP:
            return ramp_gradient_kernel(x);
        case LEAKY:
            return leaky_gradient_kernel(x);
        case TANH:
            return tanh_gradient_kernel(x);
        case PLSE:
            return plse_gradient_kernel(x);
        case STAIR:
            return stair_gradient_kernel(x);
        case HARDTAN:
            return hardtan_gradient_kernel(x);
        case LHTAN:
            return lhtan_gradient_kernel(x);
        case MISH:
            return mish_gradient_kernel(x);
        default: relu_gradient_kernel(x);
    }
    return 0;
}

__kernel void activate_array_kernel(__global float *x, int offset, int n, ACTIVATION a)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n) x[i + offset] = activate_kernel(x[i + offset], a);
}

__kernel void gradient_array_kernel(__global float *x, int offset, int n, ACTIVATION a, __global float *delta)
{
    int i = (get_group_id(0) + get_group_id(1)*get_num_groups(0)) * get_local_size(0) + get_local_id(0);
    if(i < n) delta[i + offset] *= gradient_kernel(x[i + offset], a);
}

);

#endif