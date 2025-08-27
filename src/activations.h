#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "darknet.h"
#include "opencl.h"
#include "math.h"

ACTIVATION get_activation(char *s);

char *get_activation_string(ACTIVATION a);
float activate(float x, ACTIVATION a);
float gradient(float x, ACTIVATION a);

void gradient_array(const float *x, const int n, const ACTIVATION a, float *delta);
void activate_array(float *x, const int n, const ACTIVATION a);
#ifdef GPU
void activate_array_gpu(cl_mem_ext x, int n, ACTIVATION a);
void activate_array_offset_gpu(cl_mem_ext x, int offset, int n, ACTIVATION a);
void gradient_array_gpu(cl_mem_ext x, int n, ACTIVATION a, cl_mem_ext delta);
void gradient_array_offset_gpu(cl_mem_ext x, int offset, int n, ACTIVATION a, cl_mem_ext delta);
#endif

static inline float stair_activate(float x)
{
	int n = floorf(x);
	if (n%2 == 0) return floor(x/2.);
	else return (x - n) + floor(x/2.);
}

static inline float hardtan_activate(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}
static inline float linear_activate(float x){return x;}
static inline float logistic_activate(float x){return 1./(1. + expf(-x));}
static inline float loggy_activate(float x){return 2./(1. + expf(-x)) - 1;}
static inline float relu_activate(float x){return x*(x>0);}
static inline float relu10_activate(float x){return x <= -1. ? -1. : x >= 1. ? 1. : x;}
static inline float lelu_activate(float x){return 10.f * tanhf(x);}
static inline float elu_activate(float x){return (x >= 0)*x + (x < 0)*(expf(x)-1);}
static inline float selu_activate(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(expf(x)-1);}
static inline float relie_activate(float x){return (x>0) ? x : .01*x;}
static inline float ramp_activate(float x){return x*(x>0)+.1*x;}
static inline float leaky_activate(float x){return (x>0) ? x : .1*x;}
static inline float tanh_activate(float x){return tanhf(x);}
static inline float plse_activate(float x)
{
	if(x < -4) return .01 * (x + 4);
	if(x > 4)  return .01 * (x - 4) + 1;
	return .125*x + .5;
}

static inline float softplus(float x) {
    float t = 27;
    if (x >  t) return x;
    if (x < -t) return expf(x);
    return log1pf(expf(x));
}

static inline float mish_activate(float x) {
    //https://arxiv.org/abs/1908.08681v1
    // (𝑥) = 𝑥 ⋅ 𝑡𝑎𝑛h(𝜍(𝑥))
    // 𝜍(𝑥) = ln(1 + 𝑒𝑥)
    float c = softplus(x);
    float a = x * tanhf(c);
    return a;
}

static inline float mish_gradient(float x) {
    //https://arxiv.org/abs/1908.08681v1
    //𝑓’(𝑥) = ex ⋅ 𝜔 / 𝛿^2
    //𝜔 = 4(𝑥+1)+4𝑒2𝑥 + 𝑒3𝑥 + 𝑒𝑥(4𝑥+6)
    //𝛿 = 2𝑒2𝑥 + 𝑒2𝑥 + 2
    //float d = 2 * expf(2*x) + expf(2*x) + 2;
    //float w = 4 * (x+1) + 4*expf(2*x) + expf(3*x) + expf(x*(4*x+6));
    //float g = expf(x) * w / powf(d,2);
    //return g;
    float sp = softplus(x);
    float g_sp = -expm1f(-sp);
    float tsp = tanhf(sp);
    float g_tsp = (1 - tsp*tsp) * g_sp;
    float g = x * g_tsp / tsp;
    return g;
}

static inline float lhtan_activate(float x)
{
	if(x < 0) return .001*x;
	if(x > 1) return .001*(x-1) + 1;
	return x;
}
static inline float lhtan_gradient(float x)
{
	if(x > 0 && x < 1) return 1;
	return .001;
}

static inline float hardtan_gradient(float x)
{
	if (x > -1 && x < 1) return 1;
	return 0;
}
static inline float linear_gradient(float x){return 1;}
static inline float logistic_gradient(float x){return (1-x)*x;}
static inline float loggy_gradient(float x)
{
	float y = (x+1.)/2.;
	return 2*(1-y)*y;
}
static inline float stair_gradient(float x)
{
	if (floorf(x) == x) return 0;
	return 1;
}
static inline float relu_gradient(float x){return (x>0);}
static inline float relu10_gradient(float x){return (x > -1. && x < 1.) ? 1. : 0.;}
static inline float lelu_gradient(float x){float c = coshf(x); return 10.f / (powf(c, 2.f));}
static inline float elu_gradient(float x){return (x >= 0) + (x < 0)*(x + 1);}
static inline float selu_gradient(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
static inline float relie_gradient(float x){return (x>0) ? 1 : .01;}
static inline float ramp_gradient(float x){return (x>0)+.1;}
static inline float leaky_gradient(float x){return (x>0) ? 1 : .1;}
static inline float tanh_gradient(float x){return 1.f-powf(tanhf(x),2.f);}
static inline float plse_gradient(float x){return (x < 0 || x > 1) ? .01 : .125;}

#endif

