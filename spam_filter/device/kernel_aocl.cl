#include "ihc_apint.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable
__kernel void default_function(__global float* restrict lut, __global float* restrict data, __global float* restrict label, __global float* restrict theta) {
  float _top;
  for (int32_t epoch_loop = 0; epoch_loop < 5; ++epoch_loop) {
    for (int32_t training_loop = 0; training_loop < 4500; ++training_loop) {
      float dot_product[1024];
      for (int32_t x = 0; x < 1024; ++x) {
        dot_product[x] = data[((training_loop * 1024) + x)];
      }
      float dot;
      for (int32_t dot_product_loop = 0; dot_product_loop < 1024; ++dot_product_loop) {
        dot = (dot + (dot_product[dot_product_loop] * theta[dot_product_loop]));
      }
      float grad[1024];
      float grad_[1024];
      for (int32_t x1 = 0; x1 < 1024; ++x1) {
        grad_[x1] = ((float)(((1.000000e+00 / (exp(((double)(dot * -1.000000e+00f))) + 1.000000e+00)) - ((double)label[training_loop])) * ((double)dot_product[x1])));
      }
      for (int32_t update_parameter_loop = 0; update_parameter_loop < 1024; ++update_parameter_loop) {
        theta[update_parameter_loop] = (theta[update_parameter_loop] - (grad[update_parameter_loop] * 6.000000e+04f));
      }
    }
  }
}

