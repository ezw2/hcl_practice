#include "ihc_apint.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable
__kernel void default_function(__global float* restrict data, __global float* restrict label, __global float* restrict theta) {
  float _top;
  float theta_local[1024];
  float label_local[4500];
  for (int32_t i = 0; i < 1024; ++i) {
    theta_local[i] = theta[i];
  }
  for (int32_t i1 = 0; i1 < 4500; ++i1) {
    label_local[i1] = label[i1];
  }
  float EPOCH;
  for (int32_t m = 0; m < 5; ++m) {
    float TRAINING_INST;
    for (int32_t m1 = 0; m1 < 4500; ++m1) {
      float training_instance[1024];
      for (int32_t i2 = 0; i2 < 1024; ++i2) {
        training_instance[i2] = data[((m1 * 1024) + i2)];
      }
      float gradient[1024];
      float scalar8;
      for (int32_t x = 0; x < 1; ++x) {
        scalar8 = -6.000000e+04f;
      }
      float scalar9;
      scalar9 = 5.000000e-01f;
      float scalar10;
      for (int32_t x1 = 0; x1 < 1; ++x1) {
        scalar10 = label_local[m1];
      }
      float scalar11;
      for (int32_t x2 = 0; x2 < 1; ++x2) {
        scalar11 = (scalar9 - scalar10);
      }
      for (int32_t GRAD = 0; GRAD < 3.200000e+01f; ++GRAD) {
        for (int32_t GRAD_INNER = 0; GRAD_INNER < 32; ++GRAD_INNER) {
          gradient[((GRAD * 32) + GRAD_INNER)] = (scalar11 * training_instance[((GRAD * 32) + GRAD_INNER)]);
        }
      }
      for (int32_t UPDATE = 0; UPDATE < 3.200000e+01f; ++UPDATE) {
        for (int32_t UPDATE_INNER = 0; UPDATE_INNER < 32; ++UPDATE_INNER) {
          theta_local[((UPDATE * 32) + UPDATE_INNER)] = (theta_local[((UPDATE * 32) + UPDATE_INNER)] + (scalar8 * gradient[((UPDATE * 32) + UPDATE_INNER)]));
        }
      }
    }
  }
  for (int32_t i3 = 0; i3 < 1024; ++i3) {
    theta[i3] = theta_local[i3];
  }
}

