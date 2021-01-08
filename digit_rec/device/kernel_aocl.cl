#include "ihc_apint.h"
#pragma OPENCL EXTENSION cl_intel_channels : enable
__kernel void default_function(__global int32_t* restrict training_set, __global int32_t* restrict test_set, __global int32_t* restrict result) {
  int32_t _top;
  int32_t temp_result[2000];
  for (int32_t i = 0; i < 2000; ++i) {
    int32_t dists[3];
    int32_t labels[3];
    int32_t data[64];
    for (int32_t i1 = 0; i1 < 64; ++i1) {
      data[i] = test_set[((i * 64) + i1)];
    }
    for (int32_t i2 = 0; i2 < 18000; ++i2) {
      int32_t data_[64];
      for (int32_t i3 = 0; i3 < 64; ++i3) {
        data_[i2] = training_set[((i2 * 64) + i3)];
      }
      int32_t label;
      int32_t diff[64];
      for (int32_t x = 0; x < 64; ++x) {
        diff[x] = (data[x] ^ data_[x]);
      }
      int32_t dist;
      for (int32_t x1 = 0; x1 < 1; ++x1) {
        int32_t count;
        for (int32_t i4 = 0; i4 < 64; ++i4) {
          count = (count + diff[i4]);
        }
        dist = count;
      }
      int32_t scalar3;
      int32_t scalar4;
      for (int32_t x2 = 0; x2 < 1; ++x2) {
        scalar4 = 4;
      }
      for (int32_t i5 = 0; i5 < 3; ++i5) {
        if (scalar3 < dists[i5]) {
          scalar3 = dists[i5];
          scalar4 = i5;
        }
      }
      if (dist < scalar3) {
        dists[scalar4] = dist;
        labels[scalar4] = label;
      }
    }
    int32_t max_label;
    int32_t scalar5;
    int32_t votes[10];
    for (int32_t i6 = 0; i6 < 3; ++i6) {
      votes[labels[i6]] = (votes[labels[i6]] + 1);
    }
    for (int32_t i7 = 0; i7 < 10; ++i7) {
      if (scalar5 < votes[i7]) {
        scalar5 = votes[i7];
        max_label = i7;
      }
    }
  }
}

