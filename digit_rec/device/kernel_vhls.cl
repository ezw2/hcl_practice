#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>
void default_function(float data[4608000], float label[4500], float theta[1024]) {
  float _top;
  float theta_local[1024];
  theta_local_x: for (ap_int<32> x = 0; x < 1024; ++x) {
    theta_local[x] = 0.000000e+00f;
  }
  float label_local[4500];
  label_local_x1: for (ap_int<32> x1 = 0; x1 < 4500; ++x1) {
    label_local[x1] = 0.000000e+00f;
  }
  i: for (ap_int<32> i = 0; i < 1024; ++i) {
    theta_local[i] = theta[i];
  }
  i1: for (ap_int<32> i1 = 0; i1 < 4500; ++i1) {
    label_local[i1] = label[i1];
  }
  float EPOCH;
  EPOCH_m: for (ap_int<32> m = 0; m < 5; ++m) {
    float TRAINING_INST;
    TRAINING_INST_m1: for (ap_int<32> m1 = 0; m1 < 4500; ++m1) {
      float training_instance[1024];
      training_instance_x2: for (ap_int<32> x2 = 0; x2 < 1024; ++x2) {
        training_instance[x2] = 0.000000e+00f;
      }
      i2: for (ap_int<32> i2 = 0; i2 < 1024; ++i2) {
        training_instance[i2] = data[((m1 * 1024) + i2)];
      }
      float gradient[1024];
      gradient_x3: for (ap_int<32> x3 = 0; x3 < 1024; ++x3) {
        gradient[x3] = 0.000000e+00f;
      }
      float scalar4;
      scalar4_x4: for (ap_int<32> x4 = 0; x4 < 1; ++x4) {
        scalar4 = -6.000000e+04f;
      }
      float scalar5;
      scalar5_x5: for (ap_int<32> x5 = 0; x5 < 1; ++x5) {
        scalar5 = 0.000000e+00f;
      }
      scalar5 = 5.000000e-01f;
      float scalar6;
      scalar6_x6: for (ap_int<32> x6 = 0; x6 < 1; ++x6) {
        scalar6 = label_local[m1];
      }
      float scalar7;
      scalar7_x7: for (ap_int<32> x7 = 0; x7 < 1; ++x7) {
        scalar7 = (scalar5 - scalar6);
      }
      GRAD: for (ap_int<32> GRAD = 0; GRAD < 3.200000e+01f; ++GRAD) {
        GRAD_INNER: for (ap_int<32> GRAD_INNER = 0; GRAD_INNER < 32; ++GRAD_INNER) {
          gradient[((GRAD * 32) + GRAD_INNER)] = (scalar7 * training_instance[((GRAD * 32) + GRAD_INNER)]);
        }
      }
      UPDATE: for (ap_int<32> UPDATE = 0; UPDATE < 3.200000e+01f; ++UPDATE) {
        UPDATE_INNER: for (ap_int<32> UPDATE_INNER = 0; UPDATE_INNER < 32; ++UPDATE_INNER) {
          theta_local[((UPDATE * 32) + UPDATE_INNER)] = (theta_local[((UPDATE * 32) + UPDATE_INNER)] + (scalar4 * gradient[((UPDATE * 32) + UPDATE_INNER)]));
        }
      }
    }
  }
  i3: for (ap_int<32> i3 = 0; i3 < 1024; ++i3) {
    theta[i3] = theta_local[i3];
  }
}

