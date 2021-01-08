#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>
void default_function(float lut[2048], float data[4608000], float label[4500], float theta[1024]) {
  float _top;
  epoch_loop: for (ap_int<32> epoch_loop = 0; epoch_loop < 5; ++epoch_loop) {
    training_loop: for (ap_int<32> training_loop = 0; training_loop < 4500; ++training_loop) {
      float dot_product[1024];
      dot_product_x: for (ap_int<32> x = 0; x < 1024; ++x) {
        dot_product[x] = data[((training_loop * 1024) + x)];
      }
      float dot;
      dot_x1: for (ap_int<32> x1 = 0; x1 < 1; ++x1) {
        dot = 0.000000e+00f;
      }
      dot_product_loop: for (ap_int<32> dot_product_loop = 0; dot_product_loop < 1024; ++dot_product_loop) {
        dot = (dot + (dot_product[dot_product_loop] * theta[dot_product_loop]));
      }
      float grad[1024];
      grad_x2: for (ap_int<32> x2 = 0; x2 < 1024; ++x2) {
        grad[x2] = 0.000000e+00f;
      }
      float grad_[1024];
      grad_x3: for (ap_int<32> x3 = 0; x3 < 1024; ++x3) {
        grad_[x3] = ((float)(((1.000000e+00 / (sqrt(((double)(dot * -1.000000e+00f))) + 1.000000e+00)) - ((double)label[training_loop])) * ((double)dot_product[x3])));
      }
      update_parameter_loop: for (ap_int<32> update_parameter_loop = 0; update_parameter_loop < 1024; ++update_parameter_loop) {
        theta[update_parameter_loop] = (theta[update_parameter_loop] - (grad[update_parameter_loop] * 6.000000e+04f));
      }
    }
  }
}

