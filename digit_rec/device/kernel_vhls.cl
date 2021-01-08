#include <ap_int.h>
#include <ap_fixed.h>
#include <ap_axi_sdata.h>
#include <hls_stream.h>
#include <hls_math.h>
#include <math.h>
#include <stdint.h>
void default_function(ap_int<32> training_set[1152000], ap_int<32> test_set[128000], ap_int<32> result[2000]) {
  ap_int<32> _top;
  ap_int<32> temp_result[2000];
  temp_result_x: for (ap_int<32> x = 0; x < 2000; ++x) {
    temp_result[x] = 0;
  }
  i: for (ap_int<32> i = 0; i < 2000; ++i) {
    ap_int<32> dists[3];
    dists_x1: for (ap_int<32> x1 = 0; x1 < 3; ++x1) {
      dists[x1] = 0;
    }
    ap_int<32> labels[3];
    labels_x2: for (ap_int<32> x2 = 0; x2 < 3; ++x2) {
      labels[x2] = 0;
    }
    ap_int<32> data[64];
    data_x3: for (ap_int<32> x3 = 0; x3 < 64; ++x3) {
      data[x3] = 0;
    }
    i1: for (ap_int<32> i1 = 0; i1 < 64; ++i1) {
      data[i] = test_set[((i * 64) + i1)];
    }
    i2: for (ap_int<32> i2 = 0; i2 < 18000; ++i2) {
      ap_int<32> data_[64];
      data_x4: for (ap_int<32> x4 = 0; x4 < 64; ++x4) {
        data_[x4] = 0;
      }
      i3: for (ap_int<32> i3 = 0; i3 < 64; ++i3) {
        data_[i2] = training_set[((i2 * 64) + i3)];
      }
      ap_int<32> label;
      label_x5: for (ap_int<32> x5 = 0; x5 < 1; ++x5) {
        label = 0;
      }
      ap_int<32> diff[64];
      diff_x6: for (ap_int<32> x6 = 0; x6 < 64; ++x6) {
        diff[x6] = (data[x6] ^ data_[x6]);
      }
      ap_int<32> dist;
      dist_x7: for (ap_int<32> x7 = 0; x7 < 1; ++x7) {
        ap_int<32> count;
        count_x8: for (ap_int<32> x8 = 0; x8 < 1; ++x8) {
          count = 0;
        }
        i4: for (ap_int<32> i4 = 0; i4 < 64; ++i4) {
          count = (count + diff[i4]);
        }
        dist = count;
      }
      ap_int<32> scalar0;
      scalar0_x9: for (ap_int<32> x9 = 0; x9 < 1; ++x9) {
        scalar0 = 0;
      }
      ap_int<32> scalar1;
      scalar1_x10: for (ap_int<32> x10 = 0; x10 < 1; ++x10) {
        scalar1 = 4;
      }
      i5: for (ap_int<32> i5 = 0; i5 < 3; ++i5) {
        if (scalar0 < dists[i5]) {
          scalar0 = dists[i5];
          scalar1 = i5;
        }
      }
      if (dist < scalar0) {
        dists[scalar1] = dist;
        labels[scalar1] = label;
      }
    }
    ap_int<32> max_label;
    max_label_x11: for (ap_int<32> x11 = 0; x11 < 1; ++x11) {
      max_label = 0;
    }
    ap_int<32> scalar2;
    scalar2_x12: for (ap_int<32> x12 = 0; x12 < 1; ++x12) {
      scalar2 = 0;
    }
    ap_int<32> votes[10];
    votes_x13: for (ap_int<32> x13 = 0; x13 < 10; ++x13) {
      votes[x13] = 0;
    }
    i6: for (ap_int<32> i6 = 0; i6 < 3; ++i6) {
      votes[labels[i6]] = (votes[labels[i6]] + 1);
    }
    i7: for (ap_int<32> i7 = 0; i7 < 10; ++i7) {
      if (scalar2 < votes[i7]) {
        scalar2 = votes[i7];
        max_label = i7;
      }
    }
  }
}

