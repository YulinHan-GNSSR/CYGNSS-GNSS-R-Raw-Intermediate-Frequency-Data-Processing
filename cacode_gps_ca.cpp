#include "cacode_gps_ca.h"

bool generateGpsCaCode(int prn, std::array<float, 1023>& code_out) {
    // MATLAB cacode.m mapping for PRN 1..32 (G2 tap pairs, 1-based indexing).
    static const int g2_taps[32][2] = {
        {2, 6}, {3, 7}, {4, 8}, {5, 9}, {1, 9}, {2, 10}, {1, 8}, {2, 9},
        {3, 10}, {2, 3}, {3, 4}, {5, 6}, {6, 7}, {7, 8}, {8, 9}, {9, 10},
        {1, 4}, {2, 5}, {3, 6}, {4, 7}, {5, 8}, {6, 9}, {1, 3}, {4, 6},
        {5, 7}, {6, 8}, {7, 9}, {8, 10}, {1, 6}, {2, 7}, {3, 8}, {4, 9}
    };

    if (prn < 1 || prn > 32) {
        return false;
    }

    const int a = g2_taps[prn - 1][0] - 1; // to 0-based
    const int b = g2_taps[prn - 1][1] - 1; // to 0-based

    int x[10];
    int y[10];
    for (int i = 0; i < 10; ++i) {
        x[i] = -1;
        y[i] = -1;
    }

    for (int j = 0; j < 1023; ++j) {
        const int g1 = x[9];
        const int g2 = y[a] * y[b];
        code_out[j] = static_cast<float>(g1 * g2);

        const int x1 = x[2] * x[9];
        const int y1 = y[1] * y[2] * y[5] * y[7] * y[8] * y[9];

        for (int k = 9; k >= 1; --k) {
            x[k] = x[k - 1];
            y[k] = y[k - 1];
        }
        x[0] = x1;
        y[0] = y1;
    }

    return true;
}

