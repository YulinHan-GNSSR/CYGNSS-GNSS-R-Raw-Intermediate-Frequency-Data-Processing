#pragma once

#include <array>

// Generate one full GPS L1 C/A code period (1023 chips) with values in {-1,+1}.
// Returns false when PRN is out of supported range [1,32].
bool generateGpsCaCode(int prn, std::array<float, 1023>& code_out);

