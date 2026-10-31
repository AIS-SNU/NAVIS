// Check if CPU supports AVX512 and AVX2 instructions

#include <iostream>
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif

int has_avx512() {
  int info[4];
#if defined(_MSC_VER)
  __cpuidex(info, 7, 0);
#else
  __cpuid_count(7, 0, info[0], info[1], info[2], info[3]);
#endif
  int support_avx512 = (info[1] & (1 << 16)) != 0;  // AVX512F bit
  int support_avx2 = (info[1] & (1 << 5)) != 0;     // AVX2 bit
  return support_avx512 << 1 | support_avx2;
}

int main() {
  return has_avx512();
}