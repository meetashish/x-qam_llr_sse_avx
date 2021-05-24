/// @author Ashish Meshram
/// @brief Computes Log-Likelihood Ratio (LLR) for 64-QAM
///

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include <tmmintrin.h> // SSSE3
#include <emmintrin.h> // SSE2
#include <smmintrin.h> // SSE4.1

#include <immintrin.h> // AVX

// #define debug_sse
// #define debug_avx
#define debug_sse_llr
#define debug_avx_llr

/// @brief Utility function for display sse, avx-2, and avx512 data types
void PrintIntrinsics(char *s, char *dtype, int num, void *x);

int main()
{
  clock_t start, end;
  double sse_cpu_time, avx_cpu_time;
  // Compensated received symbol
  int16_t rxFcomp[] __attribute__((aligned(32))) = {62, -63, 62, 19, 62, 60, -22, -60,
                                                    -61, -59, -61, -60, -61, -61, 61, 60,
                                                    20, -21, 22, 59, -59, 61, 60, 18,
                                                    19, -21, 18, -61, -21, -20, 21, 58};

  // First scaled channel magnitude = 4\sqrt(42)||h||^2
  int16_t dlchmag[] __attribute__((aligned(32))) = {42, 42, 40, 40, 40, 40, 38, 38,
                                                    38, 38, 38, 38, 40, 40, 40, 40,
                                                    40, 40, 38, 38, 38, 38, 38, 38,
                                                    38, 38, 38, 38, 38, 38, 36, 36};

  // llr
  int32_t llr32_sse[32] = {[0 ... 31] = 0}, // for sse
      llr32_avx[32] = {[0 ... 31] = 0};     // for avx
  int32_t *llr32sse = llr32_sse, *llr32avx = llr32_avx;

  start = clock();
  /// ------------------------------------- SSE -------------------------------------
  printf("============================ SSE ===============================\n");
  __m128i *rxFcomp128 = (__m128i *)&rxFcomp;
  __m128i *dlchmag128 = (__m128i *)&dlchmag;
  __m128i xmm1, llr128[2];

  for (size_t i = 0; i < 4; i++)
  {
#ifdef debug_sse
    PrintIntrinsics("          rxF", "short", 128, &rxFcomp128[i]);
#endif
    xmm1 = _mm_abs_epi16(rxFcomp128[i]);
#ifdef debug_sse
    PrintIntrinsics("        |rxF|", "short", 128, &xmm1);
#endif
    xmm1 = _mm_subs_epi16(dlchmag128[i], xmm1);
#ifdef debug_sse
    PrintIntrinsics("       ch_mag", "short", 128, &dlchmag128[i]);
    PrintIntrinsics("chmag - |rxF|", "short", 128, &xmm1);
#endif
    llr128[0] = _mm_unpacklo_epi32(rxFcomp128[i], xmm1);
#ifdef debug_sse
    PrintIntrinsics("Lower of (chmag - |rxF|)", "short", 128, &llr128[0]);
#endif
    llr128[1] = _mm_unpackhi_epi32(rxFcomp128[i], xmm1);
#ifdef debug_sse
    PrintIntrinsics("Upper of (chmag - |rxF|)", "short", 128, &llr128[1]);
#endif

    llr32sse[0] = _mm_extract_epi32(llr128[0], 0); // printf("llr32_sse[0] = %d\n", llr32sse[0]);
    llr32sse[1] = _mm_extract_epi32(llr128[0], 1); // printf("llr32_sse[1] = %d\n", llr32sse[1]);
#ifdef debug_sse_llr
    int16_t *llr16sse0 = (int16_t *)&llr32sse[0];
    int16_t *llr16sse1 = (int16_t *)&llr32sse[1];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm_extract_epi16(rxFcomp128[i], 0), (short)_mm_extract_epi16(rxFcomp128[i], 1),
           llr16sse0[0], llr16sse0[1], llr16sse1[0], llr16sse1[1]);
#endif
    llr32sse[2] = _mm_extract_epi32(llr128[0], 2); // printf("llr32_sse[2] = %d\n", llr32sse[2]);
    llr32sse[3] = _mm_extract_epi32(llr128[0], 3); // printf("llr32_sse[3] = %d\n", llr32sse[3]);
#ifdef debug_sse_llr
    int16_t *llr16sse2 = (int16_t *)&llr32sse[2];
    int16_t *llr16sse3 = (int16_t *)&llr32sse[3];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm_extract_epi16(rxFcomp128[i], 2), (short)_mm_extract_epi16(rxFcomp128[i], 3),
           llr16sse2[0], llr16sse2[1], llr16sse3[0], llr16sse3[1]);
#endif
    llr32sse[4] = _mm_extract_epi32(llr128[1], 0); // printf("llr32_sse[4] = %d\n", llr32sse[4]);
    llr32sse[5] = _mm_extract_epi32(llr128[1], 1); // printf("llr32_sse[5] = %d\n", llr32sse[5]);
#ifdef debug_sse_llr
    int16_t *llr16sse4 = (int16_t *)&llr32sse[4];
    int16_t *llr16sse5 = (int16_t *)&llr32sse[5];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm_extract_epi16(rxFcomp128[i], 4), (short)_mm_extract_epi16(rxFcomp128[i], 5),
           llr16sse4[0], llr16sse4[1], llr16sse5[0], llr16sse5[1]);
#endif
    llr32sse[6] = _mm_extract_epi32(llr128[1], 2); // printf("llr32_sse[6] = %d\n", llr32sse[6]);
    llr32sse[7] = _mm_extract_epi32(llr128[1], 3); // printf("llr32_sse[7] = %d\n", llr32sse[7]);
#ifdef debug_sse_llr
    int16_t *llr16sse6 = (int16_t *)&llr32sse[6];
    int16_t *llr16sse7 = (int16_t *)&llr32sse[7];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm_extract_epi16(rxFcomp128[i], 6), (short)_mm_extract_epi16(rxFcomp128[i], 7),
           llr16sse6[0], llr16sse6[1], llr16sse7[0], llr16sse7[1]);
#endif

    llr32sse += 8;
  }
  end = clock();
  sse_cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  start = clock();
  /// ------------------------------------- AVX -------------------------------------
  printf("============================ AVX ===============================\n");
  __m256i *rxFcomp256 = (__m256i *)&rxFcomp;
  __m256i *dlchmag256 = (__m256i *)&dlchmag;
  __m256i ymm1, llr256[2];

  for (size_t i = 0; i < 2; i++)
  {
#ifdef debug_avx
    PrintIntrinsics("          rxF", "short", 256, &rxFcomp256[i]);
#endif
    ymm1 = _mm256_abs_epi16(rxFcomp256[i]);
#ifdef debug_avx
    PrintIntrinsics("        |rxF|", "short", 256, &ymm1);
#endif
    ymm1 = _mm256_subs_epi16(dlchmag256[i], ymm1);
#ifdef debug_avx
    PrintIntrinsics("        chmag", "short", 256, &dlchmag256[i]);
    PrintIntrinsics("chmag - |rxF|", "short", 256, &ymm1);
#endif

    llr256[0] = _mm256_unpacklo_epi32(rxFcomp256[i], ymm1);
#ifdef debug_avx
    PrintIntrinsics("Lower of (chmag - |rxF|)", "short", 256, &llr256[0]);
#endif
    llr256[1] = _mm256_unpackhi_epi32(rxFcomp256[i], ymm1);
#ifdef debug_avx
    PrintIntrinsics("Upper of (chmag - |rxF|)", "short", 256, &llr256[1]);
#endif

    llr32avx[0] = _mm256_extract_epi32(llr256[0], 0); // printf("llr32_avx[0] = %d\n", llr32avx[0]);
    llr32avx[1] = _mm256_extract_epi32(llr256[0], 1); // printf("llr32_avx[1] = %d\n", llr32avx[1]);
#ifdef debug_avx_llr
    int16_t *llr16avx0 = (int16_t *)&llr32avx[0];
    int16_t *llr16avx1 = (int16_t *)&llr32avx[1];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 0), (short)_mm256_extract_epi16(rxFcomp256[i], 1),
           llr16avx0[0], llr16avx0[1], llr16avx1[0], llr16avx1[1]);
#endif
    llr32avx[2] = _mm256_extract_epi32(llr256[0], 2); // printf("llr32_avx[2] = %d\n", llr32avx[2]);
    llr32avx[3] = _mm256_extract_epi32(llr256[0], 3); // printf("llr32_avx[3] = %d\n", llr32avx[3]);
#ifdef debug_avx_llr
    int16_t *llr16avx2 = (int16_t *)&llr32avx[2];
    int16_t *llr16avx3 = (int16_t *)&llr32avx[3];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 2), (short)_mm256_extract_epi16(rxFcomp256[i], 3),
           llr16avx2[0], llr16avx2[1], llr16avx3[0], llr16avx3[1]);
#endif

    llr32avx[4] = _mm256_extract_epi32(llr256[1], 0); // printf("llr32_avx[4] = %d\n", llr32avx[4]);
    llr32avx[5] = _mm256_extract_epi32(llr256[1], 1); // printf("llr32_avx[5] = %d\n", llr32avx[5]);
#ifdef debug_avx_llr
    int16_t *llr16avx4 = (int16_t *)&llr32avx[4];
    int16_t *llr16avx5 = (int16_t *)&llr32avx[5];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 4), (short)_mm256_extract_epi16(rxFcomp256[i], 5),
           llr16avx4[0], llr16avx4[1], llr16avx5[0], llr16avx5[1]);
#endif
    llr32avx[6] = _mm256_extract_epi32(llr256[1], 2); // printf("llr32_avx[6] = %d\n", llr32avx[6]);
    llr32avx[7] = _mm256_extract_epi32(llr256[1], 3); // printf("llr32_avx[7] = %d\n", llr32avx[7]);
#ifdef debug_avx_llr
    int16_t *llr16avx6 = (int16_t *)&llr32avx[6];
    int16_t *llr16avx7 = (int16_t *)&llr32avx[7];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 6), (short)_mm256_extract_epi16(rxFcomp256[i], 7),
           llr16avx6[0], llr16avx6[1], llr16avx7[0], llr16avx7[1]);
#endif

    llr32avx[8] = _mm256_extract_epi32(llr256[0], 4); // printf("llr32_avx[8] = %d\n", llr32avx[8]);
    llr32avx[9] = _mm256_extract_epi32(llr256[0], 5); // printf("llr32_avx[9] = %d\n", llr32avx[9]);
#ifdef debug_avx_llr
    int16_t *llr16avx8 = (int16_t *)&llr32avx[8];
    int16_t *llr16avx9 = (int16_t *)&llr32avx[9];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 8), (short)_mm256_extract_epi16(rxFcomp256[i], 9),
           llr16avx8[0], llr16avx8[1], llr16avx9[0], llr16avx9[1]);
#endif
    llr32avx[10] = _mm256_extract_epi32(llr256[0], 6); // printf("llr32_avx[10] = %d\n", llr32avx[10]);
    llr32avx[11] = _mm256_extract_epi32(llr256[0], 7); // printf("llr32_avx[11] = %d\n", llr32avx[11]);
#ifdef debug_avx_llr
    int16_t *llr16avx10 = (int16_t *)&llr32avx[10];
    int16_t *llr16avx11 = (int16_t *)&llr32avx[11];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 10), (short)_mm256_extract_epi16(rxFcomp256[i], 11),
           llr16avx10[0], llr16avx10[1], llr16avx11[0], llr16avx11[1]);
#endif

    llr32avx[12] = _mm256_extract_epi32(llr256[1], 4); // printf("llr32_avx[12] = %d\n", llr32avx[12]);
    llr32avx[13] = _mm256_extract_epi32(llr256[1], 5); // printf("llr32_avx[13] = %d\n", llr32avx[13]);
#ifdef debug_avx_llr
    int16_t *llr16avx12 = (int16_t *)&llr32avx[12];
    int16_t *llr16avx13 = (int16_t *)&llr32avx[13];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 12), (short)_mm256_extract_epi16(rxFcomp256[i], 13),
           llr16avx12[0], llr16avx12[1], llr16avx13[0], llr16avx13[1]);
#endif
    llr32avx[14] = _mm256_extract_epi32(llr256[1], 6); // printf("llr32_avx[14] = %d\n", llr32avx[14]);
    llr32avx[15] = _mm256_extract_epi32(llr256[1], 7); // printf("llr32_avx[15] = %d\n", llr32avx[15]);
#ifdef debug_avx_llr
    int16_t *llr16avx14 = (int16_t *)&llr32avx[14];
    int16_t *llr16avx15 = (int16_t *)&llr32avx[15];
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d] \n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 14), (short)_mm256_extract_epi16(rxFcomp256[i], 15),
           llr16avx14[0], llr16avx14[1], llr16avx15[0], llr16avx15[1]);
#endif

    llr32avx += 16;
  }

  end = clock();
  avx_cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CPU time duration for SSE = %E, and AVX256 = %E\n", sse_cpu_time, avx_cpu_time);

  int s = 0, e = 0;
  for (size_t i = 0; i < 32; i++)
  {
    if (llr32_sse[i] == llr32_avx[i])
    {
      printf("Success: llr_sse[%d] == llr_avx[%d] = (%d, %d)\n", i, i, llr32_sse[i], llr32_avx[i]);
      s++;
    }
    else
    {
      printf("Error: llr_sse[%d] == llr_avx[%d] = (%d, %d)\n", i, i, llr32_sse[i], llr32_avx[i]);
      e++;
    }
  }

  printf("Success = %d, Error = %d\n", s, e);

  return 0;
}

void PrintIntrinsics(char *s, char *dtype, int num, void *x)
{
  if (strcmp(dtype, "int") == 0)
  {
    int *tempb = (int *)x;
    if (num == 128)
    {
      printf("%s: [%d, %d, %d, %d]\n", s, tempb[0], tempb[1],
             tempb[2], tempb[3]);
    }
    else if (num == 256)
    {
      printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d]\n", s,
             tempb[0], tempb[1], tempb[2], tempb[3],
             tempb[4], tempb[5], tempb[6], tempb[7]);
    }
    else if (num == 512)
    {
      printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]\n", s,
             tempb[0], tempb[1], tempb[2], tempb[3],
             tempb[4], tempb[5], tempb[6], tempb[7],
             tempb[8], tempb[9], tempb[10], tempb[11],
             tempb[12], tempb[13], tempb[14], tempb[15]);
    }
  }

  if (strcmp(dtype, "short") == 0)
  {
    short *tempb = (short *)x;
    if (num == 128)
    {
      printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d]\n", s,
             tempb[0], tempb[1], tempb[2], tempb[3],
             tempb[4], tempb[5], tempb[6], tempb[7]);
    }
    else if (num == 256)
    {
      printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]\n", s,
             tempb[0], tempb[1], tempb[2], tempb[3],
             tempb[4], tempb[5], tempb[6], tempb[7],
             tempb[8], tempb[9], tempb[10], tempb[11],
             tempb[12], tempb[13], tempb[14], tempb[15]);
    }
    else if (num == 512)
    {
      printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]\n", s,
             tempb[0], tempb[1], tempb[2], tempb[3],
             tempb[4], tempb[5], tempb[6], tempb[7],
             tempb[8], tempb[9], tempb[10], tempb[11],
             tempb[12], tempb[13], tempb[14], tempb[15],
             tempb[16], tempb[17], tempb[18], tempb[19],
             tempb[20], tempb[21], tempb[22], tempb[23],
             tempb[24], tempb[25], tempb[26], tempb[27],
             tempb[28], tempb[29], tempb[30], tempb[31]);
    }
  }
  if (strcmp(dtype, "char") == 0)
  {
    char *tempb = (char *)x;
    printf("%s: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]\n", s,
           tempb[0], tempb[1], tempb[2], tempb[3], tempb[4], tempb[5], tempb[6], tempb[7],
           tempb[8], tempb[9], tempb[10], tempb[11], tempb[12], tempb[13], tempb[14], tempb[15]);
  }
  if (strcmp(dtype, "float") == 0)
  {
    float *tempb = (float *)x;
    printf("%s: [%f, %f, %f, %f]\n", s, tempb[0], tempb[1], tempb[2], tempb[3]);
  }
  if (strcmp(dtype, "double") == 0)
  {
    double *tempb = (double *)x;
    printf("%s: [%f, %f]\n", s, tempb[0], tempb[1]);
  }
}