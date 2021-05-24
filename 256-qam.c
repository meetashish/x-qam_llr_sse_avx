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

  // First scaled channel magnitude = 8\sqrt(170)||h||^2
  int16_t dlchmag1[] __attribute__((aligned(32))) = {42, 42, 40, 40, 40, 40, 38, 38,
                                                     38, 38, 38, 38, 40, 40, 40, 40,
                                                     40, 40, 38, 38, 38, 38, 38, 38,
                                                     38, 38, 38, 38, 38, 38, 36, 36};

  // Second scaled channel magnitude = 4\sqrt(170)||h||^2
  int16_t dlchmag2[] __attribute__((aligned(32))) = {16, 16, 16, 16, 18, 18, 18, 18,
                                                     18, 18, 18, 16, 16, 18, 18, 18,
                                                     16, 18, 18, 16, 16, 18, 18, 18,
                                                     16, 16, 18, 18, 18, 18, 18, 18};

  // Third scaled channel magnitude = 2\sqrt(170)||h||^2
  int16_t dlchmag3[] __attribute__((aligned(32))) = {16, 16, 16, 16, 18, 18, 18, 18,
                                                     18, 18, 18, 16, 16, 18, 18, 18,
                                                     16, 18, 18, 16, 16, 18, 18, 18,
                                                     16, 16, 18, 18, 18, 18, 18, 18};

  // llr
  int16_t llr_sse[128] = {[0 ... 127] = 0}, // for sse
      llr_avx[128] = {[0 ... 127] = 0};     // for avx

  int16_t *llrsse = llr_sse, *llravx = llr_avx;

  start = clock();

  /// ------------------------------------- SSE -------------------------------------
  printf("============================ SSE ===============================\n");
  __m128i *rxFcomp128 = (__m128i *)&rxFcomp;
  __m128i *dlchmag1281 = (__m128i *)&dlchmag1;
  __m128i *dlchmag1282 = (__m128i *)&dlchmag2;
  __m128i *dlchmag1283 = (__m128i *)&dlchmag3;
  __m128i xmm1, xmm2, xmm3;

  for (size_t i = 0; i < 4; i++)
  {
#ifdef debug_sse
    PrintIntrinsics("                                 rxF", "short", 128, &rxFcomp128[i]);
#endif
    xmm1 = _mm_abs_epi16(rxFcomp128[i]);
#ifdef debug_sse
    PrintIntrinsics("                               |rxF|", "short", 128, &xmm1);
#endif
    xmm1 = _mm_subs_epi16(dlchmag1281[i], xmm1);
#ifdef debug_sse
    PrintIntrinsics("                              chmag1", "short", 128, &dlchmag1281[i]);
    PrintIntrinsics("                      chmag1 - |rxF|", "short", 128, &xmm1);
#endif
    xmm2 = _mm_abs_epi16(xmm1);
#ifdef debug_sse
    PrintIntrinsics("                    |chmag1 - |rxF||", "short", 128, &xmm2);
#endif
    xmm2 = _mm_subs_epi16(dlchmag1282[i], xmm2);
#ifdef debug_sse
    PrintIntrinsics("                              chmag2", "short", 128, &dlchmag1282[i]);
    PrintIntrinsics("           chmag2 - |chmag1 - |rxF||", "short", 128, &xmm2);
#endif
    xmm3 = _mm_abs_epi16(xmm2);
#ifdef debug_sse
    PrintIntrinsics("         |chmag2 - |chmag1 - |rxF|||", "short", 128, &xmm3);
#endif
    xmm3 = _mm_subs_epi16(dlchmag1283[i], xmm3);
#ifdef debug_sse
    PrintIntrinsics("                              chmag3", "short", 128, &dlchmag1282[i]);
    PrintIntrinsics("chmag3 - |chmag2 - |chmag1 - |rxF|||", "short", 128, &xmm2);
#endif

    llrsse[0] = ((short *)&rxFcomp128[i])[0]; //printf("llr_sse[0] = %d\n", llrsse[0]);
    llrsse[1] = ((short *)&rxFcomp128[i])[1]; //printf("llr_sse[1] = %d\n", llrsse[1]);
    llrsse[2] = _mm_extract_epi16(xmm1, 0);   //printf("llr_sse[2] = %d\n", llrsse[2]);
    llrsse[3] = _mm_extract_epi16(xmm1, 1);   //printf("llr_sse[3] = %d\n", llrsse[3]);
    llrsse[4] = _mm_extract_epi16(xmm2, 0);   //printf("llr_sse[4] = %d\n", llrsse[4]);
    llrsse[5] = _mm_extract_epi16(xmm2, 1);   //printf("llr_sse[5] = %d\n", llrsse[5]);
    llrsse[6] = _mm_extract_epi16(xmm3, 0);   //printf("llr_sse[6] = %d\n", llrsse[6]);
    llrsse[7] = _mm_extract_epi16(xmm3, 1);   //printf("llr_sse[7] = %d\n", llrsse[7]);
#ifdef debug_sse
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm_extract_epi16(rxFcomp128[i], 0), (short)_mm_extract_epi16(rxFcomp128[i], 1),
           llrsse[0], llrsse[1], llrsse[2], llrsse[3], llrsse[4], llrsse[5], llrsse[6], llrsse[7]);
#endif
    llrsse += 8;

    llrsse[0] = ((short *)&rxFcomp128[i])[2]; //printf("llr_sse[0] = %d\n", llrsse[0]);
    llrsse[1] = ((short *)&rxFcomp128[i])[3]; //printf("llr_sse[1] = %d\n", llrsse[1]);
    llrsse[2] = _mm_extract_epi16(xmm1, 2);   //printf("llr_sse[2] = %d\n", llrsse[2]);
    llrsse[3] = _mm_extract_epi16(xmm1, 3);   //printf("llr_sse[3] = %d\n", llrsse[3]);
    llrsse[4] = _mm_extract_epi16(xmm2, 2);   //printf("llr_sse[4] = %d\n", llrsse[4]);
    llrsse[5] = _mm_extract_epi16(xmm2, 3);   //printf("llr_sse[5] = %d\n", llrsse[5]);
    llrsse[6] = _mm_extract_epi16(xmm3, 2);   //printf("llr_sse[6] = %d\n", llrsse[6]);
    llrsse[7] = _mm_extract_epi16(xmm3, 3);   //printf("llr_sse[7] = %d\n", llrsse[7]);
#ifdef debug_sse
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm_extract_epi16(rxFcomp128[i], 2), (short)_mm_extract_epi16(rxFcomp128[i], 3),
           llrsse[0], llrsse[1], llrsse[2], llrsse[3], llrsse[4], llrsse[5], llrsse[6], llrsse[7]);
#endif
    llrsse += 8;

    llrsse[0] = ((short *)&rxFcomp128[i])[4]; //printf("llr_sse[0] = %d\n", llrsse[0]);
    llrsse[1] = ((short *)&rxFcomp128[i])[5]; //printf("llr_sse[1] = %d\n", llrsse[1]);
    llrsse[2] = _mm_extract_epi16(xmm1, 4);   //printf("llr_sse[2] = %d\n", llrsse[2]);
    llrsse[3] = _mm_extract_epi16(xmm1, 5);   //printf("llr_sse[3] = %d\n", llrsse[3]);
    llrsse[4] = _mm_extract_epi16(xmm2, 4);   //printf("llr_sse[4] = %d\n", llrsse[4]);
    llrsse[5] = _mm_extract_epi16(xmm2, 5);   //printf("llr_sse[5] = %d\n", llrsse[5]);
    llrsse[6] = _mm_extract_epi16(xmm3, 4);   //printf("llr_sse[6] = %d\n", llrsse[6]);
    llrsse[7] = _mm_extract_epi16(xmm3, 5);   //printf("llr_sse[7] = %d\n", llrsse[7]);
#ifdef debug_sse
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm_extract_epi16(rxFcomp128[i], 4), (short)_mm_extract_epi16(rxFcomp128[i], 5),
           llrsse[0], llrsse[1], llrsse[2], llrsse[3], llrsse[4], llrsse[5], llrsse[6], llrsse[7]);
#endif
    llrsse += 8;

    llrsse[0] = ((short *)&rxFcomp128[i])[6]; //printf("llr_sse[0] = %d\n", llrsse[0]);
    llrsse[1] = ((short *)&rxFcomp128[i])[7]; //printf("llr_sse[1] = %d\n", llrsse[1]);
    llrsse[2] = _mm_extract_epi16(xmm1, 6);   //printf("llr_sse[2] = %d\n", llrsse[2]);
    llrsse[3] = _mm_extract_epi16(xmm1, 7);   //printf("llr_sse[3] = %d\n", llrsse[3]);
    llrsse[4] = _mm_extract_epi16(xmm2, 6);   //printf("llr_sse[4] = %d\n", llrsse[4]);
    llrsse[5] = _mm_extract_epi16(xmm2, 7);   //printf("llr_sse[5] = %d\n", llrsse[5]);
    llrsse[6] = _mm_extract_epi16(xmm3, 6);   //printf("llr_sse[6] = %d\n", llrsse[6]);
    llrsse[7] = _mm_extract_epi16(xmm3, 7);   //printf("llr_sse[7] = %d\n", llrsse[7]);
#ifdef debug_sse
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm_extract_epi16(rxFcomp128[i], 6), (short)_mm_extract_epi16(rxFcomp128[i], 7),
           llrsse[0], llrsse[1], llrsse[2], llrsse[3], llrsse[4], llrsse[5], llrsse[6], llrsse[7]);
#endif
    llrsse += 8;
  }

  /// ------------------------------------- AVX -------------------------------------
  printf("============================ AVX ===============================\n");
  __m256i *rxFcomp256 = (__m256i *)&rxFcomp;
  __m256i *dlchmag2561 = (__m256i *)&dlchmag1;
  __m256i *dlchmag2562 = (__m256i *)&dlchmag2;
  __m256i *dlchmag2563 = (__m256i *)&dlchmag3;
  __m256i ymm1, ymm2, ymm3;

  for (size_t i = 0; i < 2; i++)
  {
#ifdef debug_avx
    PrintIntrinsics("                                  rxF", "short", 256, &rxFcomp256[i]);
#endif
    ymm1 = _mm256_abs_epi16(rxFcomp256[i]);
#ifdef debug_avx
    PrintIntrinsics("                                |rxF|", "short", 256, &ymm1);
#endif
    ymm1 = _mm256_subs_epi16(dlchmag2561[i], ymm1);
#ifdef debug_avx
    PrintIntrinsics("                               chmag1", "short", 256, &dlchmag2561[i]);
    PrintIntrinsics("                       chmag1 - |rxF|", "short", 256, &ymm1);
#endif
    ymm2 = _mm256_abs_epi16(ymm1);
#ifdef debug_avx
    PrintIntrinsics("                     |chmag1 - |rxF||", "short", 256, &ymm2);
#endif
    ymm2 = _mm256_subs_epi16(dlchmag2562[i], ymm2);
#ifdef debug_avx
    PrintIntrinsics("                               chmag2", "short", 256, &dlchmag1282[i]);
    PrintIntrinsics("            chmag2 - |chmag1 - |rxF||", "short", 256, &ymm2);
#endif
    ymm3 = _mm256_abs_epi16(ymm2);
#ifdef debug_avx
    PrintIntrinsics("          |chmag2 - |chmag1 - |rxF|||", "short", 256, &ymm3);
#endif
    ymm3 = _mm256_subs_epi16(dlchmag2563[i], ymm3);
#ifdef debug_avx
    PrintIntrinsics("                               chmag3", "short", 256, &dlchmag1282[i]);
    PrintIntrinsics(" chmag3 - |chmag2 - |chmag1 - |rxF|||", "short", 256, &ymm2);
#endif

    llravx[0] = ((short *)&rxFcomp256[i])[0];  //printf("llr_avx[0] = %d\n", llravx[0]);
    llravx[1] = ((short *)&rxFcomp256[i])[1];  //printf("llr_avx[1] = %d\n", llravx[1]);
    llravx[2] = _mm256_extract_epi16(ymm1, 0); //printf("llr_avx[2] = %d\n", llravx[2]);
    llravx[3] = _mm256_extract_epi16(ymm1, 1); //printf("llr_avx[3] = %d\n", llravx[3]);
    llravx[4] = _mm256_extract_epi16(ymm2, 0); //printf("llr_avx[4] = %d\n", llravx[4]);
    llravx[5] = _mm256_extract_epi16(ymm2, 1); //printf("llr_avx[5] = %d\n", llravx[5]);
    llravx[6] = _mm256_extract_epi16(ymm3, 0); //printf("llr_avx[6] = %d\n", llravx[6]);
    llravx[7] = _mm256_extract_epi16(ymm3, 1); //printf("llr_avx[7] = %d\n", llravx[7]);
#ifdef debug_avx
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 0), (short)_mm256_extract_epi16(rxFcomp256[i], 1),
           llravx[0], llravx[1], llravx[2], llravx[3], llravx[4], llravx[5], llravx[6], llravx[7]);
#endif
    llravx += 8;

    llravx[0] = ((short *)&rxFcomp256[i])[2];  //printf("llr_avx[0] = %d\n", llravx[0]);
    llravx[1] = ((short *)&rxFcomp256[i])[3];  //printf("llr_avx[1] = %d\n", llravx[1]);
    llravx[2] = _mm256_extract_epi16(ymm1, 2); //printf("llr_avx[2] = %d\n", llravx[2]);
    llravx[3] = _mm256_extract_epi16(ymm1, 3); //printf("llr_avx[3] = %d\n", llravx[3]);
    llravx[4] = _mm256_extract_epi16(ymm2, 2); //printf("llr_avx[4] = %d\n", llravx[4]);
    llravx[5] = _mm256_extract_epi16(ymm2, 3); //printf("llr_avx[5] = %d\n", llravx[5]);
    llravx[6] = _mm256_extract_epi16(ymm3, 2); //printf("llr_avx[6] = %d\n", llravx[6]);
    llravx[7] = _mm256_extract_epi16(ymm3, 3); //printf("llr_avx[7] = %d\n", llravx[7]);
#ifdef debug_avx
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 2), (short)_mm256_extract_epi16(rxFcomp256[i], 3),
           llravx[0], llravx[1], llravx[2], llravx[3], llravx[4], llravx[5], llravx[6], llravx[7]);
#endif
    llravx += 8;

    llravx[0] = ((short *)&rxFcomp256[i])[4];  //printf("llr_avx[0] = %d\n", llravx[0]);
    llravx[1] = ((short *)&rxFcomp256[i])[5];  //printf("llr_avx[1] = %d\n", llravx[1]);
    llravx[2] = _mm256_extract_epi16(ymm1, 4); //printf("llr_avx[2] = %d\n", llravx[2]);
    llravx[3] = _mm256_extract_epi16(ymm1, 5); //printf("llr_avx[3] = %d\n", llravx[3]);
    llravx[4] = _mm256_extract_epi16(ymm2, 4); //printf("llr_avx[4] = %d\n", llravx[4]);
    llravx[5] = _mm256_extract_epi16(ymm2, 5); //printf("llr_avx[5] = %d\n", llravx[5]);
    llravx[6] = _mm256_extract_epi16(ymm3, 4); //printf("llr_avx[6] = %d\n", llravx[6]);
    llravx[7] = _mm256_extract_epi16(ymm3, 5); //printf("llr_avx[7] = %d\n", llravx[7]);
#ifdef debug_avx
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 4), (short)_mm256_extract_epi16(rxFcomp256[i], 5),
           llravx[0], llravx[1], llravx[2], llravx[3], llravx[4], llravx[5], llravx[6], llravx[7]);
#endif
    llravx += 8;

    llravx[0] = ((short *)&rxFcomp256[i])[6];  //printf("llr_avx[0] = %d\n", llravx[0]);
    llravx[1] = ((short *)&rxFcomp256[i])[7];  //printf("llr_avx[1] = %d\n", llravx[1]);
    llravx[2] = _mm256_extract_epi16(ymm1, 6); //printf("llr_avx[2] = %d\n", llravx[2]);
    llravx[3] = _mm256_extract_epi16(ymm1, 7); //printf("llr_avx[3] = %d\n", llravx[3]);
    llravx[4] = _mm256_extract_epi16(ymm2, 6); //printf("llr_avx[4] = %d\n", llravx[4]);
    llravx[5] = _mm256_extract_epi16(ymm2, 7); //printf("llr_avx[5] = %d\n", llravx[5]);
    llravx[6] = _mm256_extract_epi16(ymm3, 6); //printf("llr_avx[6] = %d\n", llravx[6]);
    llravx[7] = _mm256_extract_epi16(ymm3, 7); //printf("llr_avx[7] = %d\n", llravx[7]);
#ifdef debug_avx
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 6), (short)_mm256_extract_epi16(rxFcomp256[i], 7),
           llravx[0], llravx[1], llravx[2], llravx[3], llravx[4], llravx[5], llravx[6], llravx[7]);
#endif
    llravx += 8;

    llravx[0] = ((short *)&rxFcomp256[i])[8];  //printf("llr_avx[0] = %d\n", llravx[0]);
    llravx[1] = ((short *)&rxFcomp256[i])[9];  //printf("llr_avx[1] = %d\n", llravx[1]);
    llravx[2] = _mm256_extract_epi16(ymm1, 8); //printf("llr_avx[2] = %d\n", llravx[2]);
    llravx[3] = _mm256_extract_epi16(ymm1, 9); //printf("llr_avx[3] = %d\n", llravx[3]);
    llravx[4] = _mm256_extract_epi16(ymm2, 8); //printf("llr_avx[4] = %d\n", llravx[4]);
    llravx[5] = _mm256_extract_epi16(ymm2, 9); //printf("llr_avx[5] = %d\n", llravx[5]);
    llravx[6] = _mm256_extract_epi16(ymm3, 8); //printf("llr_avx[6] = %d\n", llravx[6]);
    llravx[7] = _mm256_extract_epi16(ymm3, 9); //printf("llr_avx[7] = %d\n", llravx[7]);
#ifdef debug_avx
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 8), (short)_mm256_extract_epi16(rxFcomp256[i], 9),
           llravx[0], llravx[1], llravx[2], llravx[3], llravx[4], llravx[5], llravx[6], llravx[7]);
#endif
    llravx += 8;

    llravx[0] = ((short *)&rxFcomp256[i])[10];  //printf("llr_avx[0] = %d\n", llravx[0]);
    llravx[1] = ((short *)&rxFcomp256[i])[11];  //printf("llr_avx[1] = %d\n", llravx[1]);
    llravx[2] = _mm256_extract_epi16(ymm1, 10); //printf("llr_avx[2] = %d\n", llravx[2]);
    llravx[3] = _mm256_extract_epi16(ymm1, 11); //printf("llr_avx[3] = %d\n", llravx[3]);
    llravx[4] = _mm256_extract_epi16(ymm2, 10); //printf("llr_avx[4] = %d\n", llravx[4]);
    llravx[5] = _mm256_extract_epi16(ymm2, 11); //printf("llr_avx[5] = %d\n", llravx[5]);
    llravx[6] = _mm256_extract_epi16(ymm3, 10); //printf("llr_avx[6] = %d\n", llravx[6]);
    llravx[7] = _mm256_extract_epi16(ymm3, 11); //printf("llr_avx[7] = %d\n", llravx[7]);
#ifdef debug_avx
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 10), (short)_mm256_extract_epi16(rxFcomp256[i], 11),
           llravx[0], llravx[1], llravx[2], llravx[3], llravx[4], llravx[5], llravx[6], llravx[7]);
#endif
    llravx += 8;

    llravx[0] = ((short *)&rxFcomp256[i])[12];  //printf("llr_avx[0] = %d\n", llravx[0]);
    llravx[1] = ((short *)&rxFcomp256[i])[13];  //printf("llr_avx[1] = %d\n", llravx[1]);
    llravx[2] = _mm256_extract_epi16(ymm1, 12); //printf("llr_avx[2] = %d\n", llravx[2]);
    llravx[3] = _mm256_extract_epi16(ymm1, 13); //printf("llr_avx[3] = %d\n", llravx[3]);
    llravx[4] = _mm256_extract_epi16(ymm2, 12); //printf("llr_avx[4] = %d\n", llravx[4]);
    llravx[5] = _mm256_extract_epi16(ymm2, 13); //printf("llr_avx[5] = %d\n", llravx[5]);
    llravx[6] = _mm256_extract_epi16(ymm3, 12); //printf("llr_avx[6] = %d\n", llravx[6]);
    llravx[7] = _mm256_extract_epi16(ymm3, 13); //printf("llr_avx[7] = %d\n", llravx[7]);
#ifdef debug_avx
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 12), (short)_mm256_extract_epi16(rxFcomp256[i], 13),
           llravx[0], llravx[1], llravx[2], llravx[3], llravx[4], llravx[5], llravx[6], llravx[7]);
#endif
    llravx += 8;

    llravx[0] = ((short *)&rxFcomp256[i])[14];  //printf("llr_avx[0] = %d\n", llravx[0]);
    llravx[1] = ((short *)&rxFcomp256[i])[15];  //printf("llr_avx[1] = %d\n", llravx[1]);
    llravx[2] = _mm256_extract_epi16(ymm1, 14); //printf("llr_avx[2] = %d\n", llravx[2]);
    llravx[3] = _mm256_extract_epi16(ymm1, 15); //printf("llr_avx[3] = %d\n", llravx[3]);
    llravx[4] = _mm256_extract_epi16(ymm2, 14); //printf("llr_avx[4] = %d\n", llravx[4]);
    llravx[5] = _mm256_extract_epi16(ymm2, 15); //printf("llr_avx[5] = %d\n", llravx[5]);
    llravx[6] = _mm256_extract_epi16(ymm3, 14); //printf("llr_avx[6] = %d\n", llravx[6]);
    llravx[7] = _mm256_extract_epi16(ymm3, 15); //printf("llr_avx[7] = %d\n", llravx[7]);
#ifdef debug_avx
    printf("llr of symbol (%d, %d) = [%d, %d, %d, %d, %d, %d, %d, %d]\n",
           (short)_mm256_extract_epi16(rxFcomp256[i], 14), (short)_mm256_extract_epi16(rxFcomp256[i], 15),
           llravx[0], llravx[1], llravx[2], llravx[3], llravx[4], llravx[5], llravx[6], llravx[7]);
#endif
    llravx += 8;
  }

  end = clock();
  avx_cpu_time = ((double)(end - start)) / CLOCKS_PER_SEC;
  printf("CPU time duration for SSE = %E, and AVX256 = %E\n", sse_cpu_time, avx_cpu_time);

  int s = 0, e = 0;
  for (size_t i = 0; i < 128; i++)
  {
    if (llr_sse[i] == llr_avx[i]){
      printf("Success: llr_sse[%d] == llr_avx[%d] = (%d, %d)\n", i, i, llr_sse[i], llr_avx[i]);
      s++;
    }
    else {
      printf("Error: llr_sse[%d] == llr_avx[%d] = (%d, %d)\n", i, i, llr_sse[i], llr_avx[i]);
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