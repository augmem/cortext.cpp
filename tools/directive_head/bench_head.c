/* CPU cost of one standing-directive head call, as the engine would run it (scalar C, one thread).

   build: cc -O2 -o bench_head bench_head.c -lm
   run:   ./bench_head D H [iters]      D = input dim (256 or 1280), H = hidden (0 = logistic)
          ./bench_head attn T K [iters]  attention pool over T token states (384-d), K heads, LayerNorm, linear
   Each call: (for D=256) slice + L2 re-norm of the first 256 dims of a 1280-d vector, fc1 + erf GELU, fc2.
   Weights are random: cost does not depend on their values. Prints ns per call. */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ns (void)
{
  struct timespec t;
  clock_gettime (CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e9 + t.tv_nsec;
}

static int bench_attn (int T, int K, long iters)
{
  enum { DT = 384 };
  float *tok = malloc (sizeof (float) * 128 * DT * 8), *wg = malloc (sizeof (float) * K * DT), bg[16], g[128 * 16];
  float *gam = malloc (sizeof (float) * K * DT), *bet = malloc (sizeof (float) * K * DT), *wf = malloc (sizeof (float) * K * DT);
  float p[16 * DT];
  srand (2);
  for (int i = 0; i < 128 * DT * 8; ++i) tok[i] = rand () / (float) RAND_MAX - 0.5f;
  for (int i = 0; i < K * DT; ++i) { wg[i] = (rand () / (float) RAND_MAX - 0.5f) * 0.1f; gam[i] = 1; bet[i] = 0; wf[i] = 0.01f; }
  for (int k = 0; k < K; ++k) bg[k] = 0;
  volatile float sink = 0;
  double t0 = now_ns ();
  for (long it = 0; it < iters; ++it)
    {
      const float *x = tok + (it & 7) * 128 * DT;
      for (int t = 0; t < T; ++t)
        for (int k = 0; k < K; ++k)
          {
            float a = bg[k];
            for (int i = 0; i < DT; ++i) a += wg[k * DT + i] * x[t * DT + i];
            g[t * K + k] = a;
          }
      for (int k = 0; k < K; ++k)
        {
          float mx = -1e30f, sum = 0;
          for (int t = 0; t < T; ++t) mx = g[t * K + k] > mx ? g[t * K + k] : mx;
          for (int t = 0; t < T; ++t) { g[t * K + k] = expf (g[t * K + k] - mx); sum += g[t * K + k]; }
          for (int i = 0; i < DT; ++i) p[k * DT + i] = 0;
          for (int t = 0; t < T; ++t)
            {
              float wt = g[t * K + k] / sum;
              for (int i = 0; i < DT; ++i) p[k * DT + i] += wt * x[t * DT + i];
            }
        }
      double mu = 0, var = 0;
      for (int i = 0; i < K * DT; ++i) mu += p[i];
      mu /= K * DT;
      for (int i = 0; i < K * DT; ++i) var += (p[i] - mu) * (p[i] - mu);
      float inv = (float) (1.0 / sqrt (var / (K * DT) + 1e-5));
      float out = 0;
      for (int i = 0; i < K * DT; ++i) out += wf[i] * ((p[i] - (float) mu) * inv * gam[i] + bet[i]);
      sink += out;
    }
  double dt = now_ns () - t0;
  printf ("attn T=%d K=%d iters=%ld ns/call=%.1f params=%d (sink %.3f)\n", T, K, iters, dt / iters,
          K * DT + K + 2 * K * DT + K * DT + 1, (double) sink);
  return 0;
}

int main (int argc, char **argv)
{
  if (argc > 1 && argv[1][0] == 'a')
    return bench_attn (argc > 2 ? atoi (argv[2]) : 32, argc > 3 ? atoi (argv[3]) : 4, argc > 4 ? atol (argv[4]) : 100000);
  int D = argc > 1 ? atoi (argv[1]) : 256, H = argc > 2 ? atoi (argv[2]) : 64;
  long iters = argc > 3 ? atol (argv[3]) : 200000;
  int HH = H ? H : 1;
  float *w1 = malloc (sizeof (float) * HH * D), *b1 = malloc (sizeof (float) * HH), *w2 = malloc (sizeof (float) * HH);
  float *x = malloc (sizeof (float) * 1280 * 64), z[1280], h[1024];
  srand (1);
  for (int i = 0; i < HH * D; ++i) w1[i] = (rand () / (float) RAND_MAX - 0.5f) * 0.1f;
  for (int i = 0; i < HH; ++i) { b1[i] = 0.01f; w2[i] = 0.1f; }
  for (int i = 0; i < 1280 * 64; ++i) x[i] = rand () / (float) RAND_MAX - 0.5f;
  volatile float sink = 0;
  double t0 = now_ns ();
  for (long it = 0; it < iters; ++it)
    {
      const float *v = x + (it & 63) * 1280;
      double n = 0;
      for (int i = 0; i < D; ++i) n += (double) v[i] * v[i];
      float inv = (float) (1.0 / sqrt (n > 1e-24 ? n : 1e-24));
      for (int i = 0; i < D; ++i) z[i] = v[i] * inv;
      float out = 0.0f;
      if (H)
        {
          for (int j = 0; j < H; ++j)
            {
              const float *r = w1 + (long) j * D;
              float a = b1[j];
              for (int i = 0; i < D; ++i) a += r[i] * z[i];
              h[j] = 0.5f * a * (1.0f + erff (a * 0.70710678f));
            }
          for (int j = 0; j < H; ++j) out += w2[j] * h[j];
        }
      else
        for (int i = 0; i < D; ++i) out += w1[i] * z[i];
      sink += out;
    }
  double dt = now_ns () - t0;
  printf ("D=%d H=%d iters=%ld ns/call=%.1f params=%d (sink %.3f)\n", D, H, iters, dt / iters,
          H ? H * D + H + H + 1 : D + 1, (double) sink);
  return 0;
}
