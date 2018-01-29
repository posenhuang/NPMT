// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// Forward-backward probability computation using dynamic programming.
//
#include <iostream>
#include <vector>
#include <queue>
#include <tuple>
#include <algorithm>
#include <thread>
#include <cmath>
#include <cassert>
#include <random>
#include <cstring>
using namespace std;

extern "C" {
  #include "lua.h"
  #include "lualib.h"
  #include "lauxlib.h"
};

#define real double

#define IND_LOGPY(i,j,k) ((i)*(T2+1)*(max_segment_len+1)+(j)*(max_segment_len+1)+(k))
#define IND_ALPHA(i,j) ((i)*(T2+1)+(j))
#define IND_BETA(i,j) ((i)*(T2+1)+(j))

#define LOGINF 1000000
#define EPS 1e-10
#define MAX(x,y) ((x)>(y))?(x):(y)

inline real log1sub(real log_a) {
    return log1p(-exp(log_a));
}

inline real logadd(real log_a, real log_b) {
  return max(log_a, log_b) + log1p(exp(-abs(log_a-log_b)));
}

void set_all(real* x, int start, int end, real val) {
  for (int i = start; i < end; ++i) 
    x[i] = val;
}

void in_place_softmax(real* x, int start, int end) {
  real max_x = x[start];
  for (int i = start+1; i < end; ++i) {
    max_x = max(max_x, x[i]);
  }
  real sum_x = 0.;
  for (int i = start; i < end; ++i) {
    x[i] = exp(x[i] - max_x);
    sum_x += x[i];
  }
  for (int i = start; i < end; ++i) {
    x[i] /= sum_x;
  }
}

void in_place_cumsum(real* x, int start, int end) {
  real sum_x = 0.0;
  for (int i = start; i < end; ++i) {
    sum_x += x[i]; 
    x[i] = sum_x;
  }
}

extern "C" {
    static int c_sample_dp(lua_State*);
    static int c_reverse_log_cumsum(lua_State*);
    int luaopen_libdp_lib(lua_State*);
};

typedef struct {
    int batch_size;
    int T1;
    int T2;
    int max_segment_len;
    real* logpy;
    real* alpha;
    real* beta;
    real* seg_weight;
    real* ylength;
    real* xlength;
} strct_states;

static void subprocess_c_sample_dp(strct_states* s, int p_batch) {
  //default_random_engine generator;
  //uniform_real_distribution<real> distribution(0.0,1.0);
  int batch_size = s->batch_size;
  int T1 = s->T1;
  int T2 = s->T2;
  int max_segment_len = s->max_segment_len;
  real* logpy = s->logpy + p_batch * T1 * (T2+1) * (max_segment_len+1);
  real* alpha = s->alpha + p_batch * (T1+1) * (T2+1);
  real* beta = s->beta + p_batch * (T1+1) * (T2+1);
  real* seg_weight = s->seg_weight + p_batch * T1 * (T2+1) * (max_segment_len+1);
  int ylength = (int)(s->ylength[p_batch]);
  int xlength = (int)(s->xlength[p_batch]);
  alpha[IND_ALPHA(0, 0)] = 0.;
  for (int t = 1; t <= T1; ++t) {
    for (int j = 0; j <= ylength; ++j) {
      int j_low = max(1, j - max_segment_len + 1);
      for (int j_start = j_low; j_start <= j+1; ++j_start) {
        real prob = alpha[IND_ALPHA(t-1, j_start-1)] + logpy[IND_LOGPY(t-1, j_start-1, j-j_start+1)];
        alpha[IND_ALPHA(t, j)] = logadd(alpha[IND_ALPHA(t, j)], prob);
      }
    }
  }
  beta[IND_BETA(xlength, ylength)] = 0.;
  for (int t = xlength-1; t >= 0; --t) {
    for (int j = 0; j <= ylength; ++j) {
      int j_high = min(ylength, j + max_segment_len);
      for (int j_end = j; j_end <= j_high; ++j_end) {
        real prob = beta[IND_BETA(t+1, j_end)] + logpy[IND_LOGPY(t, j, j_end-j)];
        beta[IND_BETA(t, j)] = logadd(beta[IND_BETA(t, j)], prob); 
      }
    }
  }
  for (int t = 1; t <= T1; ++t) {
    int jstart_l = max(1, ylength - (T1 - t + 1) * max_segment_len +1);
    int jstart_u = min(ylength+1, (t - 1) * max_segment_len + 1);
    for (int j_start = jstart_l; j_start <= jstart_u; j_start++) {
      int j_len = min(max_segment_len, ylength-j_start+1);
      int j_end = j_start + j_len - 1;
      for (int j = j_start-1; j <= j_end; ++j) {
        seg_weight[IND_LOGPY(t-1, j_start-1, j-j_start+1)] 
          = logpy[IND_LOGPY(t-1, j_start-1,j-j_start+1)]
          + alpha[IND_ALPHA(t-1, j_start-1)]
          + beta[IND_BETA(t, j)]; 
      }
    }
  }
}

static int c_sample_dp(lua_State* L) {
  strct_states s;
  s.batch_size = (int)(lua_tonumber(L, 1));
  s.T1 = (int)(lua_tonumber(L, 2));
  s.T2 = (int)(lua_tonumber(L, 3));
  s.max_segment_len = (int)(lua_tonumber(L, 4));
  int num_thread = (int)(lua_tonumber(L, 5));
  s.logpy = (real*)((unsigned long long)(lua_tonumber(L, 6)));
  s.alpha = (real*)((unsigned long long)(lua_tonumber(L, 7)));
  s.beta = (real*)((unsigned long long)(lua_tonumber(L, 8)));
  s.seg_weight = (real*)((unsigned long long)(lua_tonumber(L, 9)));
  s.ylength = (real*)((unsigned long long)(lua_tonumber(L, 10)));
  s.xlength = (real*)((unsigned long long)(lua_tonumber(L, 11)));

  int p = 0;
  std::thread* ths = new std::thread[num_thread];
  while (p < s.batch_size) {
    int p_ths = 0;
    for(int i = 0; i < num_thread; i++) {
      ths[p_ths++] = std::thread(subprocess_c_sample_dp, &s, p++);
      if (p >= s.batch_size) break;
    }
    for(int i = 0; i < p_ths; i++) {
      ths[i].join();
    }
  }
  delete[] ths;
  return 0;
}

static void subprocess_c_reverse_log_cumsum(strct_states* s, int p_batch) {
  int batch_size = s->batch_size;
  int T1 = s->T1;
  int T2 = s->T2;
  int max_segment_len = s->max_segment_len;
  real* seg_weight = s->seg_weight + p_batch * T1 * (T2+1) * (max_segment_len+1);
  int ylength = (int)(s->ylength[p_batch]);
  for (int t = 1; t <= T1; ++t) {
    int jstart_l = max(1, ylength - (T1 - t + 1) * max_segment_len +1);
    int jstart_u = min(ylength+1, (t - 1) * max_segment_len + 1);
    for (int j_start = jstart_l; j_start <= jstart_u; j_start++) {
      int j_len = min(max_segment_len, ylength-j_start+1);
      int j_end = j_start + j_len - 1;
      for (int j = j_end-1; j >= j_start; --j) {
        seg_weight[IND_LOGPY(t-1, j_start-1, j-j_start+1)] = logadd(seg_weight[IND_LOGPY(t-1, j_start-1, j-j_start+1)],
                                                                    seg_weight[IND_LOGPY(t-1, j_start-1, j-j_start+2)]);
      }
    }
  }
}

static int c_reverse_log_cumsum(lua_State* L) {
  strct_states s;
  s.batch_size = (int)(lua_tonumber(L, 1));
  s.T1 = (int)(lua_tonumber(L, 2));
  s.T2 = (int)(lua_tonumber(L, 3));
  s.max_segment_len = (int)(lua_tonumber(L, 4));
  int num_thread = (int)(lua_tonumber(L, 5));
  s.seg_weight = (real*)((unsigned long long)(lua_tonumber(L, 6)));
  s.ylength = (real*)((unsigned long long)(lua_tonumber(L, 7)));

  int p = 0;
  std::thread* ths = new std::thread[num_thread];
  while (p < s.batch_size) {
    int p_ths = 0;
    for(int i = 0; i < num_thread; i++) {
      ths[p_ths++] = std::thread(subprocess_c_reverse_log_cumsum, &s, p++);
      if (p >= s.batch_size) break;
    }
    for(int i = 0; i < p_ths; i++) {
      ths[i].join();
    }
  }
  delete[] ths;
  return 0;

}

int luaopen_libdp_lib(lua_State* L) {
    lua_register(L, "c_sample_dp", c_sample_dp);
    lua_register(L, "c_reverse_log_cumsum", c_reverse_log_cumsum);
    return 0;
}
