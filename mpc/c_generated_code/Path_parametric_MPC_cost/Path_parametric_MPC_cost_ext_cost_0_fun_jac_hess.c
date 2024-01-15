/* This file was automatically generated by CasADi 3.6.3.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_clear CASADI_PREFIX(clear)
#define casadi_copy CASADI_PREFIX(copy)
#define casadi_densify CASADI_PREFIX(densify)
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_fill CASADI_PREFIX(fill)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)
#define casadi_s6 CASADI_PREFIX(s6)
#define casadi_s7 CASADI_PREFIX(s7)
#define casadi_sq CASADI_PREFIX(sq)
#define casadi_trans CASADI_PREFIX(trans)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

casadi_real casadi_sq(casadi_real x) { return x*x;}

void casadi_clear(casadi_real* x, casadi_int n) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = 0;
  }
}

#define CASADI_CAST(x,y) ((x) y)

void casadi_densify(const casadi_real* x, const casadi_int* sp_x, casadi_real* y, casadi_int tr) {
  casadi_int nrow_x, ncol_x, i, el;
  const casadi_int *colind_x, *row_x;
  if (!y) return;
  nrow_x = sp_x[0]; ncol_x = sp_x[1];
  colind_x = sp_x+2; row_x = sp_x+ncol_x+3;
  casadi_clear(y, nrow_x*ncol_x);
  if (!x) return;
  if (tr) {
    for (i=0; i<ncol_x; ++i) {
      for (el=colind_x[i]; el!=colind_x[i+1]; ++el) {
        y[i + row_x[el]*ncol_x] = CASADI_CAST(casadi_real, *x++);
      }
    }
  } else {
    for (i=0; i<ncol_x; ++i) {
      for (el=colind_x[i]; el!=colind_x[i+1]; ++el) {
        y[row_x[el]] = CASADI_CAST(casadi_real, *x++);
      }
      y += nrow_x;
    }
  }
}

void casadi_copy(const casadi_real* x, casadi_int n, casadi_real* y) {
  casadi_int i;
  if (y) {
    if (x) {
      for (i=0; i<n; ++i) *y++ = *x++;
    } else {
      for (i=0; i<n; ++i) *y++ = 0.;
    }
  }
}

void casadi_fill(casadi_real* x, casadi_int n, casadi_real alpha) {
  casadi_int i;
  if (x) {
    for (i=0; i<n; ++i) *x++ = alpha;
  }
}

void casadi_trans(const casadi_real* x, const casadi_int* sp_x, casadi_real* y,
    const casadi_int* sp_y, casadi_int* tmp) {
  casadi_int ncol_x, nnz_x, ncol_y, k;
  const casadi_int* row_x, *colind_y;
  ncol_x = sp_x[1];
  nnz_x = sp_x[2 + ncol_x];
  row_x = sp_x + 2 + ncol_x+1;
  ncol_y = sp_y[1];
  colind_y = sp_y+2;
  for (k=0; k<ncol_y; ++k) tmp[k] = colind_y[k];
  for (k=0; k<nnz_x; ++k) {
    y[tmp[row_x[k]]++] = x[k];
  }
}

static const casadi_int casadi_s0[6] = {9, 1, 0, 2, 0, 2};
static const casadi_int casadi_s1[14] = {9, 9, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2};
static const casadi_int casadi_s2[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s3[5] = {1, 1, 0, 1, 0};
static const casadi_int casadi_s4[3] = {0, 0, 0};
static const casadi_int casadi_s5[10] = {6, 1, 0, 6, 0, 1, 2, 3, 4, 5};
static const casadi_int casadi_s6[13] = {9, 1, 0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8};
static const casadi_int casadi_s7[12] = {0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess:(i0[8],i1,i2[],i3[6])->(o0,o1[9],o2[9x9,2nz],o3[],o4[0x9]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real *rr, *ss;
  casadi_real w0, w1, w2, w3, w4, w5, w6, w7, *w15=w+8, *w16=w+10, *w17=w+19, *w18=w+21;
  /* #0: @0 = input[3][3] */
  w0 = arg[3] ? arg[3][3] : 0;
  /* #1: @1 = input[0][1] */
  w1 = arg[0] ? arg[0][1] : 0;
  /* #2: @2 = input[3][4] */
  w2 = arg[3] ? arg[3][4] : 0;
  /* #3: @1 = (@1-@2) */
  w1 -= w2;
  /* #4: @2 = sq(@1) */
  w2 = casadi_sq( w1 );
  /* #5: @2 = (@0*@2) */
  w2  = (w0*w2);
  /* #6: @3 = 0.001 */
  w3 = 1.0000000000000000e-03;
  /* #7: @4 = input[1][0] */
  w4 = arg[1] ? arg[1][0] : 0;
  /* #8: @5 = (@3*@4) */
  w5  = (w3*w4);
  /* #9: @6 = (@5*@4) */
  w6  = (w5*w4);
  /* #10: @2 = (@2+@6) */
  w2 += w6;
  /* #11: @6 = input[3][0] */
  w6 = arg[3] ? arg[3][0] : 0;
  /* #12: @6 = (@4-@6) */
  w6  = (w4-w6);
  /* #13: @7 = sq(@6) */
  w7 = casadi_sq( w6 );
  /* #14: @2 = (@2+@7) */
  w2 += w7;
  /* #15: output[0][0] = @2 */
  if (res[0]) res[0][0] = w2;
  /* #16: @6 = (2.*@6) */
  w6 = (2.* w6 );
  /* #17: @6 = (@6+@5) */
  w6 += w5;
  /* #18: @4 = (@3*@4) */
  w4  = (w3*w4);
  /* #19: @6 = (@6+@4) */
  w6 += w4;
  /* #20: @8 = 00 */
  /* #21: @1 = (2.*@1) */
  w1 = (2.* w1 );
  /* #22: @1 = (@1*@0) */
  w1 *= w0;
  /* #23: @9 = 00 */
  /* #24: @10 = 00 */
  /* #25: @11 = 00 */
  /* #26: @12 = 00 */
  /* #27: @13 = 00 */
  /* #28: @14 = 00 */
  /* #29: @15 = vertcat(@6, @8, @1, @9, @10, @11, @12, @13, @14) */
  rr=w15;
  *rr++ = w6;
  *rr++ = w1;
  /* #30: @16 = dense(@15) */
  casadi_densify(w15, casadi_s0, w16, 0);
  /* #31: output[1][0] = @16 */
  casadi_copy(w16, 9, res[1]);
  /* #32: @15 = zeros(9x9,2nz) */
  casadi_clear(w15, 2);
  /* #33: @16 = ones(9x1) */
  casadi_fill(w16, 9, 1.);
  /* #34: {@6, NULL, @1, NULL, NULL, NULL, NULL, NULL, NULL} = vertsplit(@16) */
  w6 = w16[0];
  w1 = w16[2];
  /* #35: @4 = (2.*@6) */
  w4 = (2.* w6 );
  /* #36: @5 = (@3*@6) */
  w5  = (w3*w6);
  /* #37: @4 = (@4+@5) */
  w4 += w5;
  /* #38: @3 = (@3*@6) */
  w3 *= w6;
  /* #39: @4 = (@4+@3) */
  w4 += w3;
  /* #40: @8 = 00 */
  /* #41: @1 = (2.*@1) */
  w1 = (2.* w1 );
  /* #42: @0 = (@0*@1) */
  w0 *= w1;
  /* #43: @9 = 00 */
  /* #44: @10 = 00 */
  /* #45: @11 = 00 */
  /* #46: @12 = 00 */
  /* #47: @13 = 00 */
  /* #48: @14 = 00 */
  /* #49: @17 = vertcat(@4, @8, @0, @9, @10, @11, @12, @13, @14) */
  rr=w17;
  *rr++ = w4;
  *rr++ = w0;
  /* #50: @18 = @17[:2] */
  for (rr=w18, ss=w17+0; ss!=w17+2; ss+=1) *rr++ = *ss;
  /* #51: (@15[:2] = @18) */
  for (rr=w15+0, ss=w18; rr!=w15+2; rr+=1) *rr = *ss++;
  /* #52: @18 = @17[:2] */
  for (rr=w18, ss=w17+0; ss!=w17+2; ss+=1) *rr++ = *ss;
  /* #53: (@15[:2] = @18) */
  for (rr=w15+0, ss=w18; rr!=w15+2; rr+=1) *rr = *ss++;
  /* #54: @18 = @15' */
  casadi_trans(w15,casadi_s1, w18, casadi_s1, iw);
  /* #55: output[2][0] = @18 */
  casadi_copy(w18, 2, res[2]);
  return 0;
}

CASADI_SYMBOL_EXPORT int Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_n_in(void) { return 4;}

CASADI_SYMBOL_EXPORT casadi_int Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_n_out(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_real Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    case 3: return "o3";
    case 4: return "o4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s2;
    case 1: return casadi_s3;
    case 2: return casadi_s4;
    case 3: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s6;
    case 2: return casadi_s1;
    case 3: return casadi_s4;
    case 4: return casadi_s7;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int Path_parametric_MPC_cost_ext_cost_0_fun_jac_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 13;
  if (sz_res) *sz_res = 14;
  if (sz_iw) *sz_iw = 10;
  if (sz_w) *sz_w = 23;
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
