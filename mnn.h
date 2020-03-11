#pragma once
#include "lin.h"


void fw (double v1[], double const v0[], double const mw[], unsigned n1, unsigned n0)
{
	lin_mv_mul (v1, mw, v0, n1, n0); //v1 := mw * v0
	lin_v_fx (v1, v1, sigmoid, n1); //v1 := sigmoid (v1)
	//lin_print (v1, n1, 1);
}


void bp (double vd0[], double const vd1[], double const va0[], double const mw[], unsigned n1, unsigned n0)
{
	double va0s[10];
	lin_mv_mul_t (vd0, mw, vd1, n1, n0); //vd0 := transpose(mw) * vd1
	lin_v_fx (va0s, va0, sigmoid_pd, n0); //va0s := sigmoid_pd (va0)
	lin_vv_hadamard (vd0, vd0, va0s, n0); //vd0 := vd0 hadamard va0s
}


void cw (double mw1[], double const vd1[], double const va0[], unsigned n1, unsigned n0, double lr)
{
	for (unsigned r = 0; r < n1; ++r)
	{
		for (unsigned c = 0; c < n0; ++c)
		{
			//w_rc = a_c * d_r
			//TODO: What is happening here?
			mw1 [n1*c + r] -= va0[c] * vd1[r] * lr;
		}
	}
}
