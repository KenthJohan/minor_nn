#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#include "lin.h"



void fw (double v1[], double const v0[], double const mw[], unsigned n1, unsigned n0)
{
	assert (lin_v_nan_index (v0, n0) == -1);
	assert (lin_v_nan_index (mw, n1*n0) == -1);
	lin_mv_mul (v1, mw, v0, n1, n0); //v1 := mw * v0
	assert (lin_v_nan_index (v1, n1) == -1);
	lin_v_fx (v1, v1, sigmoid, n1); //v1 := sigmoid (v1)
	//lin_print (v1, n1, 1);
}


void bp (double vd0[], double const vd1[], double vz0[], double const mw[], unsigned n1, unsigned n0)
{
	//Backpropagate
	lin_mv_mul_t (vd0, mw, vd1, n1, n0); //vd0 := transpose(mw) * vd1
	lin_v_fx (vz0, vz0, sigmoid_pd, n0); //vz0 := sigmoid_pd (vz0)
	lin_vv_hadamard (vd0, vd0, vz0, n0); //vd0 := vd0 hadamard vz0
}



void cw (double mw1[], double const vd1[], double const va0[], unsigned n1, unsigned n0)
{
	for (unsigned r = 0; r < n1; ++r)
	{
		for (unsigned c = 0; c < n0; ++c)
		{
			//w_rc = a_c * d_r
			mw1 [n0*r + c] -= va0[c] * vd1[r] * 0.05;
		}
	}
}


#define L0 2
#define L1 2
#define L2 1
#define SAMPLECOUNT 4


static double x[SAMPLECOUNT][L0] =
{
{0.0, 0.0},
{0.0, 1.0},
{1.0, 0.0},
{1.0, 1.0},
};

static double y[SAMPLECOUNT][L2] =
{
{0.0},
{1.0},
{1.0},
{0.0},
};

int main (int argc, char * argv [])
{
	assert (argc);
	assert (argv);
	//lin_test_mv_mul ();
	//lin_test_mv_mul_t ();

	//Network topology is (L0, L1, L2):
	//w1 : L0 inputs and L1 outputs:
	//w2 : L1 inputs and L2 outputs:
	double w1 [L1*L0];
	double w2 [L2*L1];
	//Activated values:
	double a1 [L1] = {0};
	double a2 [L2] = {0};
	//Delta for packpropagation:
	double d1 [L1] = {0};
	double d2 [L2] = {0};
	//Mean square error:
	double mse = 0;

	//Init every weight to a random value with seed = currentime:
	srand ((unsigned int)time(0));
	lin_v_f (w1, lin_rnd, L1*L0);
	lin_v_f (w2, lin_rnd, L2*L1);

	/*
	lin_print (w1, L1, L0);
	lin_print (w2, L2, L1);
	return 1;
	*/

	while (1)
	{
		for (int j = 0; j < 100000; ++j)
		for (int i = 0; i < SAMPLECOUNT; ++i)
		{
			fw (a1, x[i], w1, L1, L0);
			fw (a2,   a1, w2, L2, L1);
			lin_vv_sub (d2, a2, y[i], L2); //d2 := a2 - y
			lin_v_fx (a2, a2, sigmoid_pd, L2); //a2 := sigmoid_pd (a2)
			lin_vv_hadamard (d2, d2, a2, L1); //d2 := d2 hadamard a2
			bp (d1, d2, a1, w2, L2, L1);
			cw (w1, d1, x[i], L1, L0);
			cw (w2, d2, a1, L2, L1);
		}



		for (int i = 0; i < SAMPLECOUNT; ++i)
		{
			//lin_print (w1, L1, L0);
			//lin_print (w2, L2, L1);
			fw (a1, x[i], w1, L1, L0);
			fw (a2,   a1, w2, L2, L1);
			printf ("%i %i % 2.6f (% 2.6f % 2.6f % 2.6f % 2.6f) (% 2.6f % 2.10f)\n", (int)x[i][0], (int)x[i][1], a2[0], w1[0], w1[1], w1[2], w1[3], w2[0], w2[1]);
			/*
			lin_print (w1, L1, L0);
			lin_print (w2, L2, L1);
			lin_print (x[i], 1, L0);
			lin_print (a1, 1, L1);
			lin_print (a2, 1, L2);
			lin_print (y [i], 1, L2);
			printf ("mse % 3.10f\n", lin_vv_mse (a2, y[i], L2));
			*/
			lin_vv_sub (d2, a2, y[i], L2);
			mse += d2[0] * d2[0];
		}

		int ch = fgetc(stdin);
		printf ("mse %f\n", mse);
		mse = 0;
	}




}
