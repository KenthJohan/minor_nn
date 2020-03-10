#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

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


void cw (double mw1[], double const vd1[], double const va0[], unsigned n1, unsigned n0, double learningrate)
{
	for (unsigned r = 0; r < n1; ++r)
	{
		for (unsigned c = 0; c < n0; ++c)
		{
			//w_rc = a_c * d_r
			//TODO: What is happening here?
			//mw1 [n0*r + c] -= va0[c] * vd1[r] * learningrate;
			mw1 [n1*c + r] -= va0[c] * vd1[r] * learningrate;
		}
	}
}


#define L0 2
#define L1 2
#define L2 1
#define SAMPLECOUNT 4
#define LEARNINGRATE 0.75

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
	//return 1;

	double w1 [L1*L0] = {0};
	double w2 [L2*L1] = {0};
	double a1 [L1] = {0};
	double a2 [L2] = {0};
	double d1 [L1] = {0};
	double d2 [L2] = {0};
	double mse = 0;

	//srand ((unsigned int)time(0));
	srand ((unsigned int)1);
	lin_v_f (w1, lin_rnd, L1*L0);
	lin_v_f (w2, lin_rnd, L2*L1);

	int iterations = 0;
	double p[10] = {0};
	while (1)
	{
		//Train:
		for (int j = 0; j < 100; ++j)
		for (int i = 0; i < SAMPLECOUNT; ++i)
		{
			iterations++;
			fw (a1, x[i], w1, L1, L0);
			fw (a2,   a1, w2, L2, L1);
			lin_vv_sub (d2, a2, y[i], L2); //d2 := a2 - y
			lin_v_fx (p, a2, sigmoid_pd, L2); //a2 := sigmoid_pd (a2)
			lin_vv_hadamard (d2, d2, p, L2); //d2 := d2 hadamard a2
			bp (d1, d2, a1, w2, L2, L1);
			cw (w1, d1, x[i], L1, L0, LEARNINGRATE);
			cw (w2, d2, a1, L2, L1, LEARNINGRATE);
		}


		printf ("=============Evaluate==========\n");
		lin_print (w1, L1, L0, "% 3.1f ", "\n");printf ("\n");
		lin_print (w2, L2, L1, "% 3.1f ", "\n");printf ("\n");
		for (int i = 0; i < SAMPLECOUNT; ++i)
		{
			fw (a1, x[i], w1, L1, L0);
			fw (a2,   a1, w2, L2, L1);
			lin_vv_sub (d2, a2, y[i], L2);
			mse += d2[0] * d2[0];
			printf ("(");lin_print (x[i], L0, 1, " %1.0f", "");printf (")");
			printf (" => ");
			printf ("(");lin_print (a2, L2, 1, "%1.3f", "");printf (")");
			fputs ("\n", stdout);
		}
		printf ("mse = %4.4f, iterations : %i\n", mse, iterations);
		mse = 0;


		int c = fgetc (stdin);
		if (c == 'q') {return 0;}
	}



	return 0;
}
