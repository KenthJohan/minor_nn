#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "lin.h"
#include "mnn.h"


void fw1 (unsigned const l[], unsigned const l1[], double const w[], double a[], double const x[])
{
	assert (l < l1);
	//(L0 L1 L2 L3)
	//( x a1 a2 a3)
	//(w0 w1 w2)
	fw (a, x, w, l[1], l[0]);
	w += l[1] * l[0];
	l++;
	while ((l != l1) && l[1])
	{
		fw (a + l[0], a, w, l[1], l[0]);
		//lin_mv_mul (a+n0, w, a, l[1], l[0]);
		//lin_v_fx (a+n0, a+n0, sigmoid, n1);
		a += l[0];
		w += l[1] * l[0];
		l++;
	}
}


void bp1 (unsigned l0[], unsigned l[], double w[], double a[], double d[], double y[])
{
	assert (l > l0);
	//(w0 w1 w2)
	//(a0 a1 a2 a3)
	//(d0 d1 d2 d3)
	//(L0 L1 L2 L3 00)
	//          n0 n1
	//          d0 d1
	//             w1
	double p[10];
	lin_vv_sub (d, a, y, l[0]);
	lin_v_fx (p, a, sigmoid_pd, l[0]); //p := sigmoid_pd (a)
	lin_vv_hadamard (d, d, p, l[0]); //d := d hadamard p
	l--;
	a -= l[0];
	d -= l[0];


	while (l != l0)
	{
		lin_mv_mul_t (d, w, d + l[0], l[1], l[0]); //d := transpose(w) * d'
		lin_v_fx (p, a, sigmoid_pd, l[0]); //p := sigmoid_pd (a)
		lin_vv_hadamard (d, d, p, l[0]); //d := d hadamard p
		a -= l[0];
		d -= l[0];
		w -= l[1]*l[0];
		l--;
	}

}


void cw1 (unsigned const l[], unsigned const l1[], double w[], double const a[], double const d[], double const x[], double lr)
{
	cw (w, d, x, l[1], l[0], lr);
	d += l[1];
	w += l[1] * l[0];
	l++;

	while (l != l1)
	{
		cw (w, d, a, l[1], l[0], lr);
		d += l[0];
		a += l[0];
		w += l[1] * l[0];
		l++;
	}

}


#define L0 3
#define L1 2
#define L2 1


#define SAMPLECOUNT 4
#define LR 0.5
#define BATCH 10000

static double x[SAMPLECOUNT][L0] =
{
{0.0, 0.0, 1.0},
{0.0, 1.0, 1.0},
{1.0, 0.0, 1.0},
{1.0, 1.0, 1.0},
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

	double w [L1*L0 + L2*L1] = {0};
	double a [L1+L2] = {0};
	double d [L1+L2] = {0};
	unsigned t [4] = {L0, L1, L2, 0};

	union
	{
		double val_f64;
		uint64_t val_u64;
	} mse;

	//Init every weight to a random value with seed = currentime:
	//srand ((unsigned int)time(0));
	srand ((unsigned int)1);
	lin_v_f (w, lin_rnd, L1*L0 + L2*L1);

	int iterations = 0;
	while (1)
	{
		//Train:
		for (int j = 0; j < BATCH; ++j)
		for (int i = 0; i < SAMPLECOUNT; ++i)
		{
			iterations++;
			fw1 (t, t+2, w, a, x[i]);
			bp1 (t, t+2, w + L1*L0, a + L1, d + L1, y[i]);
			cw1 (t, t+2, w, a, d, x[i], LR);
		}

		//Evaluate:
		printf ("=============Evaluate==========\n");
		lin_print (w,       L1, L0, "% 3.1f ", "\n");printf ("\n");
		lin_print (w+L1*L0, L2, L1, "% 3.1f ", "\n");printf ("\n");
		for (int i = 0; i < SAMPLECOUNT; ++i)
		{
			fw1 (t, t+2, w, a, x[i]);
			lin_vv_sub (d + L1, a + L1, y[i], L2);
			mse.val_f64 += lin_vv_dot (d + L1, d + L1, L2);
			printf ("(");lin_print (x[i], L0, 1, " %1.0f", "");printf (")");
			printf (" => ");
			printf ("(");lin_print (a + L1, L2, 1, "%1.3f", "");printf (")");
			fputs ("\n", stdout);
		}
		printf ("mse = %4.4f %016jx, iterations : %i\n", mse.val_f64, mse.val_u64, iterations);
		//assert (mse.val_u64 == 0x3fd02e3e8eef8f55);
		mse.val_f64 = 0;


		int c = fgetc (stdin);
		if (c == 'q') {return 0;}
	}




}
