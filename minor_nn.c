/*

Column major:
double a[] =
{
	1.0, 2.0,//Column 0
	3.0, 4.0,//Column 1
};

(1.0 2.0) is column 0
(3.0 4.0) is column 1

true matrix form:
	1.0 3.0
	2.0 4.0

Matrix have rn*cn elements.
Vector have rn*cn elements where cn = 1.

Convention, decleare scalar starting with s:
	double sx;
Convention, decleare vector starting with v:
	double vx[2];
Convention, declare matrix starting with m:
	double mx[2*2]

*/

#include <stdio.h>
#include <assert.h>
#include <math.h>


static void lin_print (double const ma[], unsigned rn, unsigned cn)
{
	for (unsigned r = 0; r < rn; ++r)
	{
		for (unsigned c = 0; c < cn; ++c)
		{
			printf ("%2.2f ", ma [rn*c + r]);
		}
		printf ("\n");
	}
	printf ("\n");
}


static void lin_print_t (double const ma[], unsigned rn, unsigned cn)
{
	for (unsigned c = 0; c < cn; ++c)
	{
		for (unsigned r = 0; r < rn; ++r)
		{
			printf ("%2.2f ", ma [rn*c + r]);
		}
		printf ("\n");
	}
	printf ("\n");
}


/**
 * @brief lin_vs_macc Multiply and accumulate vector \p va by scalar \p sb.
 * @param vy Output vector
 * @param vx Input left side vector
 * @param sb Input right side scalar
 * @param n Number of elements in vector \p vy and vector \p vx
 */
static void lin_vs_macc (double vy[], double const vx[], double sb, unsigned n)
{
	for (unsigned i = 0; i < n; ++i)
	{
		vy [i] += vx [i] * sb;
	}
}


/**
 * @brief lin_mv_mul Multiply matrix \p ma by vector \p vx.
 * @param vy Output vector
 * @param ma Input left side matrix
 * @param vx Input right side vector
 * @param rn Number of rows in matrix \p ma and number of elements in vector \p vy
 * @param cn Number of columns in matrix \p ma and number of elements in vector \p vx
 */
static void lin_mv_mul (double vy[], double const ma[], double const vx[], unsigned rn, unsigned cn)
{
	for (unsigned i = 0; i < cn; ++i)
	{
		lin_vs_macc (vy, ma + rn * i, vx [i], rn);
	}
}


static void lin_vv_hadamard (double vy[], double const va[], double const vb[], unsigned n)
{
	for (unsigned i = 0; i < n; ++i)
	{
		vy[i] = va[i] * vb[i];
	}
}


static void lin_vv_sub (double vy[], double const va[], double const vb[], unsigned n)
{
	for (unsigned i = 0; i < n; ++i)
	{
		vy[i] = va[i] - vb[i];
	}
}



static double lin_vv_dot (double const va[], double const vb[], unsigned n)
{
	double sum = 0;
	for (unsigned i = 0; i < n; ++i)
	{
		sum += va [i] * vb [i];
	}
	return sum;
}


/**
 * @brief lin_mv_mul_t Multiply transposed matrix \p ma by vector \p vx.
 * @param vy Output vector
 * @param ma Input left side matrix
 * @param vx Input right side vector
 * @param rn Number of rows in matrix \p ma and number of elements in vector \p vx
 * @param cn Number of columns in matrix \p ma and number of elements in vector \p vy
 */
static void lin_mv_mul_t (double vy[], double const ma[], double const vx[], unsigned rn, unsigned cn)
{
	for (unsigned i = 0; i < cn; ++i)
	{
		vy [i] = lin_vv_dot (ma + rn * i, vx, rn);
	}
}


/**
 * @brief lin_v_fx
 * @param vy Output vector
 * @param vx Input vector
 * @param f(x)
 * @param n Number of elements in vector \p vy and \p vx
 */
static void lin_v_fx (double vy[], double const vx[], double (*f)(double x), unsigned n)
{
	for (unsigned i = 0; i < n; ++i)
	{
		vy [i] = f (vx [i]);
	}
}



static double sigmoid (double x)
{
	/*
	if (x < -45.0) return 0;
	if (x > 45.0) return 1;
	*/
	return 1.0 / (1.0 + exp (-x));
}


static double sigmoid_pd (double x)
{
	return x * (1.0 - x);
}


static double cost (double a, double b)
{
	double d = a - b;
	return 0.5 * d * d;
}


static double cost_pd (double a, double b)
{
	double d = a - b;
	return d;
}


void lin_test_mv_mul ()
{
#define RN 3
#define CN 2
	double ma[RN*CN] =
	{
	1.0, 2.0, 3.0,//Column 0
	4.0, 5.0, 6.0,//Column 1
	};
	double vx[CN] = {1.0, 10.0};
	double vy[RN] = {0};
	lin_mv_mul (vy, ma, vx, RN, CN);
	lin_print (ma, RN, CN);
	lin_print (vx, CN, 1);
	lin_print (vy, RN, 1);
	assert (vy[0] == (vx[0]*ma[0]+vx[1]*ma[3]));
	assert (vy[1] == (vx[0]*ma[1]+vx[1]*ma[4]));
	assert (vy[2] == (vx[0]*ma[2]+vx[1]*ma[5]));
	assert (vy[0] == 41.0);
	assert (vy[1] == 52.0);
	assert (vy[2] == 63.0);
#undef RN
#undef CN
}


void lin_test_mv_mul_t ()
{
#define RN 3
#define CN 2
	double ma[RN*CN] =
	{
	1.0, 2.0, 3.0,//Column 0
	4.0, 5.0, 6.0,//Column 1
	};
	double vx1[RN] = {1.0, 10.0, 100.0};
	double vy1[CN] = {0};
	lin_mv_mul_t (vy1, ma, vx1, RN, CN);
	lin_print_t (ma, RN, CN);
	lin_print_t (vx1, RN, 1);
	lin_print_t (vy1, CN, 1);
	assert (vy1[0] == (vx1[0]*ma[0] + vx1[1]*ma[1] + vx1[2]*ma[2]));
	assert (vy1[1] == (vx1[0]*ma[3] + vx1[1]*ma[4] + vx1[2]*ma[5]));
	assert (vy1[0] == 321.0);
	assert (vy1[1] == 654.0);
#undef RN
#undef CN
}




#define L0 3
#define L1 2
#define L2 1
#define SAMPLECOUNT 4

double x[SAMPLECOUNT][L0] =
{
	{0.0, 0.0, 0.0},
	{0.0, 1.0, 0.0},
	{1.0, 0.0, 0.0},
	{1.0, 1.0, 0.0},
};

double y[SAMPLECOUNT][L2] =
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

	//L1 by L0 weight matrix, takes (L0) number of inputs and (L1) number of outputs.
	double a0 [L0]; //Layer 0 inputs
	double d0 [L0]; //Error
	double w0 [L1*L0];

	double z1 [L1]; //Layer 1 input/output
	double a1 [L1]; //Layer 1 input/output
	double d1 [L1]; //Error
	double w1 [L2*L1];

	double z2 [L2];
	double a2 [L2];
	double d2 [L2];

	//Feedforward:
	for (int i = 0; i < SAMPLECOUNT; ++i)
	{
		lin_mv_mul (z1, w0, x[i], L1, L0); //z1 := w0 * a0
		lin_v_fx (a1, z1, sigmoid, L1); //a1 := sigmoid (z1)
		lin_print (a1, L1, 1);


		lin_mv_mul (z2, w1, a1, L2, L1); //z2 := w1 * a1
		lin_v_fx (a2, z2, sigmoid, L2); //a2 := sigmoid (z2)
		lin_print (a2, L2, 1);

		//error
		lin_vv_sub (d2, a2, y[i], L2);

		//Backpropagate
		lin_mv_mul_t (d1, w1, d2, L1, L0); //d0 := transpose(w0) * d1
		lin_v_fx (a0, a0, sigmoid_pd, L0); //a0 := sigmoid_d (a0)
		lin_vv_hadamard (d0, d0, a0, L1); //d0 := d0 hadamard a0
	}





}
