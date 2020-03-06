#pragma once
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

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




/**
 * @brief lin_v_nan_index Find the index of the first NaN element of vector \p v
 * @param v Input vector
 * @param n Numner of element
 * @return Negative 1 or index of the first NaN element
 */
static int lin_v_nan_index (double const v[], unsigned n)
{
	for (unsigned i = 0; i < n; ++i)
	{
		if (isnan (v [i]))
		{
			return (int)i;
		}
	}
	return -1;
}


/**
 * @brief lin_print Print matrix or vector
 * @param ma
 * @param rn
 * @param cn
 */
static void lin_print (double const ma[], unsigned rn, unsigned cn)
{
	for (unsigned r = 0; r < rn; ++r)
	{
		for (unsigned c = 0; c < cn; ++c)
		{
			printf ("% 3.6f ", ma [rn*c + r]);
		}
		printf ("\n");
	}
	printf ("\n");
}


/**
 * @brief lin_print_t Print matrix or vector transposed
 * @param ma
 * @param rn
 * @param cn
 */
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
	assert (isnan (sb) == 0);
	for (unsigned i = 0; i < n; ++i)
	{
		assert (isnan (vx [i]) == 0);
		assert (isnan (vy [i]) == 0);
		vy [i] += vx [i] * sb;
		assert (isnan (vy [i]) == 0);
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
		assert (isnan (vx [i]) == 0);
		lin_vs_macc (vy, ma + rn * i, vx [i], rn);
	}
}



static void lin_vv_hadamard (double vy[], double const va[], double const vb[], unsigned n)
{
	for (unsigned i = 0; i < n; ++i)
	{
		assert (isnan (va [i]) == 0);
		assert (isnan (vb [i]) == 0);
		vy[i] = va[i] * vb[i];
		assert (isnan (vy [i]) == 0);
	}
}


/**
 * @brief lin_vv_sub Subtract two vectors
 * @param vy
 * @param va
 * @param vb
 * @param n
 */
static void lin_vv_sub (double vy[], double const va[], double const vb[], unsigned n)
{
	for (unsigned i = 0; i < n; ++i)
	{
		assert (isnan (va [i]) == 0);
		assert (isnan (vb [i]) == 0);
		vy[i] = va[i] - vb[i];
		assert (isnan (vy [i]) == 0);
	}
}


/**
 * @brief lin_vv_dot Dot two vectors
 * @param va
 * @param vb
 * @param n
 * @return
 */
static double lin_vv_dot (double const va[], double const vb[], unsigned n)
{
	double sum = 0;
	for (unsigned i = 0; i < n; ++i)
	{
		assert (isnan (va [i]) == 0);
		assert (isnan (vb [i]) == 0);
		sum += va [i] * vb [i];
		assert (isnan (sum) == 0);
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
		assert (isnan (vy [i]) == 0);
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
		assert (isnan (vx [i]) == 0);
		vy [i] = f (vx [i]);
		assert (isnan (vy [i]) == 0);
	}
}



/**
 * @brief lin_v_f Set function \p f return value for every element is \p vy
 * @param vy Output vector
 * @param f
 * @param n
 */
static void lin_v_f (double vy[], double (*f)(void), unsigned n)
{
	for (unsigned i = 0; i < n; ++i)
	{
		vy [i] = f ();
		assert (isnan (vy [i]) == 0);
	}
}



static double sigmoid (double x)
{
	assert (isnan (x) == 0);
	double r = 1.0 / (1.0 + exp (-x));
	assert (isnan (r) == 0);
	return r;
}


static double sigmoid_pd (double x)
{
	assert (isnan (x) == 0);
	x = x * (1.0 - x);
	assert (isnan (x) == 0);
	return x;
}


static double lin_rnd ()
{
	double r = rand() / (double)RAND_MAX;
	r = r - 0.5;
	assert (isnan (r) == 0);
	return r;
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
