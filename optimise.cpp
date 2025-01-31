// dllmain.cpp : Defines the entry point for the DLL application.
#include <windows.h>

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include "lbfgsb.h"
#include "gradient_strategy.h"

#if defined(USING_EIGEN)
#define EIGEN_NO_DEBUG  // Disable debug mode for Eigen (faster)
//#define EIGEN_DONT_PARALLELIZE  // Disable multi-threading if the overhead is too large for small matrices
#define EIGEN_VECTORIZE  // Enable vectorization
#endif

BOOL APIENTRY DllMain(HMODULE hModule,
					  DWORD  ul_reason_for_call,
					  LPVOID lpReserved
)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
	case DLL_THREAD_ATTACH:
	case DLL_THREAD_DETACH:
	case DLL_PROCESS_DETACH:
		break;
	}
	return TRUE;
}

#define DllExport __declspec(dllexport)

extern "C" DllExport bool lbfgsb(double (*func)(const double*), void(*grad)(const double*, double*), size_t n, double* x,
								 const double* lba, const double* uba, int max_history, int max_iter, 
								 int lnsrch_max_iter, double tol, double c1, double c2, double alpha_max, 
								 double eps_factor, bool debug)
{
	/*
	func - pointer to a function to be optimised
	grad - pointer to the gradient of the function - returns the gradient as the second parameter
	n - the size of the problem
	x - pointer to the array that holds the initial guess and returns the optimised value
	lba - pointer to the array holding the lower bounds
	uba - pointer to the array holding the upper bounds
	max_iter - maimum number of iterations
	tol - the tolerance for optimsality
	c1, c2, alpha_ma - parameters for the line search
	*/
	VectorXd ga(n);
	auto f = [&func](const VectorXd& x_) -> double
		{
			double f = func(x_.data());
			return f;
		};
	auto g = [&grad, &ga](const VectorXd& x_) -> VectorXd
		{
			grad(x_.data(), ga.data());
			return ga;
		};
#if defined(USING_EIGEN)
	Eigen::Map<const Eigen::VectorXd> lb(lba, n);
	Eigen::Map<const Eigen::VectorXd> ub(uba, n);
	Eigen::VectorXd xv = Eigen::VectorXd::Map(x, n);
#else
	VectorXd lb(lba, lba + n), ub(uba, uba + n), xv(x, x + n);
#endif
	bool status;
	if (grad == nullptr)
	{
		gradients::NumericalGradient n_grad(f);
		status = LBFGSB::optimize(f, n_grad, xv, lb, ub, max_history, max_iter, lnsrch_max_iter, tol, c1, c2, alpha_max, eps_factor, 
								  debug);
	}
	else
	{
		status = LBFGSB::optimize(f, g, xv, lb, ub, max_history, max_iter, lnsrch_max_iter, tol, c1, c2, alpha_max, eps_factor,
								  debug);
	}
	std::copy(xv.data(), xv.data() + n, x);
	return status;
}
