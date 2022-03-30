// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// package lp implements routines for solving linear programs.
package lp

import (
	"errors"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// // TODO(btracey): Could have a solver structure with an abstract factorizer. With
// // this transformation the same high-level code could handle both Dense and Sparse.
// // TODO(btracey): Need to improve error handling. Only want to panic if condition number inf.
// // TODO(btracey): Performance enhancements. There are currently lots of linear
// // solves that can be improved by doing rank-one updates. For example, the swap
// // step is just a rank-one update.
// // TODO(btracey): Better handling on the linear solve errors. If the condition
// // number is not inf and the equation solved "well", should keep moving.

var (
	ErrBland      = errors.New("lp: bland: all replacements are negative or cause ill-conditioned ab")
	ErrInfeasible = errors.New("lp: problem is infeasible")
	ErrLinSolve   = errors.New("lp: linear solve failure")
	ErrUnbounded  = errors.New("lp: problem is unbounded")
	ErrSingular   = errors.New("lp: A is singular")
	ErrZeroColumn = errors.New("lp: A has a column of all zeros")
	ErrZeroRow    = errors.New("lp: A has a row of all zeros")
)

const (
	INITIALMU = float64(10)
	ALHPHA    = float64(0.01)
	BETA      = float64(0.5)
)

type FunctionInfo struct {
	// Func evaluates the objective function at the given location. Func
	// must not modify x.
	Func func(x []float64) float64

	// Grad evaluates the gradient at x and stores the result in grad which will
	// be the same length as x. Grad must not modify x.
	Grad func(grad, x []float64)

	// Hess evaluates the Hessian at x and stores the result in-place in hess which
	// will have dimensions matching the length of x. Hess must not modify x.
	Hess func(hess *mat.SymDense, x []float64)
}

func PrimalDualInteriorPointMethod(problem *FunctionInfo, nonLinearConstrain []*FunctionInfo, linearConstrain mat.Matrix, initialPt []float64) (float64, []float64, error) {
	x := make([]float64, len(initialPt))
	copy(x, initialPt)
	// TODO: How do we set lambda ?
	m := len(nonLinearConstrain)
	lambda := make([]float64, m)
	mu := INITIALMU

	f := computef(nonLinearConstrain, x)
	Df := computeDf(nonLinearConstrain, x)
	etaHat := computeEtaHat(f, lambda)

	// Step1: Determine t. Set t := μm/ηˆ.
	t := mu * m / etaHat

	return 0, nil, nil
}

// TODO: Compute General Gradient
func computeDf(nonLinearConstrain []*FunctionInfo, x []float64) *mat.Dense {
	result := mat.NewDense(len(nonLinearConstrain), len(x), nil)
	for i := 0; i < len(nonLinearConstrain); i++ {
		temp := make([]float64, len(x))
		nonLinearConstrain[i].Grad(temp, x)
		for j := 0; j < len(x); j++ {
			result.Set(i, j, temp[j])
		}
	}
	return result
}

func computefAndInverse(nonLinearConstrain []*FunctionInfo, x []float64) []float64 {
	result := make([]float64, len(nonLinearConstrain))
	for i := 0; i < len(nonLinearConstrain); i++ {
		result[i] = nonLinearConstrain[i].Func(x)
	}
	return result
}

// ηˆ(x, λ) = −f (x)T λ.
func computeEtaHat(fvalue []float64, lambda []float64) float64 {
	return (-1) * floats.Dot(fvalue, lambda)
}

func computePrimalDualSearchDirection() {
	return
}

func computeRemainder(f0 *FunctionInfo, x []float64, Df *mat.Dense, lambda *mat.Dense, mu *mat.Dense, A *mat.Dense, b *mat.DenseG) *mat.Dense {
	totalLength := len(x) + Df.RawMatrix().Rows + A.RawMatrix().Rows
	result := mat.NewDense(totalLength, 1, nil)

	Df0 := make([]float64, len(x))
	for i := 0; i < len(Df0); i++ {
		f0.Grad(Df0, x)
	}
	mat.NewDense(len(x), 1, Df0)

	var rdual, Atranspose *mat.Dense
	rdual.Mul(Df, lambda)
	Atranspose.CloneFrom(A)
	Atranspose.T()
	rdual.Add(rdual, Atranspose)
	rdual.Add(rdual, mat.NewDense(len(x), 1, Df0))



}
