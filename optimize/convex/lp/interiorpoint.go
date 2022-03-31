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
	ErrOverRetry  = errors.New("lp: Over the max Retry")
)

const (
	INITIALMU = float64(10)
	ALHPHA    = float64(0.01)
	BETA      = float64(0.5)
	MaxRetry  = 1000
	// TODO: Reset this value
	EPSILONFEAS = float64(0.00000001)
	EPSILON     = float64(0.00000001)
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
	Hess func(hess *mat.Dense, x []float64)
}

func PrimalDualInteriorPointMethod(problem *FunctionInfo, nonLinearConstrain []*FunctionInfo, linearConstrain, b *mat.Dense, initialPt []float64) ([]float64, error) {
	x := make([]float64, len(initialPt))
	copy(x, initialPt)
	// TODO: How do we set lambda ?
	m := len(nonLinearConstrain)
	lambda := mat.NewDense(m, 1, nil)
	mu := INITIALMU
	nu := mat.NewDense(linearConstrain.RawMatrix().Rows, 1, nil)

	fFloatArray := computef(nonLinearConstrain, x)
	Df := computeDf(nonLinearConstrain, x)
	etaHat := float64(100)

	f := mat.NewDense(m, 1, fFloatArray)
	r := mat.NewDense(len(x)+m+linearConstrain.RawMatrix().Rows, 1, nil)
	rdualNorm2 := floats.Norm(r.RawMatrix().Data[0:len(x)], 2)
	rpriNorm2 := floats.Norm(r.RawMatrix().Data[len(x)+m:len(r.RawMatrix().Data)], 2)
	Deltax := make([]float64, len(x))
	DeltaLambda := make([]float64, m)
	DeltaNu := make([]float64, linearConstrain.RawMatrix().Rows)
	Finalx := make([]float64, len(x))
	FinalLambda := make([]float64, m)
	FinalNu := make([]float64, linearConstrain.RawMatrix().Rows)

	// ∥rpri∥2 ≤ epsilon_feas, ∥rdual∥2 ≤ oepsilon_feas, and ηˆ ≤ epsilon.
	for (rdualNorm2 > EPSILONFEAS) || (rpriNorm2 > EPSILONFEAS) || (etaHat > EPSILON) {
		etaHat = computeEtaHat(fFloatArray, lambda.RawMatrix().Data)

		// Step1: Determine t. Set t := μm/ηˆ.
		t := mu * float64(m) / etaHat
		tInverse := float64(1) / t
		// Step2: Compute primal-dual search direction ∆ypd.
		deltaY, err := computePrimeDualMethodMove(problem, x, nonLinearConstrain, f, Df, nu, b, lambda, linearConstrain, tInverse)
		if err != nil {
			return nil, err
		}
		Deltax = deltaY[0:len(x)]
		DeltaLambda = deltaY[len(x) : len(x)+m]
		DeltaNu = deltaY[len(x)+m : len(deltaY)]

		// Step3: Line search and update. Determine step length s > 0 and set y := y + s∆ypd.
		Finalx, FinalLambda, FinalNu, err = lineSearch(problem, nonLinearConstrain, x, nu.RawMatrix().Data, lambda.RawMatrix().Data, DeltaLambda, Deltax, DeltaNu, f, Df, linearConstrain, b, tInverse)
		if err != nil {
			return nil, err
		}

		r = computeRemainder(problem, Finalx, f, Df, mat.NewDense(m, 1, FinalLambda), mat.NewDense(linearConstrain.RawMatrix().Rows, 1, FinalNu), linearConstrain, b, tInverse)
		rdualNorm2 = floats.Norm(r.RawMatrix().Data[0:len(x)], 2)
		rpriNorm2 = floats.Norm(r.RawMatrix().Data[len(x)+m:len(r.RawMatrix().Data)], 2)
	}
	return Finalx, nil
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

func computef(nonLinearConstrain []*FunctionInfo, x []float64) []float64 {
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

func computeRemainder(f0 *FunctionInfo, x []float64, f, Df *mat.Dense, lambda *mat.Dense, Nu *mat.Dense, A *mat.Dense, b *mat.Dense, tInverse float64) *mat.Dense {
	Df0 := make([]float64, len(x))
	for i := 0; i < len(Df0); i++ {
		f0.Grad(Df0, x)
	}
	mat.NewDense(len(x), 1, Df0)

	var rdual, rpri, AtransposeMulNu *mat.Dense
	rdual.Mul(Df, lambda)
	AtransposeMulNu.CloneFrom(A.T())
	AtransposeMulNu.Mul(AtransposeMulNu, Nu)
	rdual.Add(rdual, AtransposeMulNu)
	rdual.Add(rdual, mat.NewDense(len(x), 1, Df0))

	rcentFloat64 := make([]float64, lambda.RawMatrix().Rows)
	fFloat64 := f.RawMatrix().Data
	copy(rcentFloat64, lambda.RawMatrix().Data)
	for i := 0; i < len(rcentFloat64); i++ {
		rcentFloat64[i] *= fFloat64[i]
		rcentFloat64[i] += tInverse
		rcentFloat64[i] = -1 * rcentFloat64[i]
	}

	rpri.Mul(A, mat.NewDense(len(x), 1, x))
	rpri.Sub(rpri, b)

	result := append(rdual.RawMatrix().Data, rcentFloat64...)
	result = append(result, rpri.RawMatrix().Data...)
	return mat.NewDense(len(result), 1, result)
}

func computeKKTMatrix(f0 *FunctionInfo, x []float64, nonLinearConstrain []*FunctionInfo, f, df, lambda, A *mat.Dense) *mat.Dense {
	hass := mat.NewDense(len(x), len(x), nil)
	f0.Hess(hass, x)
	sumHass := make([]float64, len(hass.RawMatrix().Data))
	for i := 0; i < len(sumHass); i++ {
		sumHass[i] = hass.RawMatrix().Data[i]
	}
	for i := 0; i < len(nonLinearConstrain); i++ {
		temp := mat.NewDense(len(x), len(x), nil)
		nonLinearConstrain[i].Hess(temp, x)
		tempScale := make([]float64, len(temp.RawMatrix().Data))
		// TODO : Check it
		floats.ScaleTo(tempScale, lambda.RawMatrix().Data[i], temp.RawMatrix().Data)
		floats.Add(sumHass, tempScale)
	}

	var DfTranspose, ATranspose *mat.Dense
	DfTranspose.CloneFrom(df.T())
	ATranspose.CloneFrom(A.T())

	negDiagLambdaMulDf := make([]float64, len(df.RawMatrix().Data))
	for i := 0; i < len(lambda.RawMatrix().Data); i++ {
		temp := 0 - lambda.RawMatrix().Data[i]
		iIndex := i * df.RawMatrix().Cols
		for j := 0; j < df.RawMatrix().Cols; j++ {
			index := iIndex + j
			negDiagLambdaMulDf[index] = temp * df.RawMatrix().Data[index]
		}
	}

	negDiagf := mat.NewDense(len(nonLinearConstrain), len(nonLinearConstrain), nil)
	for i := 0; i < len(nonLinearConstrain); i++ {
		negDiagf.Set(i, i, -1*f.RawMatrix().Data[i])
	}

	// Combine all matrices
	dim := len(x) + len(nonLinearConstrain) + A.RawMatrix().Rows
	result := mat.NewDense(dim, dim, nil)

	// Set ∇^2f_0(x)+sum_i=1^m = λi∇^2f_i(x)
	for i := 0; i < len(x); i++ {
		index := i * len(x)
		for j := 0; j < len(x); j++ {
			result.Set(i, j, sumHass[index+j])
		}
	}

	// Set -diag(lambda)*Df(x)
	startRowIndex := len(x)
	for i := 0; i < len(nonLinearConstrain); i++ {
		for j := 0; j < len(x); j++ {
			result.Set(startRowIndex+i, j, negDiagf.At(i, j))
		}
	}

	// Set Df^T
	startColumnIndex := len(x)
	for i := 0; i < len(nonLinearConstrain); i++ {
		for j := 0; j < len(x); j++ {
			result.Set(i, startColumnIndex+j, DfTranspose.At(i, j))
		}
	}

	// Set -diag(f)
	for i := 0; i < len(nonLinearConstrain); i++ {
		for j := 0; j < len(nonLinearConstrain); j++ {
			result.Set(startRowIndex+i, startColumnIndex+j, negDiagf.At(i, j))
		}
	}

	// Set A^T
	startColumnIndex += len(nonLinearConstrain)
	for i := 0; i < ATranspose.RawMatrix().Rows; i++ {
		for j := 0; j < ATranspose.RawMatrix().Cols; j++ {
			result.Set(i, startColumnIndex+j, ATranspose.At(i, j))
		}
	}
	// Set A
	startRowIndex += len(nonLinearConstrain)
	for i := 0; i < A.RawMatrix().Rows; i++ {
		for j := 0; j < A.RawMatrix().Cols; j++ {
			result.Set(i+startRowIndex, j, A.At(i, j))
		}
	}
	return result
}

func computePrimeDualMethodMove(f0 *FunctionInfo, x []float64, nonLinearConstrain []*FunctionInfo, f, df, Nu, b, lambda, A *mat.Dense, tInverse float64) ([]float64, error) {
	KKTMatrix := computeKKTMatrix(f0, x, nonLinearConstrain, f, df, lambda, A)
	r := computeRemainder(f0, x, f, df, lambda, Nu, A, b, tInverse)

	var solution *mat.Dense
	err := solution.Inverse(KKTMatrix)
	if err != nil {
		return nil, err
	}
	solution.Mul(solution, r)
	result := solution.RawMatrix().Data
	floats.Scale(-1, result)
	return result, nil
}

func lineSearch(f0 *FunctionInfo, nonLinearConstrain []*FunctionInfo, x, Nu, lambda, DeltaLambda, Deltax, Deltamu []float64, f, Df, A, b *mat.Dense, tInverse float64) ([]float64, []float64, []float64, error) {
	sMax := 1.0
	for i := 0; i < len(lambda); i++ {
		temp := (0 - lambda[i]) / DeltaLambda[i]
		if sMax > temp {
			sMax = temp
		}
	}
	s := 0.99 * sMax
	xPlus := make([]float64, len(x))
	lambdaPlus := make([]float64, len(lambda))
	muPlus := make([]float64, len(Nu))
	floats.ScaleTo(xPlus, s, Deltax)
	floats.Add(xPlus, x)

	for i := 0; i <= MaxRetry; i++ {
		bCorrect := true
		for j := 0; j < len(nonLinearConstrain); j++ {
			if nonLinearConstrain[j].Func(xPlus) > 0 {
				bCorrect = false
				break
			}
		}
		if bCorrect {
			break
		}
		s = s * BETA
		floats.ScaleTo(xPlus, s, Deltax)
		floats.Add(xPlus, x)
		if i == MaxRetry {
			return nil, nil, nil, ErrOverRetry
		}
	}
	floats.ScaleTo(lambdaPlus, s, DeltaLambda)
	floats.Add(lambdaPlus, lambda)
	floats.ScaleTo(muPlus, s, Deltamu)
	floats.Add(muPlus, Nu)
	compareNorm := floats.Norm(computeRemainder(f0, x, f, Df, mat.NewDense(len(lambda), 1, lambda), mat.NewDense(len(Nu), 1, Nu), A, b, tInverse).RawMatrix().Data, 2)
	for i := 0; i <= MaxRetry; i++ {
		rMatrix := computeRemainder(f0, xPlus, f, Df, mat.NewDense(len(lambdaPlus), 1, lambdaPlus), mat.NewDense(len(muPlus), 1, muPlus), A, b, tInverse)
		rNorm := floats.Norm(rMatrix.RawMatrix().Data, 2)
		comparePart := (1 - ALHPHA*s) * compareNorm
		if rNorm > comparePart {
			s = s * BETA
			floats.ScaleTo(xPlus, s, Deltax)
			floats.Add(xPlus, x)
			floats.ScaleTo(lambdaPlus, s, DeltaLambda)
			floats.Add(lambdaPlus, lambda)
			floats.ScaleTo(muPlus, s, Deltamu)
			floats.Add(muPlus, Nu)
			break
		}
		return nil, nil, nil, ErrOverRetry
	}
	return xPlus, lambdaPlus, muPlus, nil
}
