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

var (
	ErrOverRetry = errors.New("lp: Over the max Retry")
)

const (
	INITIALMU = float64(10)
	ALHPHA    = float64(0.01)
	BETA      = float64(0.5)
	MaxRetry  = 1000
	// TODO: Reset this value
	EPSILONFEAS = 1e-12
	EPSILON     = 1e-12
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

func PrimalDualInteriorPointMethod(problem *FunctionInfo, nonEqConstrain []*FunctionInfo, EqConstrain, b *mat.Dense, initialPt []float64, mu float64) ([]float64, error) {
	x := make([]float64, len(initialPt))
	copy(x, initialPt)
	// TODO: How do we set lambda ?
	m := len(nonEqConstrain)
	nu := mat.NewDense(EqConstrain.RawMatrix().Rows, 1, nil)
	// for i := 0; i < nu.RawMatrix().Rows; i++ {
	// 	nu.Set(i, 0, 10)
	// }

	fFloatArray := computef(nonEqConstrain, x)
	Df := computeDf(nonEqConstrain, x)
	etaHat := float64(100)

	f := mat.NewDense(m, 1, fFloatArray)
	r := mat.NewDense(len(x)+m+EqConstrain.RawMatrix().Rows, 1, nil)
	rdualNorm2 := floats.Norm(r.RawMatrix().Data[0:len(x)], 2)
	rpriNorm2 := floats.Norm(r.RawMatrix().Data[len(x)+m:len(r.RawMatrix().Data)], 2)
	Finalx := make([]float64, len(x))
	FinalLambda := make([]float64, m)
	FinalNu := make([]float64, EqConstrain.RawMatrix().Rows)

	// Set lambda
	lambdaSlice := make([]float64, m)
	for i := 0; i < m; i++ {
		lambdaSlice[i] = -1 / nonEqConstrain[i].Func(initialPt)
	}
	lambda := mat.NewDense(m, 1, lambdaSlice)
	// Step1: Determine t. Set t := μm/ηˆ.
	etaHat = computeEtaHat(fFloatArray, lambda.RawMatrix().Data)
	tInverse := etaHat / (mu * float64(m))

	// ∥rpri∥2 ≤ epsilon_feas, ∥rdual∥2 ≤ oepsilon_feas, and ηˆ ≤ epsilon.
	for (rdualNorm2 > EPSILONFEAS) || (rpriNorm2 > EPSILONFEAS) || (etaHat > EPSILON) {
		// Step2: Compute primal-dual search direction ∆ypd.
		deltaY, err := computePrimeDualMethodMove(problem, x, nonEqConstrain, f, Df, nu, b, lambda, EqConstrain, tInverse)
		if err != nil {
			return nil, err
		}

		// Step3: Line search and update. Determine step length s > 0 and set y := y + s∆ypd.
		Finalx, FinalLambda, FinalNu, err = lineSearch(problem, nonEqConstrain, x, nu.RawMatrix().Data, lambda.RawMatrix().Data, deltaY, f, Df, EqConstrain, b, tInverse)
		if err != nil {
			return nil, err
		}
		// Reset
		x = Finalx
		nu = mat.NewDense(nu.RawMatrix().Rows, nu.RawMatrix().Cols, FinalNu)
		lambda = mat.NewDense(lambda.RawMatrix().Rows, lambda.RawMatrix().Cols, FinalLambda)
		fFloatArray = computef(nonEqConstrain, x)
		Df = computeDf(nonEqConstrain, x)
		f = mat.NewDense(m, 1, fFloatArray)

		etaHat = computeEtaHat(fFloatArray, lambda.RawMatrix().Data)
		tInverse = etaHat / (mu * float64(m))

		r = computeRemainder(problem, x, f, Df, lambda, nu, EqConstrain, b, tInverse)
		rdualNorm2 = floats.Norm(r.RawMatrix().Data[0:len(x)], 2)
		rpriNorm2 = floats.Norm(r.RawMatrix().Data[len(x)+m:len(r.RawMatrix().Data)], 2)
	}
	return x, nil
}

// TODO: Compute General Gradient
func computeDf(nonEqConstrain []*FunctionInfo, x []float64) *mat.Dense {
	//result := mat.NewDense(len(nonEqConstrain), len(x), nil)
	gradSlice := make([]float64, 0)
	for i := 0; i < len(nonEqConstrain); i++ {
		temp := make([]float64, len(x))
		nonEqConstrain[i].Grad(temp, x)
		gradSlice = append(gradSlice, temp...)
		// for j := 0; j < len(x); j++ {
		// 	result.Set(i, j, temp[j])
		// }
	}
	return mat.NewDense(len(nonEqConstrain), len(x), gradSlice)
}

func computef(nonEqConstrain []*FunctionInfo, x []float64) []float64 {
	result := make([]float64, len(nonEqConstrain))
	for i := 0; i < len(nonEqConstrain); i++ {
		result[i] = nonEqConstrain[i].Func(x)
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
	f0.Grad(Df0, x)
	//fmt.Println("f0:", Df0)

	rdual := mat.NewDense(len(x), 1, nil)
	rpri := mat.NewDense(A.RawMatrix().Rows, 1, nil)

	DfTransPose := mat.NewDense(Df.RawMatrix().Cols, Df.RawMatrix().Rows, nil)
	DfTransPose.CloneFrom(Df.T())

	AtransposeMulNu := mat.NewDense(A.RawMatrix().Cols, A.RawMatrix().Rows, nil)
	// Compute ∇f_0(x)+Df(x)^T*λ+A^Tv
	rdual.Mul(DfTransPose, lambda)
	AtransposeMulNu.CloneFrom(A.T())
	product := mat.NewDense(AtransposeMulNu.RawMatrix().Rows, Nu.RawMatrix().Cols, nil)
	product.Mul(AtransposeMulNu, Nu)
	rdual.Add(rdual, product)
	rdual.Add(rdual, mat.NewDense(len(x), 1, Df0))

	// Compute -diag(λ)f(x) - 1/t*I
	rcentFloat64 := make([]float64, lambda.RawMatrix().Rows)
	fFloat64 := f.RawMatrix().Data
	//copy(rcentFloat64, lambda.RawMatrix().Data)
	for i := 0; i < len(rcentFloat64); i++ {
		rcentFloat64[i] = lambda.RawMatrix().Data[i] * fFloat64[i]
		rcentFloat64[i] += tInverse
		rcentFloat64[i] *= -1
	}
	// Compute Ax-b
	rpri.Mul(A, mat.NewDense(len(x), 1, x))
	rpri.Sub(rpri, b)

	result := append(rdual.RawMatrix().Data, rcentFloat64...)
	result = append(result, rpri.RawMatrix().Data...)
	return mat.NewDense(len(result), 1, result)
}

func computeKKTMatrix(f0 *FunctionInfo, x []float64, nonEqConstrain []*FunctionInfo, f, Df, lambda, A *mat.Dense) *mat.Dense {
	hass := mat.NewDense(len(x), len(x), nil)
	f0.Hess(hass, x)
	sumHass := make([]float64, len(hass.RawMatrix().Data))
	for i := 0; i < len(sumHass); i++ {
		sumHass[i] = hass.RawMatrix().Data[i]
	}
	// Compute ∇^2f_0(x)+sum_i=1^m = λi∇^2f_i(x)
	for i := 0; i < len(nonEqConstrain); i++ {
		temp := mat.NewDense(len(x), len(x), nil)
		nonEqConstrain[i].Hess(temp, x)
		tempScale := make([]float64, len(temp.RawMatrix().Data))
		// TODO : Check it
		floats.ScaleTo(tempScale, lambda.RawMatrix().Data[i], temp.RawMatrix().Data)
		floats.AddTo(sumHass, sumHass, tempScale)
	}

	DfTranspose := mat.NewDense(Df.RawMatrix().Cols, Df.RawMatrix().Rows, nil)
	ATranspose := mat.NewDense(A.RawMatrix().Cols, A.RawMatrix().Rows, nil)
	DfTranspose.CloneFrom(Df.T())
	ATranspose.CloneFrom(A.T())

	// -diag(lambda)*Df(x)
	negDiagLambdaMulDf := make([]float64, len(Df.RawMatrix().Data))
	for i := 0; i < len(lambda.RawMatrix().Data); i++ {
		temp := 0 - lambda.RawMatrix().Data[i]
		iIndex := i * Df.RawMatrix().Cols
		for j := 0; j < Df.RawMatrix().Cols; j++ {
			index := iIndex + j
			negDiagLambdaMulDf[index] = temp * Df.RawMatrix().Data[index]
		}
	}

	// -diag(f(x))
	negDiagf := mat.NewDense(len(nonEqConstrain), len(nonEqConstrain), nil)
	for i := 0; i < len(nonEqConstrain); i++ {
		negDiagf.Set(i, i, -1*f.RawMatrix().Data[i])
	}

	// Combine all matrices
	dim := len(x) + len(nonEqConstrain) + A.RawMatrix().Rows
	result := mat.NewDense(dim, dim, nil)

	// Set ∇^2f_0(x)+sum_i=1^m = λi∇^2f_i(x)
	for i := 0; i < len(x); i++ {
		index := i * len(x)
		for j := 0; j < len(x); j++ {
			result.Set(i, j, sumHass[index+j])
		}
	}

	// Set -diag(lambda)*Df(x)
	diagNegDf := make([]float64, len(Df.RawMatrix().Data))
	for i := 0; i < lambda.RawMatrix().Rows; i++ {
		starIndex := i * Df.RawMatrix().Cols
		for j := 0; j < Df.RawMatrix().Cols; j++ {
			index := starIndex + j
			diagNegDf[index] = (0 - lambda.RawMatrix().Data[i]) * Df.RawMatrix().Data[index]
		}
	}
	matNegDiagDf := mat.NewDense(Df.RawMatrix().Rows, Df.RawMatrix().Cols, diagNegDf)
	startRowIndex := len(x)
	for i := 0; i < len(nonEqConstrain); i++ {
		for j := 0; j < len(x); j++ {
			result.Set(startRowIndex+i, j, matNegDiagDf.At(i, j))
		}
	}

	// Set Df^T
	startColumnIndex := len(x)
	for i := 0; i < len(x); i++ {
		for j := 0; j < len(nonEqConstrain); j++ {
			result.Set(i, startColumnIndex+j, DfTranspose.At(i, j))
		}
	}

	// Set -diag(f)
	for i := 0; i < len(nonEqConstrain); i++ {
		for j := 0; j < len(nonEqConstrain); j++ {
			result.Set(startRowIndex+i, startColumnIndex+j, negDiagf.At(i, j))
		}
	}

	// Set A^T
	startColumnIndex += len(nonEqConstrain)
	for i := 0; i < ATranspose.RawMatrix().Rows; i++ {
		for j := 0; j < ATranspose.RawMatrix().Cols; j++ {
			result.Set(i, startColumnIndex+j, ATranspose.At(i, j))
		}
	}

	// Set A
	startRowIndex += len(nonEqConstrain)
	for i := 0; i < A.RawMatrix().Rows; i++ {
		for j := 0; j < A.RawMatrix().Cols; j++ {
			result.Set(i+startRowIndex, j, A.At(i, j))
		}
	}

	return result
}

func computePrimeDualMethodMove(f0 *FunctionInfo, x []float64, nonEqConstrain []*FunctionInfo, f, Df, Nu, b, lambda, A *mat.Dense, tInverse float64) ([]float64, error) {
	KKTMatrix := computeKKTMatrix(f0, x, nonEqConstrain, f, Df, lambda, A)
	r := computeRemainder(f0, x, f, Df, lambda, Nu, A, b, tInverse)
	solution := mat.NewDense(KKTMatrix.RawMatrix().Rows, KKTMatrix.RawMatrix().Cols, nil)
	err := solution.Inverse(KKTMatrix)
	if err != nil {
		return nil, err
	}
	result := mat.NewDense(solution.RawMatrix().Rows, r.RawMatrix().Cols, nil)
	result.Mul(solution, r)
	floats.ScaleTo(result.RawMatrix().Data, -1, result.RawMatrix().Data)
	return result.RawMatrix().Data, nil
}

func lineSearch(f0 *FunctionInfo, nonEqConstrain []*FunctionInfo, x, Nu, lambda, deltaY []float64, f, Df, A, b *mat.Dense, tInverse float64) ([]float64, []float64, []float64, error) {
	Deltax := deltaY[0:len(x)]
	DeltaLambda := deltaY[len(x) : len(x)+len(nonEqConstrain)]
	DeltaNu := deltaY[len(x)+len(nonEqConstrain):]
	sMax := 1.0
	// min{1, min{-lambda_i / Delta lambda_i | Delta lambda_i < 0}}
	for i := 0; i < len(lambda); i++ {
		if DeltaLambda[i] >= 0 {
			continue
		}
		temp := (0 - lambda[i]) / DeltaLambda[i]
		if sMax > temp {
			sMax = temp
		}
	}
	s := 0.99 * sMax
	xPlus := make([]float64, len(x))
	lambdaPlus := make([]float64, len(lambda))
	nuPlus := make([]float64, len(Nu))
	// x^+ := x+s*Delta_pd
	floats.ScaleTo(xPlus, s, Deltax)
	floats.AddTo(xPlus, xPlus, x)

	for i := 0; i <= MaxRetry; i++ {
		bCorrect := true
		for j := 0; j < len(nonEqConstrain); j++ {
			if nonEqConstrain[j].Func(xPlus) > 0 {
				bCorrect = false
				break
			}
		}
		if bCorrect {
			break
		}
		s *= BETA
		floats.ScaleTo(xPlus, s, Deltax)
		floats.AddTo(xPlus, xPlus, x)
		if i == MaxRetry {
			return nil, nil, nil, ErrOverRetry
		}
	}
	// || r_t(x^+, lambda^+, nu^+)||_2 <= (1-alpha*s)*||r_t(x, lambda, nu)||_2
	floats.ScaleTo(lambdaPlus, s, DeltaLambda)
	floats.AddTo(lambdaPlus, lambdaPlus, lambda)
	floats.ScaleTo(nuPlus, s, DeltaNu)
	floats.AddTo(nuPlus, nuPlus, Nu)
	compareNorm := floats.Norm(computeRemainder(f0, x, f, Df, mat.NewDense(len(lambda), 1, lambda), mat.NewDense(len(Nu), 1, Nu), A, b, tInverse).RawMatrix().Data, 2)
	for i := 0; i <= MaxRetry; i++ {
		rMatrix := computeRemainder(f0, xPlus, f, Df, mat.NewDense(len(lambdaPlus), 1, lambdaPlus), mat.NewDense(len(nuPlus), 1, nuPlus), A, b, tInverse)
		rNorm := floats.Norm(rMatrix.RawMatrix().Data, 2)
		comparePart := (1 - ALHPHA*s) * compareNorm
		//fmt.Println("rNorm:", rNorm)
		//fmt.Println("compareNorm:", compareNorm)
		if rNorm <= comparePart {
			return xPlus, lambdaPlus, nuPlus, nil
		}
		s *= BETA
		floats.ScaleTo(xPlus, s, Deltax)
		floats.AddTo(xPlus, xPlus, x)
		floats.ScaleTo(lambdaPlus, s, DeltaLambda)
		floats.AddTo(lambdaPlus, lambdaPlus, lambda)
		floats.ScaleTo(nuPlus, s, DeltaNu)
		floats.AddTo(nuPlus, nuPlus, Nu)
		//fmt.Println("s:", s)
	}
	return nil, nil, nil, ErrOverRetry
}
