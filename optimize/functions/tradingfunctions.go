// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package functions

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type LinearSum struct {
	scalar []float64
}

func NewLinearSumTradingFunction(p []float64) *LinearSum {
	return &LinearSum{
		scalar: p,
	}
}

func (f *LinearSum) Func(R []float64) float64 {
	if len(R) < 1 {
		panic("dimension should > 1")
	}
	result := f.scalar[0] * R[0]
	for i := 1; i < len(R); i++ {
		result += f.scalar[i] * R[i]
	}
	return result
}

func (f *LinearSum) Grad(grad, x []float64) {
	if len(x) != len(grad) {
		panic("dimension must be the same")
	}
	if len(x) != len(f.scalar) {
		panic("dimension must be the same")
	}
	if len(x) < 1 {
		panic("dimension should > 1")
	}
	for i := 0; i < len(grad); i++ {
		grad[i] = f.scalar[i]
	}
}

func (f *LinearSum) Hess(dst *mat.Dense, x []float64) {
	//if len(x) != 2 {
	//	panic("dimension of the problem must be 2")
	//}
	if len(x) != len(f.scalar) {
		panic("dimension must be the same")
	}
	if len(x) != dst.RawMatrix().Rows {
		panic("incorrect size of the Hessian")
	}
	dst.Zero()
}

type GeometricTradingFunction struct {
	R             []float64
	weightInverse []float64
	scalar        float64
	prodR         float64
}

func NewGeometricMeanLogFormTradingFunction(w, R []float64, scalar float64) *GeometricTradingFunction {
	if len(R) < 1 {
		panic("dimension should > 1")
	}
	if len(w) < 1 {
		panic("dimension should > 1")
	}
	if len(R) != len(w) {
		panic("dimension must be the same")
	}
	weightInverse := make([]float64, len(w)<<1)
	for i := 0; i < len(w); i++ {
		weightInverse[i] = 1 / w[i]
		weightInverse[len(w)+i] = weightInverse[i]
	}
	prodR := 1.0
	for i := 0; i < len(R); i++ {
		prodR *= math.Exp(math.Log(R[i]) * weightInverse[i])
	}

	return &GeometricTradingFunction{
		weightInverse: weightInverse,
		R:             R,
		scalar:        scalar,
		prodR:         prodR,
	}
}

// f := prod_i R_i^wi - prod_i (R_i - scalar*input[i]+output[i])
func (f *GeometricTradingFunction) Func(x []float64) float64 {
	if (len(f.R) << 1) != len(x) {
		panic("dimension must be the same")
	}
	mid := len(x) >> 1
	input := x[0:mid]
	output := x[mid:]
	result := f.prodR
	// R-input+output
	subSlice := make([]float64, len(f.R))
	floats.ScaleTo(subSlice, f.scalar, input)
	floats.SubTo(subSlice, f.R, subSlice)
	floats.AddTo(subSlice, subSlice, output)
	result -= computeEvaluation(f, subSlice)
	return result
}

func computeEvaluation(f *GeometricTradingFunction, x []float64) float64 {
	result := math.Exp(math.Log(x[0]) * f.weightInverse[0])
	for i := 1; i < len(x); i++ {
		result *= math.Exp(f.weightInverse[i] * math.Log(x[i]))
	}
	return result
}

func (f *GeometricTradingFunction) Grad(grad, x []float64) {
	if (len(f.R) << 1) != len(x) {
		panic("dimension must be the same")
	}
	mid := len(x) >> 1
	input := x[0:mid]
	output := x[mid:]
	totalValue := f.Func(x) - f.prodR
	for i := 0; i < mid; i++ {
		grad[i] = ((0 - f.scalar) * (totalValue * f.weightInverse[i])) / (f.R[i] - f.scalar*input[i] + output[i])
	}
	for i := 0; i < mid; i++ {
		grad[i+mid] = (0 - grad[i]) / f.scalar
	}
}

func (f *GeometricTradingFunction) Hess(dst *mat.Dense, x []float64) {
	if len(x) < 2 {
		panic("dimension of the problem must be 2")
	}
	if len(x) != dst.RawMatrix().Rows {
		panic("incorrect size of the Hessian")
	}

	mid := len(x) >> 1
	input := x[0:mid]
	output := x[mid:]
	totalValue := f.Func(x) - f.prodR

	diffDerivative := make([]float64, len(x))
	//fmt.Println(f.Grad(diffDerivative, x).RawMatrix().Data)
	for i := 0; i < mid; i++ {
		diffDerivative[i] = (0 - f.scalar) / (f.R[i] - f.scalar*input[i] + output[i])
	}
	for i := 0; i < mid; i++ {
		diffDerivative[i+mid] = (0 - diffDerivative[i]) / f.scalar
	}

	scalarSquare := f.scalar * f.scalar
	for i := 0; i < len(x); i++ {
		for j := i; j < len(x); j++ {
			if i < j {
				if j == mid+i {
					dst.Set(i, j, f.weightInverse[i]*(f.weightInverse[j]-1)*diffDerivative[i]*diffDerivative[j]*totalValue)
					dst.Set(j, i, dst.At(i, j))
					continue
				}
				dst.Set(i, j, f.weightInverse[i]*(f.weightInverse[j])*diffDerivative[i]*diffDerivative[j]*totalValue)
				dst.Set(j, i, dst.At(i, j))
				continue
			} else {
				if i < mid {
					diff := (f.R[i] - f.scalar*input[i] + output[i])
					value := f.weightInverse[i] * (f.weightInverse[i] - 1) / (diff * diff) * totalValue
					dst.Set(i, j, scalarSquare*value)
					dst.Set(i+mid, j+mid, value)
				}
			}
		}
	}
}
