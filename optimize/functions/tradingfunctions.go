// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package functions

import (
	"math"

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

func (f *LinearSum) Hess(dst *mat.SymDense, x []float64) {
	if len(x) != 2 {
		panic("dimension of the problem must be 2")
	}
	if len(x) != len(f.scalar) {
		panic("dimension must be the same")
	}
	if len(x) != dst.SymmetricDim() {
		panic("incorrect size of the Hessian")
	}
	dst.Zero()
}

type GeometricMeanLogForm struct {
	weightInverse []float64
}

func NewGeometricMeanLogFormTradingFunction(w []float64) *GeometricMeanLogForm {
	weightInverse := make([]float64, len(w))
	for i := 0; i < len(w); i++ {
		weightInverse[i] = 1 / w[i]
	}
	return &GeometricMeanLogForm{
		weightInverse: weightInverse,
	}
}

func (f *GeometricMeanLogForm) Func(R []float64) float64 {
	if len(R) < 1 {
		panic("dimension should > 1")
	}
	result := math.Exp(math.Log(R[0]) * f.weightInverse[0])
	for i := 1; i < len(R); i++ {
		result *= math.Exp(f.weightInverse[i] * math.Log(R[i]))
	}
	return result
}

func (f *GeometricMeanLogForm) Grad(grad, R []float64) {
	if len(R) != len(grad) {
		panic("dimension must be the same")
	}
	if len(R) != len(f.weightInverse) {
		panic("dimension must be the same")
	}
	if len(R) < 1 {
		panic("dimension should > 1")
	}
	totalValue := f.Func(R)
	for i := 0; i < len(grad); i++ {
		grad[i] = (totalValue * f.weightInverse[i]) / R[i]
	}
}

func (f *GeometricMeanLogForm) Hess(dst *mat.SymDense, x []float64) {
	if len(x) != 2 {
		panic("dimension of the problem must be 2")
	}
	if len(x) != len(f.weightInverse) {
		panic("dimension must be the same")
	}
	if len(x) != dst.SymmetricDim() {
		panic("incorrect size of the Hessian")
	}

	n := dst.SymmetricDim()
	totalValue := f.Func(x)

	diffDerivative := make([]float64, n)
	for i := 0; i < n; i++ {
		diffDerivative[i] = f.weightInverse[i] / x[i]
	}
	for i := 0; i < n; i++ {
		for j := i; j < n; j++ {
			if i != j {
				dst.SetSym(i, j, diffDerivative[i]*diffDerivative[j]*totalValue)
			} else {
				dst.SetSym(i, j, f.weightInverse[i]*(f.weightInverse[i]-1)*diffDerivative[i]*diffDerivative[i]*totalValue)
			}
		}
	}
}
