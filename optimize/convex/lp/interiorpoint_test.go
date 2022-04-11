// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lp

import (
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/optimize/functions"
)

const convergenceTolPrimeDualMethod = 1e-5

func TestPrimalDualInteriorPointMethods(t *testing.T) {
	p := []float64{-2, -2, 0}
	f0 := functions.NewLinearSumTradingFunction(p)
	problem := &FunctionInfo{
		Func: f0.Func,
		Grad: f0.Grad,
		Hess: f0.Hess,
	}

	NonEQConstrain1 := functions.NewLinearSumTradingFunction([]float64{1, 1, -1})
	funcInfo1 := &FunctionInfo{
		Func: NonEQConstrain1.Func,
		Grad: NonEQConstrain1.Grad,
		Hess: NonEQConstrain1.Hess,
	}

	NonEQConstrain2 := functions.NewLinearSumTradingFunction([]float64{-1, 0, 0})
	funcInfo2 := &FunctionInfo{
		Func: NonEQConstrain2.Func,
		Grad: NonEQConstrain2.Grad,
		Hess: NonEQConstrain2.Hess,
	}

	NonEQConstrain3 := functions.NewLinearSumTradingFunction([]float64{0, -1, 0})
	funcInfo3 := &FunctionInfo{
		Func: NonEQConstrain3.Func,
		Grad: NonEQConstrain3.Grad,
		Hess: NonEQConstrain3.Hess,
	}

	nonEqConstrain := []*FunctionInfo{funcInfo2, funcInfo3, funcInfo1}

	A := mat.NewDense(1, 3, []float64{0, 0, 1})
	b := mat.NewDense(1, 1, []float64{3})

	// p := []float64{0, 0, 0, -1}
	// f0 := functions.NewLinearSumTradingFunction(p)
	// problem := &FunctionInfo{
	// 	Func: f0.Func,
	// 	Grad: f0.Grad,
	// 	Hess: f0.Hess,
	// }

	// delta1 := 0.9
	// delta2 := 0.87
	// nonL1 := functions.NewGeometricMeanLogFormTradingFunction([]float64{0.5, 0.5}, []float64{100, 90}, delta1)
	// nonL2 := functions.NewGeometricMeanLogFormTradingFunction([]float64{0.5, 0.5}, []float64{90, 87}, delta2)
	// funcInfo1 := &FunctionInfo{
	// 	Func: nonL1.Func,
	// 	Grad: nonL1.Grad,
	// 	Hess: nonL1.Hess,
	// }
	// funcInfo2 := &FunctionInfo{
	// 	Func: nonL2.Func,
	// 	Grad: nonL2.Grad,
	// 	Hess: nonL2.Hess,
	// }
	// nonEqConstrain := []*FunctionInfo{funcInfo1, funcInfo2}

	// A := mat.NewDense(2, 4, []float64{0, 1, 0, 0, 0, 0, 1, 0})
	// b := mat.NewDense(2, 1, []float64{0, 0})

	initailPoint := []float64{1, 1, 3}
	expected := []float64{1.5, 1.5, 3}

	result, err := PrimalDualInteriorPointMethod(problem, nonEqConstrain, A, b, initailPoint, 10)
	if err != nil {
		t.Errorf(err.Error())
	}
	if !floats.EqualApprox(result, expected, convergenceTolPrimeDualMethod) {
		t.Errorf("Not converge to expected result")
	}
}

func Test2PrimalDualInteriorPointMethods(t *testing.T) {
	p := []float64{-3, -5, 0, 0, 0}
	f0 := functions.NewLinearSumTradingFunction(p)
	problem := &FunctionInfo{
		Func: f0.Func,
		Grad: f0.Grad,
		Hess: f0.Hess,
	}

	NonEQConstrain1 := functions.NewLinearSumTradingFunction([]float64{1, 0, 1, 0, 0})
	funcInfo1 := &FunctionInfo{
		Func: NonEQConstrain1.Func,
		Grad: NonEQConstrain1.Grad,
		Hess: NonEQConstrain1.Hess,
	}

	NonEQConstrain2 := functions.NewLinearSumTradingFunction([]float64{0, 2, 0, 1, 0})
	funcInfo2 := &FunctionInfo{
		Func: NonEQConstrain2.Func,
		Grad: NonEQConstrain2.Grad,
		Hess: NonEQConstrain2.Hess,
	}

	NonEQConstrain3 := functions.NewLinearSumTradingFunction([]float64{3, 2, 0, 0, 1})
	funcInfo3 := &FunctionInfo{
		Func: NonEQConstrain3.Func,
		Grad: NonEQConstrain3.Grad,
		Hess: NonEQConstrain3.Hess,
	}

	NonEQConstrain4 := functions.NewLinearSumTradingFunction([]float64{-1, 0, 0, 0, 0})
	funcInfo4 := &FunctionInfo{
		Func: NonEQConstrain4.Func,
		Grad: NonEQConstrain4.Grad,
		Hess: NonEQConstrain4.Hess,
	}

	NonEQConstrain5 := functions.NewLinearSumTradingFunction([]float64{0, -1, 0, 0, 0})
	funcInfo5 := &FunctionInfo{
		Func: NonEQConstrain5.Func,
		Grad: NonEQConstrain5.Grad,
		Hess: NonEQConstrain5.Hess,
	}

	nonEqConstrain := []*FunctionInfo{funcInfo1, funcInfo2, funcInfo3, funcInfo4, funcInfo5}

	A := mat.NewDense(3, 5, []float64{0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1})
	b := mat.NewDense(3, 1, []float64{-3, -12, -18})

	initailPoint := []float64{1, 1, -3, -12, -18}
	expected := []float64{2, 6, -3, -12, -18}

	result, err := PrimalDualInteriorPointMethod(problem, nonEqConstrain, A, b, initailPoint, 10)
	if err != nil {
		t.Errorf(err.Error())
	}
	if !floats.EqualApprox(result, expected, convergenceTolPrimeDualMethod) {
		t.Errorf("Not converge to expected result")
	}
}
