// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package functions

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

const (
	TOL = float64(0.00000000001)
)

func TestLinearSum(t *testing.T) {
	p := []float64{1.0, 2.0}
	x := []float64{5.0, 10.0}
	f := NewLinearSumTradingFunction(p)
	fEvaluation := f.Func(x)
	grad := []float64{0.0, 0.0}
	f.Grad(grad, x)

	gotfEvaluation := float64(25)
	gotGrad := []float64{1, 2}

	if math.Abs(gotfEvaluation-fEvaluation) > TOL {
		t.Errorf("evaluation wrong")
	}
	if !floats.EqualApprox(grad, gotGrad, TOL) {
		t.Errorf("gradient wrong")
	}
}

func Test1GeometricMeanLogForm(t *testing.T) {
	w := []float64{4}
	x := []float64{1, 2}
	R := []float64{100}
	f := NewGeometricMeanLogFormTradingFunction(w, R, 0.01)
	fEvaluation := f.Func(x)
	grad := []float64{0.0, 0.0}
	f.Grad(grad, x)
	hess := mat.NewDense(2, 2, nil)
	f.Hess(hess, x)

	gotfEvaluation := float64(96.8221060665189)
	gotGrad := []float64{0.00007789719417298454, -0.007789719417298454}
	gotHess := []float64{5.72829646335311e-9, -5.72829646335311e-7, -5.72829646335311e-7, 0.0000572829646335311}
	if math.Abs(gotfEvaluation-fEvaluation) > TOL {
		t.Errorf("evaluation wrong")
	}
	if !floats.EqualApprox(grad, gotGrad, TOL) {
		t.Errorf("gradient wrong")
	}
	if !floats.EqualApprox(hess.RawMatrix().Data, gotHess, TOL) {
		t.Errorf("hess wrong")
	}
}

func Test2GeometricMeanLogForm(t *testing.T) {
	w := []float64{4.0, 4.0}
	x := []float64{1, 2, 3, 2}
	R := []float64{100, 90}
	f := NewGeometricMeanLogFormTradingFunction(w, R, 0.01)
	fEvaluation := f.Func(x)
	grad := []float64{0.0, 0.0, 0, 0}
	f.Grad(grad, x)
	hess := mat.NewDense(4, 4, nil)
	f.Hess(hess, x)

	gotfEvaluation := float64(298.1414603362318)
	gotGrad := []float64{0.00023947879619066185, 0.0002681443924731057, -0.023947879619066186, -0.026814439247310574}
	gotHess := []float64{1.74394695740360e-8, -6.50899098148135e-9, -1.74394695740360e-6, 6.50899098148135e-7,
		-6.50899098148135e-9, 2.18643503321189e-8, 6.50899098148135e-7, -2.18643503321189e-6,
		-1.74394695740360e-6, 6.50899098148135e-7, 0.000174394695740360, -0.0000650899098148135,
		6.50899098148135e-7, -2.18643503321189e-6, -0.0000650899098148135, 0.000218643503321189}
	if math.Abs(gotfEvaluation-fEvaluation) > TOL {
		t.Errorf("evaluation wrong")
	}
	if !floats.EqualApprox(grad, gotGrad, TOL) {
		t.Errorf("gradient wrong")
	}
	if !floats.EqualApprox(hess.RawMatrix().Data, gotHess, TOL) {
		t.Errorf("hess wrong")
	}
}
