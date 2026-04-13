package main

import "math"

// AdamW implements the AdamW optimizer (decoupled weight decay).
type AdamW struct {
	LR          float32
	Beta1       float32
	Beta2       float32
	Eps         float32
	WeightDecay float32

	// Per-parameter state
	M map[string][]float32 // first moment (mean)
	V map[string][]float32 // second moment (variance)
	T int                   // step count
}

// NewAdamW creates a new AdamW optimizer.
func NewAdamW(lr, beta1, beta2, eps, weightDecay float32) *AdamW {
	return &AdamW{
		LR:          lr,
		Beta1:       beta1,
		Beta2:       beta2,
		Eps:         eps,
		WeightDecay: weightDecay,
		M:           make(map[string][]float32),
		V:           make(map[string][]float32),
		T:           0,
	}
}

// Step performs one optimizer step. params maps name -> weight data, grads maps name -> gradient data.
func (opt *AdamW) Step(params map[string][]float32, grads map[string]*Tensor) {
	opt.T++
	t := opt.T

	// Bias correction factors
	bc1 := float32(1.0 - math.Pow(float64(opt.Beta1), float64(t)))
	bc2 := float32(1.0 - math.Pow(float64(opt.Beta2), float64(t)))

	for name, param := range params {
		grad, ok := grads[name]
		if !ok || grad == nil {
			continue
		}
		n := len(param)
		if n == 0 {
			continue
		}

		// Lazily initialize state
		if _, exists := opt.M[name]; !exists {
			opt.M[name] = make([]float32, n)
			opt.V[name] = make([]float32, n)
		}

		m := opt.M[name]
		v := opt.V[name]
		g := grad.Data

		for i := 0; i < n; i++ {
			// Update biased first moment estimate
			m[i] = opt.Beta1*m[i] + (1-opt.Beta1)*g[i]
			// Update biased second moment estimate
			v[i] = opt.Beta2*v[i] + (1-opt.Beta2)*g[i]*g[i]
			// Compute bias-corrected estimates
			mHat := m[i] / bc1
			vHat := v[i] / bc2
			// Update parameter
			param[i] -= opt.LR * (mHat/(float32(math.Sqrt(float64(vHat)))+opt.Eps) + opt.WeightDecay*param[i])
		}
	}
}

// LRSchedule computes learning rate with warmup and cosine decay.
//
//	step: current step (0-indexed)
//	warmupSteps: number of warmup steps
//	totalSteps: total training steps
//	maxLR: peak learning rate
//	minLR: minimum learning rate (at end of cosine decay)
func LRSchedule(step, warmupSteps, totalSteps int, maxLR, minLR float32) float32 {
	if step < warmupSteps {
		// Linear warmup
		return minLR + (maxLR-minLR)*float32(step)/float32(warmupSteps)
	}
	// Cosine decay
	progress := float32(step-warmupSteps) / float32(totalSteps-warmupSteps)
	if progress > 1.0 {
		progress = 1.0
	}
	cosVal := float32(math.Cos(float64(progress) * math.Pi * 0.5))
	return minLR + (maxLR-minLR)*cosVal
}
