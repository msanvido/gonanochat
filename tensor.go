package main

import (
	"fmt"
	"math"
)

// Tensor is a multi-dimensional array of float32 values.
// Data is stored in row-major (C) order.
type Tensor struct {
	Data  []float32
	Shape []int
}

func NewTensor(shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	return &Tensor{
		Data:  make([]float32, size),
		Shape: shape,
	}
}

func TensorFrom(data []float32, shape ...int) *Tensor {
	size := 1
	for _, s := range shape {
		size *= s
	}
	if len(data) != size {
		panic(fmt.Sprintf("tensor: data length %d != shape product %d", len(data), size))
	}
	return &Tensor{Data: data, Shape: shape}
}

func (t *Tensor) Size() int {
	s := 1
	for _, d := range t.Shape {
		s *= d
	}
	return s
}

func (t *Tensor) Ndim() int {
	return len(t.Shape)
}

// Row returns a slice of the underlying data for a 2D tensor's row.
func (t *Tensor) Row(i int) []float32 {
	cols := t.Shape[len(t.Shape)-1]
	start := i * cols
	return t.Data[start : start+cols]
}

// Clone returns a deep copy of the tensor.
func (t *Tensor) Clone() *Tensor {
	data := make([]float32, len(t.Data))
	copy(data, t.Data)
	shape := make([]int, len(t.Shape))
	copy(shape, t.Shape)
	return &Tensor{Data: data, Shape: shape}
}

// --- Vector operations (operate on flat []float32 slices) ---

// RMSNorm normalizes a vector in-place: x[i] /= rms(x)
func RMSNorm(x []float32) {
	n := len(x)
	var sumSq float64
	for i := 0; i < n; i++ {
		sumSq += float64(x[i]) * float64(x[i])
	}
	rms := float32(math.Sqrt(sumSq/float64(n) + 1e-6))
	inv := 1.0 / rms
	for i := 0; i < n; i++ {
		x[i] *= inv
	}
}

// Softmax computes softmax in-place on a float32 slice.
func Softmax(x []float32) {
	// Find max for numerical stability
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	var sum float64
	for i, v := range x {
		e := math.Exp(float64(v - maxVal))
		x[i] = float32(e)
		sum += e
	}
	invSum := float32(1.0 / sum)
	for i := range x {
		x[i] *= invSum
	}
}

// Sigmoid computes sigmoid(x).
func Sigmoid(x float32) float32 {
	return float32(1.0 / (1.0 + math.Exp(-float64(x))))
}

// MatVecMul computes y = A @ x where A is (rows, cols) and x is (cols,).
// Returns y of length rows.
func MatVecMul(A *Tensor, x []float32) []float32 {
	rows := A.Shape[0]
	cols := A.Shape[1]
	y := make([]float32, rows)
	for i := 0; i < rows; i++ {
		var sum float64
		rowStart := i * cols
		for j := 0; j < cols; j++ {
			sum += float64(A.Data[rowStart+j]) * float64(x[j])
		}
		y[i] = float32(sum)
	}
	return y
}

// MatMul computes C = A @ B where A is (M, K) and B is (K, N).
// Returns C of shape (M, N).
func MatMul(A, B *Tensor) *Tensor {
	M := A.Shape[0]
	K := A.Shape[1]
	N := B.Shape[1]
	C := NewTensor(M, N)
	for i := 0; i < M; i++ {
		for j := 0; j < N; j++ {
			var sum float64
			for k := 0; k < K; k++ {
				sum += float64(A.Data[i*K+k]) * float64(B.Data[k*N+j])
			}
			C.Data[i*N+j] = float32(sum)
		}
	}
	return C
}

// BatchMatVecMul computes y[b] = A[b] @ x[b] for batched 2D matrix @ vector.
// A has shape (batch, rows, cols), x has shape (batch, cols).
// Returns y of shape (batch, rows).
func BatchMatVecMul(A []float32, aRows, aCols int, x []float32, batch int) []float32 {
	y := make([]float32, batch*aRows)
	for b := 0; b < batch; b++ {
		aOff := b * aRows * aCols
		xOff := b * aCols
		yOff := b * aRows
		for i := 0; i < aRows; i++ {
			var sum float64
			for j := 0; j < aCols; j++ {
				sum += float64(A[aOff+i*aCols+j]) * float64(x[xOff+j])
			}
			y[yOff+i] = float32(sum)
		}
	}
	return y
}

// VecAdd computes dst = a + b element-wise.
func VecAdd(dst, a, b []float32) {
	for i := range dst {
		dst[i] = a[i] + b[i]
	}
}

// VecScaleAdd computes dst = alpha*a + beta*b element-wise.
func VecScaleAdd(dst []float32, alpha float32, a []float32, beta float32, b []float32) {
	for i := range dst {
		dst[i] = alpha*a[i] + beta*b[i]
	}
}

// VecScale multiplies a vector by a scalar in-place.
func VecScale(x []float32, s float32) {
	for i := range x {
		x[i] *= s
	}
}

// VecMulAdd computes dst[i] += scale * src[i].
func VecMulAdd(dst []float32, scale float32, src []float32) {
	for i := range dst {
		dst[i] += scale * src[i]
	}
}

// VecDot computes the dot product of two vectors.
func VecDot(a, b []float32) float32 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return float32(sum)
}

// ReLUSquared applies relu(x)^2 in-place.
func ReLUSquared(x []float32) {
	for i, v := range x {
		if v > 0 {
			x[i] = v * v
		} else {
			x[i] = 0
		}
	}
}

// TanhSoftcap applies softcap * tanh(x / softcap) in-place.
func TanhSoftcap(x []float32, softcap float32) {
	inv := 1.0 / float64(softcap)
	sc := float64(softcap)
	for i, v := range x {
		x[i] = float32(sc * math.Tanh(float64(v)*inv))
	}
}

// ApplyRotaryEmb applies rotary position embedding to a single head vector.
// x has length head_dim. cos/sin have length head_dim/2 for the given position.
func ApplyRotaryEmb(x []float32, cos, sin []float32) {
	d := len(x) / 2
	for i := 0; i < d; i++ {
		x0 := x[i]
		x1 := x[d+i]
		x[i] = x0*cos[i] + x1*sin[i]
		x[d+i] = x0*(-sin[i]) + x1*cos[i]
	}
}

// TopKSample samples from the top-k logits with temperature.
// Returns the sampled index.
func TopKSample(logits []float32, topK int, temperature float32, rng *RNG) int {
	n := len(logits)
	if topK <= 0 || topK > n {
		topK = n
	}

	// Find top-k indices using partial sort
	type indexVal struct {
		idx int
		val float32
	}
	topItems := make([]indexVal, topK)
	// Initialize with first k items
	for i := 0; i < topK; i++ {
		topItems[i] = indexVal{i, logits[i]}
	}
	// Find the minimum in topItems
	minIdx := 0
	for i := 1; i < topK; i++ {
		if topItems[i].val < topItems[minIdx].val {
			minIdx = i
		}
	}
	// Scan remaining items
	for i := topK; i < n; i++ {
		if logits[i] > topItems[minIdx].val {
			topItems[minIdx] = indexVal{i, logits[i]}
			// Find new min
			minIdx = 0
			for j := 1; j < topK; j++ {
				if topItems[j].val < topItems[minIdx].val {
					minIdx = j
				}
			}
		}
	}

	if temperature == 0 {
		// Greedy: return argmax
		bestIdx := 0
		for i := 1; i < topK; i++ {
			if topItems[i].val > topItems[bestIdx].val {
				bestIdx = i
			}
		}
		return topItems[bestIdx].idx
	}

	// Apply temperature and softmax to top-k values
	probs := make([]float32, topK)
	maxVal := topItems[0].val
	for _, item := range topItems {
		if item.val > maxVal {
			maxVal = item.val
		}
	}
	var sum float64
	invT := 1.0 / float64(temperature)
	for i, item := range topItems {
		e := math.Exp(float64(item.val-maxVal) * invT)
		probs[i] = float32(e)
		sum += e
	}
	invSum := float32(1.0 / sum)
	for i := range probs {
		probs[i] *= invSum
	}

	// Sample from distribution
	r := rng.Float32()
	var cumProb float32
	for i, p := range probs {
		cumProb += p
		if r < cumProb {
			return topItems[i].idx
		}
	}
	return topItems[topK-1].idx
}

// RNG is a simple PRNG (xoshiro128+) for reproducible sampling.
type RNG struct {
	s [4]uint32
}

func NewRNG(seed uint64) *RNG {
	r := &RNG{}
	// SplitMix64 to seed the state
	z := seed
	for i := 0; i < 4; i++ {
		z += 0x9e3779b97f4a7c15
		z2 := (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
		z2 = (z2 ^ (z2 >> 27)) * 0x94d049bb133111eb
		z2 = z2 ^ (z2 >> 31)
		r.s[i] = uint32(z2)
	}
	return r
}

func (r *RNG) Uint32() uint32 {
	result := r.s[0] + r.s[3]
	t := r.s[1] << 9
	r.s[2] ^= r.s[0]
	r.s[3] ^= r.s[1]
	r.s[1] ^= r.s[2]
	r.s[0] ^= r.s[3]
	r.s[2] ^= t
	r.s[3] = (r.s[3] << 11) | (r.s[3] >> 21)
	return result
}

func (r *RNG) Float32() float32 {
	return float32(r.Uint32()>>8) / float32(1<<24)
}

// PrecomputeRotaryEmbeddings computes cos and sin tables for rotary embeddings.
// Returns cos, sin each of shape (maxSeqLen, headDim/2).
func PrecomputeRotaryEmbeddings(maxSeqLen, headDim int, base float64) (cos, sin *Tensor) {
	halfDim := headDim / 2
	cos = NewTensor(maxSeqLen, halfDim)
	sin = NewTensor(maxSeqLen, halfDim)
	for pos := 0; pos < maxSeqLen; pos++ {
		for i := 0; i < halfDim; i++ {
			freq := 1.0 / math.Pow(base, float64(2*i)/float64(headDim))
			angle := float64(pos) * freq
			cos.Data[pos*halfDim+i] = float32(math.Cos(angle))
			sin.Data[pos*halfDim+i] = float32(math.Sin(angle))
		}
	}
	return cos, sin
}
