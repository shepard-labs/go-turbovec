package turbovec

import (
	"errors"
	"math"
	"path/filepath"
	"testing"
)

func TestNewTurboQuantIndex_Eager(t *testing.T) {
	idx, err := NewTurboQuantIndex(128, 4)
	if err != nil {
		t.Fatalf("NewTurboQuantIndex: %v", err)
	}
	defer idx.Close()
	if got, _ := idx.Dim(); got != 128 {
		t.Errorf("Dim() = %d, want 128", got)
	}
	if got := idx.BitWidth(); got != 4 {
		t.Errorf("BitWidth() = %d, want 4", got)
	}
	if got := idx.Len(); got != 0 {
		t.Errorf("Len() = %d, want 0", got)
	}
}

func TestNewTurboQuantIndex_InvalidBitWidth(t *testing.T) {
	_, err := NewTurboQuantIndex(128, 7)
	if err == nil {
		t.Fatal("want error for bit_width=7, got nil")
	}
	if !errors.Is(err, ErrInvalidBitWidth) {
		t.Errorf("err = %v, want errors.Is(ErrInvalidBitWidth)", err)
	}
}

func TestNewTurboQuantIndex_InvalidDim(t *testing.T) {
	_, err := NewTurboQuantIndex(7, 4) // not a multiple of 8
	if err == nil {
		t.Fatal("want error for dim=7, got nil")
	}
	if !errors.Is(err, ErrInvalidDim) {
		t.Errorf("err = %v, want errors.Is(ErrInvalidDim)", err)
	}
}

func TestNewLazyTurboQuantIndex(t *testing.T) {
	idx, err := NewLazyTurboQuantIndex(4)
	if err != nil {
		t.Fatalf("NewLazyTurboQuantIndex: %v", err)
	}
	defer idx.Close()
	if got, ok := idx.Dim(); ok {
		t.Errorf("Dim() committed before add: got (%d, %v), want (0, false)", got, ok)
	}
}

func TestTurboQuantIndex_Close_Idempotent(t *testing.T) {
	idx, err := NewTurboQuantIndex(64, 4)
	if err != nil {
		t.Fatalf("NewTurboQuantIndex: %v", err)
	}
	if err := idx.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := idx.Close(); err != nil {
		t.Errorf("second Close: %v, want nil", err)
	}
}

func TestTurboQuantIndex_Add_Eager(t *testing.T) {
	idx, err := NewTurboQuantIndex(8, 4)
	if err != nil {
		t.Fatalf("NewTurboQuantIndex: %v", err)
	}
	defer idx.Close()
	vecs := []float32{1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0}
	if err := idx.Add(vecs); err != nil {
		t.Fatalf("Add: %v", err)
	}
	if got := idx.Len(); got != 2 {
		t.Errorf("Len() = %d, want 2", got)
	}
}

func TestTurboQuantIndex_Add_LazyCommitsDim(t *testing.T) {
	idx, err := NewLazyTurboQuantIndex(4)
	if err != nil {
		t.Fatalf("NewLazyTurboQuantIndex: %v", err)
	}
	defer idx.Close()
	if d, ok := idx.Dim(); ok {
		t.Fatalf("Dim() committed before add: (%d, %v)", d, ok)
	}
	vecs := make([]float32, 16) // 2 vectors of dim 8
	if err := idx.AddWithDim(vecs, 8); err != nil {
		t.Fatalf("AddWithDim: %v", err)
	}
	if d, ok := idx.Dim(); !ok || d != 8 {
		t.Errorf("Dim() after first add = (%d, %v), want (8, true)", d, ok)
	}
}

func TestTurboQuantIndex_AddWithDim_DimNotMultipleOf8(t *testing.T) {
	idx, _ := NewLazyTurboQuantIndex(4)
	defer idx.Close()
	// dim=7 is not a multiple of 8; the first add on a lazy index must use a
	// multiple of 8. The upstream's AddError::DimNotMultipleOf8 surfaces as
	// ErrInvalidDim (the spec collapses both dim-shape errors into one
	// sentinel for callers).
	vecs := make([]float32, 14) // 2 vectors of dim 7
	if err := idx.AddWithDim(vecs, 7); !errors.Is(err, ErrInvalidDim) {
		t.Errorf("err = %v, want errors.Is(ErrInvalidDim)", err)
	}
}

func TestTurboQuantIndex_Add_BufferNotMultipleOfDim(t *testing.T) {
	idx, _ := NewTurboQuantIndex(8, 4)
	defer idx.Close()
	vecs := make([]float32, 7) // not a multiple of 8
	if err := idx.Add(vecs); !errors.Is(err, ErrBufferNotMultipleOfDim) {
		t.Errorf("err = %v, want errors.Is(ErrBufferNotMultipleOfDim)", err)
	}
}

func TestTurboQuantIndex_Add_InvalidInputValue(t *testing.T) {
	idx, _ := NewTurboQuantIndex(8, 4)
	defer idx.Close()
	vecs := []float32{float32(math.NaN()), 0, 0, 0, 0, 0, 0, 0}
	if err := idx.Add(vecs); !errors.Is(err, ErrInvalidInputValue) {
		t.Errorf("err = %v, want errors.Is(ErrInvalidInputValue)", err)
	}
}

func TestTurboQuantIndex_Search_Basic(t *testing.T) {
	idx, _ := NewTurboQuantIndex(8, 4)
	defer idx.Close()
	// 4 unit vectors along the first 4 of 8 axes (dim must be a multiple of 8).
	vecs := []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0,
	}
	if err := idx.Add(vecs); err != nil {
		t.Fatalf("Add: %v", err)
	}
	idx.Prepare()
	// Query close to v0.
	r, err := idx.Search([]float32{1, 0, 0, 0, 0, 0, 0, 0}, 2)
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if r.Nq != 1 || r.K != 2 {
		t.Fatalf("Nq=%d K=%d, want 1 2", r.Nq, r.K)
	}
	if len(r.Scores) != 2 || len(r.Indices) != 2 {
		t.Fatalf("len(Scores)=%d len(Indices)=%d, want 2 2", len(r.Scores), len(r.Indices))
	}
	// The first result should be index 0 with the highest score.
	if r.Indices[0] != 0 {
		t.Errorf("Indices[0] = %d, want 0", r.Indices[0])
	}
}

func TestTurboQuantIndex_SearchWithMask(t *testing.T) {
	idx, _ := NewTurboQuantIndex(8, 4)
	defer idx.Close()
	vecs := []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0,
	}
	idx.Add(vecs)
	idx.Prepare()
	// Allow only index 2.
	mask := []bool{false, false, true, false}
	r, err := idx.SearchWithMask([]float32{1, 0, 0, 0, 0, 0, 0, 0}, 2, mask)
	if err != nil {
		t.Fatalf("SearchWithMask: %v", err)
	}
	if r.K != 1 {
		t.Errorf("K = %d, want 1", r.K)
	}
	if r.Indices[0] != 2 {
		t.Errorf("Indices[0] = %d, want 2", r.Indices[0])
	}
}

func TestTurboQuantIndex_Search_WrongDim(t *testing.T) {
	idx, _ := NewTurboQuantIndex(8, 4)
	defer idx.Close()
	_, err := idx.Search([]float32{1, 0, 0, 0}, 1)
	if !errors.Is(err, ErrInvalidDim) {
		t.Errorf("err = %v, want errors.Is(ErrInvalidDim)", err)
	}
}

func TestTurboQuantIndex_SearchWithMask_LengthMismatch(t *testing.T) {
	idx, _ := NewTurboQuantIndex(8, 4)
	defer idx.Close()
	vecs := []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
	}
	idx.Add(vecs)
	idx.Prepare()
	mask := []bool{true} // length 1, but Len()=2
	_, err := idx.SearchWithMask([]float32{1, 0, 0, 0, 0, 0, 0, 0}, 1, mask)
	if err == nil {
		t.Error("want error for mask length mismatch, got nil")
	}
}

func TestTurboQuantIndex_SwapRemove(t *testing.T) {
	idx, _ := NewTurboQuantIndex(8, 4)
	defer idx.Close()
	vecs := []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
	}
	idx.Add(vecs)
	if got := idx.Len(); got != 3 {
		t.Fatalf("Len()=%d, want 3", got)
	}
	movedFrom, err := idx.SwapRemove(0)
	if err != nil {
		t.Fatalf("SwapRemove: %v", err)
	}
	if got := idx.Len(); got != 2 {
		t.Errorf("Len()=%d, want 2", got)
	}
	if movedFrom != 2 {
		t.Errorf("movedFrom=%d, want 2", movedFrom)
	}
}

func TestTurboQuantIndex_SwapRemove_OutOfRange(t *testing.T) {
	idx, _ := NewTurboQuantIndex(8, 4)
	defer idx.Close()
	vecs := []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
	}
	idx.Add(vecs)
	_, err := idx.SwapRemove(5)
	if !errors.Is(err, ErrIndexOutOfRange) {
		t.Errorf("err = %v, want errors.Is(ErrIndexOutOfRange)", err)
	}
}

func TestTurboQuantIndex_WriteLoad(t *testing.T) {
	idx, _ := NewTurboQuantIndex(8, 4)
	defer idx.Close()
	vecs := []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 0,
	}
	idx.Add(vecs)
	idx.Prepare()
	path := filepath.Join(tmpDir(t), "idx.tq")
	if err := idx.Write(path); err != nil {
		t.Fatalf("Write: %v", err)
	}
	loaded, err := LoadTurboQuantIndex(path)
	if err != nil {
		t.Fatalf("LoadTurboQuantIndex: %v", err)
	}
	defer loaded.Close()
	if got := loaded.Len(); got != 4 {
		t.Errorf("loaded Len()=%d, want 4", got)
	}
	r, err := loaded.Search([]float32{1, 0, 0, 0, 0, 0, 0, 0}, 1)
	if err != nil {
		t.Fatalf("loaded Search: %v", err)
	}
	if r.Indices[0] != 0 {
		t.Errorf("loaded Indices[0]=%d, want 0", r.Indices[0])
	}
}

func tmpDir(t *testing.T) string {
	t.Helper()
	d := t.TempDir()
	return filepath.Clean(d)
}
