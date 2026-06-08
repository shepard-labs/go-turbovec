package turbovec

import (
	"errors"
	"testing"
)

func TestNewIdMapIndex_Eager(t *testing.T) {
	idx, err := NewIdMapIndex(128, 4)
	if err != nil {
		t.Fatalf("NewIdMapIndex: %v", err)
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

func TestNewIdMapIndex_InvalidBitWidth(t *testing.T) {
	_, err := NewIdMapIndex(128, 7)
	if !errors.Is(err, ErrInvalidBitWidth) {
		t.Errorf("err = %v, want errors.Is(ErrInvalidBitWidth)", err)
	}
}

func TestNewLazyIdMapIndex(t *testing.T) {
	idx, err := NewLazyIdMapIndex(4)
	if err != nil {
		t.Fatalf("NewLazyIdMapIndex: %v", err)
	}
	defer idx.Close()
	if _, ok := idx.Dim(); ok {
		t.Error("Dim() committed before add")
	}
}

func TestIdMapIndex_Close_Idempotent(t *testing.T) {
	idx, _ := NewIdMapIndex(64, 4)
	if err := idx.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := idx.Close(); err != nil {
		t.Errorf("second Close: %v, want nil", err)
	}
}

func TestIdMapIndex_AddWithIDs(t *testing.T) {
	idx, _ := NewIdMapIndex(8, 4)
	defer idx.Close()
	vecs := []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
	}
	ids := []uint64{100, 200}
	if err := idx.AddWithIDs(vecs, ids); err != nil {
		t.Fatalf("AddWithIDs: %v", err)
	}
	if got := idx.Len(); got != 2 {
		t.Errorf("Len()=%d, want 2", got)
	}
	if !idx.Contains(100) || !idx.Contains(200) {
		t.Error("Contains(100/200) = false, want true")
	}
	if idx.Contains(999) {
		t.Error("Contains(999) = true, want false")
	}
}

func TestIdMapIndex_AddWithIDsWithDim(t *testing.T) {
	idx, _ := NewLazyIdMapIndex(4)
	defer idx.Close()
	vecs := make([]float32, 16)
	ids := []uint64{1, 2}
	if err := idx.AddWithIDsWithDim(vecs, ids, 8); err != nil {
		t.Fatalf("AddWithIDsWithDim: %v", err)
	}
	if d, ok := idx.Dim(); !ok || d != 8 {
		t.Errorf("Dim() = (%d, %v), want (8, true)", d, ok)
	}
}

func TestIdMapIndex_AddWithIDs_DuplicateID(t *testing.T) {
	idx, _ := NewIdMapIndex(8, 4)
	defer idx.Close()
	vecs := []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
	}
	ids := []uint64{1, 1}
	if err := idx.AddWithIDs(vecs, ids); !errors.Is(err, ErrDuplicateID) {
		t.Errorf("err = %v, want errors.Is(ErrDuplicateID)", err)
	}
}

func TestIdMapIndex_AddWithIDs_CountMismatch(t *testing.T) {
	idx, _ := NewIdMapIndex(8, 4)
	defer idx.Close()
	vecs := []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
	}
	ids := []uint64{1} // 1 id, 2 vectors
	if err := idx.AddWithIDs(vecs, ids); !errors.Is(err, ErrIDsCountMismatch) {
		t.Errorf("err = %v, want errors.Is(ErrIDsCountMismatch)", err)
	}
}

func TestIdMapIndex_Remove(t *testing.T) {
	idx, _ := NewIdMapIndex(8, 4)
	defer idx.Close()
	vecs := []float32{
		1, 0, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0, 0,
	}
	ids := []uint64{1, 2}
	idx.AddWithIDs(vecs, ids)
	if !idx.Remove(1) {
		t.Error("Remove(1) = false, want true")
	}
	if idx.Remove(1) {
		t.Error("Remove(1) again = true, want false")
	}
	if idx.Contains(1) {
		t.Error("Contains(1) after Remove = true, want false")
	}
	if got := idx.Len(); got != 1 {
		t.Errorf("Len()=%d, want 1", got)
	}
}
