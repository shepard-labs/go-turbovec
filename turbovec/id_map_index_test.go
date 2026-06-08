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
