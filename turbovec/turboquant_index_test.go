package turbovec

import (
	"errors"
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

func tmpDir(t *testing.T) string {
	t.Helper()
	d := t.TempDir()
	return filepath.Clean(d)
}
