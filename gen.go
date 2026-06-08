//go:build ignore
// +build ignore

package main

// Build the C-ABI shim so cgo has a lib to link against.
//go:generate go run gen.go
func main() {
	// Intentionally empty: `make generate` (cargo build) is what actually
	// builds the lib. This file exists so `go generate ./...` is a no-op
	// and developers are pointed at the Makefile.
}
