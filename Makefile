.PHONY: all build test clean generate

UNAME_S := $(shell uname -s)

all: build

generate:
	cd turbovec-ffi && cargo build --release

build: generate
	go build ./...

test: generate
	go test ./...

clean:
	rm -rf turbovec-ffi/target
	go clean -testcache

# CI uses these. Locally you may have already initialized the submodule.
submodules:
	git submodule update --init --recursive

