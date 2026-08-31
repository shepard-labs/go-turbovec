.PHONY: all build test clean generate submodules

UNAME_S := $(shell uname -s)

all: build

generate: submodules
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
	git submodule update --init --filter=blob:none vendor/turbovec
	git -C vendor/turbovec sparse-checkout set --no-cone /turbovec/ /LICENSE /README.md
