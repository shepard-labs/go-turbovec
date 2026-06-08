#!/usr/bin/env bash
# Verify the build host has the toolchains go-turbovec needs.
# Exits non-zero on the first missing tool.

set -euo pipefail

need() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "missing required tool: $1" >&2
        exit 1
    fi
}

need go
need cargo
need cc      # any C compiler; cgo uses it
need git

echo "tools OK"
