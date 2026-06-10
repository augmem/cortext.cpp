#!/usr/bin/env sh

set -eu

prefix=${1:?install prefix is required}
build_dir=${2:-"$PWD/build/install-smoke"}

if [ ! -f "$prefix/lib/cmake/objstore/objstoreConfig.cmake" ]; then
  echo "missing objstoreConfig.cmake in install prefix: $prefix" >&2
  exit 1
fi

if [ -e "$prefix/lib/libunity.a" ] || [ -d "$prefix/include/unity" ] || [ -d "$prefix/lib/cmake/unity" ]; then
  echo "install prefix leaked Unity test artifacts: $prefix" >&2
  exit 1
fi

rm -rf "$build_dir"

cmake -S "$PWD/tests/downstream_smoke" -B "$build_dir" -DCMAKE_PREFIX_PATH="$prefix"
cmake --build "$build_dir"
ctest --test-dir "$build_dir" --output-on-failure
