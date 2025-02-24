#!/usr/bin/env bash

set -e

ORG_PATH=$(pwd)
SCRIPT_BASE_DIR=$(cd "$(dirname $0)"; pwd)
cd "${SCRIPT_BASE_DIR}"

if [ ! -d build ]; then
    mkdir build
fi

cd build

BUILD_TYPE="Debug"
if [ -n "$1" ]; then
    BUILD_TYPE="$1"
fi

cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" ..

cmake --build .

cd "${ORG_PATH}"