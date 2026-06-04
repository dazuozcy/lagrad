#!/bin/bash
set -e

LAGRAD_OPT=/home/zuo/code/lagrad-fork/build/bin/lagrad-opt
MLIR_CPU_RUNNER=/home/zuo/code/llvm-project/build/bin/mlir-cpu-runner
MLIR_TRANSLATE=/home/zuo/code/llvm-project/build/bin/mlir-translate
LLC=/home/zuo/code/llvm-project/build/bin/llc
CC=gcc

INPUT=train_pipeline.mlir
OUTPUT_DIR=output

mkdir -p $OUTPUT_DIR

echo "=== Step 0: inline ==="
$LAGRAD_OPT $INPUT \
  -inline="inlining-threshold=10000" \
  -canonicalize \
  -linalg-canonicalize \
  -standalone-dce \
  -symbol-dce \
  -canonicalize \
  -cse \
  -o $OUTPUT_DIR/00_inlined.mlir

echo "=== Step 1: AD + canonicalization ==="
$LAGRAD_OPT $OUTPUT_DIR/00_inlined.mlir \
  -take-grads \
  -canonicalize \
  -linalg-canonicalize \
  -standalone-dce \
  -symbol-dce \
  -canonicalize \
  -cse \
  -o $OUTPUT_DIR/01_autograded.mlir

echo "=== Step 2: Bufferization ==="
$LAGRAD_OPT $OUTPUT_DIR/01_autograded.mlir \
  -convert-elementwise-to-linalg \
  -one-shot-bufferize="bufferize-function-boundaries" \
  -convert-bufferization-to-memref \
  -standalone-bufferize \
  -buffer-hoisting \
  -buffer-loop-hoisting \
  -standalone-loop-hoisting \
  -promote-buffers-to-stack \
  -canonicalize \
  -o $OUTPUT_DIR/02_bufferized.mlir

echo "=== Step 3: Lower to LLVM ==="
$LAGRAD_OPT $OUTPUT_DIR/02_bufferized.mlir \
  -convert-linalg-to-loops \
  -lower-affine \
  -convert-scf-to-cf \
  -expand-strided-metadata \
  -convert-cf-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-math-to-llvm \
  -convert-math-to-libm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o $OUTPUT_DIR/03_llvm.mlir

echo "=== Step 4: Translate to LLVM IR ==="
$MLIR_TRANSLATE --mlir-to-llvmir $OUTPUT_DIR/03_llvm.mlir -o $OUTPUT_DIR/04.ll

echo "=== Step 5: Compile to object file ==="
$LLC -filetype=obj -relocation-model=pic $OUTPUT_DIR/04.ll -o $OUTPUT_DIR/05.o

echo "=== Step 5b: Compile C wrappers ==="
$CC -c -fPIC print_utils.c -o $OUTPUT_DIR/print_utils.o
$CC -c -fPIC main_wrapper.c -o $OUTPUT_DIR/main_wrapper.o

echo "=== Step 6: Link executable ==="
$CC $OUTPUT_DIR/main_wrapper.o $OUTPUT_DIR/05.o $OUTPUT_DIR/print_utils.o -o $OUTPUT_DIR/train -lm

echo "=== Step 7: Run training ==="
./$OUTPUT_DIR/train
echo "Training completed successfully!"
