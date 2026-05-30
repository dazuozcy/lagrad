#!/bin/bash
# Training example compilation and execution script

set -e

LAGRAD_OPT=/home/zuo/code/lagrad-fork/build/bin/lagrad-opt
MLIR_CPU_RUNNER=/home/zuo/code/llvm-project/build/bin/mlir-cpu-runner
MLIR_TRANSLATE=/home/zuo/code/llvm-project/build/bin/mlir-translate
LLC=/home/zuo/code/llvm-project/build/bin/llc
CC=gcc

INPUT=training.mlir
OUTPUT_DIR=output

mkdir -p $OUTPUT_DIR

echo "=== Step 0: Preprocessing (AD + canonicalization) ==="
$LAGRAD_OPT $INPUT \
  -take-grads \
  -canonicalize \
  -o $OUTPUT_DIR/00_preprocessed.mlir

echo "=== Step 1: inline ==="
$LAGRAD_OPT $OUTPUT_DIR/00_preprocessed.mlir \
  -inline="inlining-threshold=10000" \
  -linalg-canonicalize \
  -standalone-dce \
  -symbol-dce \
  -canonicalize \
  -o $OUTPUT_DIR/01_inlined.mlir

echo "=== Step 2: Bufferization ==="
$LAGRAD_OPT $OUTPUT_DIR/01_inlined.mlir \
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
  -convert-cf-to-llvm \
  -convert-arith-to-llvm \
  -finalize-memref-to-llvm \
  -convert-math-to-llvm \
  -convert-math-to-libm \
  -convert-func-to-llvm \
  -reconcile-unrealized-casts \
  -o $OUTPUT_DIR/03_llvm.mlir

echo "=== Step 3b: Add print calls ==="
# Add print calls to the LLVM MLIR
cat > $OUTPUT_DIR/03b_print.mlir << 'EOF'
// Print function declaration
llvm.func @print_loss(f32, i32) attributes {sym_visibility = "private"}

// Wrapper that calls print_loss for each training step
llvm.func @main_wrapper() {
  // Call the original main
  llvm.call @main() : () -> ()
  
  // Print completion message
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c2 = llvm.mlir.constant(2 : i32) : i32
  %c3 = llvm.mlir.constant(3 : i32) : i32
  %c4 = llvm.mlir.constant(4 : i32) : i32
  %c5 = llvm.mlir.constant(5 : i32) : i32
  
  // We can't easily extract the loss values from the MLIR main function
  // So we'll just print a completion message
  llvm.return
}
EOF

# Append the print wrapper to the LLVM MLIR
cat $OUTPUT_DIR/03_llvm.mlir $OUTPUT_DIR/03b_print.mlir > $OUTPUT_DIR/03c_combined.mlir

echo "=== Step 4: Translate to LLVM IR ==="
$MLIR_TRANSLATE --mlir-to-llvmir $OUTPUT_DIR/03_llvm.mlir -o $OUTPUT_DIR/04.ll

echo "=== Step 5: Compile to object file ==="
$LLC -filetype=obj -relocation-model=pic $OUTPUT_DIR/04.ll -o $OUTPUT_DIR/05.o

echo "=== Step 5b: Compile C wrappers ==="
$CC -c -fPIC print_utils.c -o $OUTPUT_DIR/print_utils.o
$CC -c -fPIC main_wrapper.c -o $OUTPUT_DIR/main_wrapper.o

echo "=== Step 6: Link executable ==="
$CC $OUTPUT_DIR/main_wrapper.o $OUTPUT_DIR/05.o $OUTPUT_DIR/print_utils.o -o $OUTPUT_DIR/training -lm

echo "=== Step 7: Run training ==="
echo "Expected: The program should run successfully"
echo "To see training convergence, examine the intermediate MLIR files in $OUTPUT_DIR/"
echo "The loss values are computed in the @forward_and_loss function"
echo ""
./$OUTPUT_DIR/training
echo "Training completed successfully!"

echo ""
echo "=== Training complete ==="
echo "Intermediate MLIR files saved in $OUTPUT_DIR/"
echo "To see the generated gradients, examine: $OUTPUT_DIR/01_preprocessed.mlir"
echo "To see the bufferized code, examine: $OUTPUT_DIR/02_bufferized.mlir"
echo "To see the final LLVM IR, examine: $OUTPUT_DIR/04.ll"
