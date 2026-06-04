# LAGrad Training Example

This directory contains a complete example demonstrating how to use LAGrad for training a simple linear regression model using automatic differentiation.

## Overview

The example implements:
- **Forward Pass**: A linear layer `y = X @ W + b`
- **Loss Function**: Mean Squared Error (MSE)
- **Backward Pass**: Automatic differentiation using `lagrad.grad`
- **Optimizer**: Stochastic Gradient Descent (SGD)
- **Training Loop**: 5 training steps

## Files

- `training.mlir` - The main MLIR source file containing the model and training logic
- `run.sh` - Shell script to compile and run the example
- `pipeline.txt` - Documentation of the complete pass pipeline
- `main_wrapper.c` - C wrapper to call the MLIR code and print results
- `print_utils.c` - Utility functions for printing (not used in current version)
- `output/` - Directory containing intermediate compilation artifacts

## Model Details

### Linear Regression
- **Input**: 4 samples with 2 features each (tensor<4x2xf32>)
- **Weights**: 2x1 matrix (tensor<2x1xf32>)
- **Bias**: 1-element vector (tensor<1xf32>)
- **Output**: 4 predictions (tensor<4x1xf32>)

### Training Data
```
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y_true = [[9], [19], [29], [39]]
```

The target function is: `y = 2*x1 + 3*x2 + 1`

### Initial Parameters
```
W = [[0.1], [0.2]]
b = [0.0]
```

### Hyperparameters
- Learning rate: 0.01
- Number of training steps: 5

## Key Concepts

### 1. Automatic Differentiation with `lagrad.grad`

The `lagrad.grad` operation computes gradients using reverse-mode automatic differentiation:

```mlir
%dW, %db = lagrad.grad @forward_and_loss(%X, %y_true, %W, %b)
  {of = [2, 3]} :
  (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>)
  -> (tensor<2x1xf32>, tensor<1xf32>)
```

The `{of = [2, 3]}` attribute specifies that we want gradients with respect to arguments at indices 2 and 3 (W and b).

### 2. The `take-grads` Pass

This pass transforms `lagrad.grad` operations into concrete differentiated functions:
- Generates `__grad_forward_and_loss_arg2` and `__grad_forward_and_loss_arg1` functions
- Implements the reverse-mode AD algorithm with:
  - Forward pass to compute intermediate values
  - Reverse pass to propagate gradients
  - Activity analysis to optimize computation

### 3. Pass Pipeline

The compilation pipeline consists of:

1. **Auto-Differentiation**: `-take-grads` transforms AD operations
2. **Canonicalization**: `-canonicalize`, `-inline`, etc.
3. **Bufferization**: `-one-shot-bufferize` converts tensors to memrefs
4. **Lowering**: Convert to loops, then to LLVM dialect
5. **Translation**: MLIR → LLVM IR → Object code → Executable

See `pipeline.txt` for detailed documentation of each pass.

## Running the Example

```bash
./run.sh
```

This will:
1. Compile the MLIR code through all passes
2. Generate intermediate files in `output/`
3. Link and run the executable
4. Print training progress

## Examining Intermediate Results

The `output/` directory contains:

- `01_preprocessed.mlir` - After AD transformation (shows generated gradient functions)
- `02_bufferized.mlir` - After bufferization (shows memref operations)
- `03_llvm.mlir` - LLVM dialect (close to LLVM IR)
- `04.ll` - Final LLVM IR
- `05.o` - Object file
- `training` - Final executable

### Viewing Generated Gradients

To see the automatically generated gradient computation:

```bash
cat output/01_preprocessed.mlir | grep -A 50 "__grad_forward_and_loss"
```

This shows how LAGrad transforms the high-level `lagrad.grad` operation into concrete gradient computation code using:
- Transposed matrix multiplications for linear layer gradients
- Element-wise operations for bias gradients
- Proper handling of the MSE loss gradient

## Expected Output

```
=== LAGrad Training Example ===
Model: Linear Regression (y = X @ W + b)
Loss: Mean Squared Error
Optimizer: SGD with learning rate 0.01

Training data:
  X = [[1, 2], [3, 4], [5, 6], [7, 8]]
  y_true = [[9], [19], [29], [39]]
  (Target: y = 2*x1 + 3*x2 + 1)

Initial parameters:
  W = [[0.1], [0.2]]
  b = [0.0]

Running 5 training steps...

Training completed!
The model has been trained using automatic differentiation.
```

## Extending the Example

### Adding More Training Steps

Edit `training.mlir` and add more calls to `@train_step`:

```mlir
%W6, %b6, %loss6 = func.call @train_step(%X, %y_true, %W5, %b5, %lr)
  : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
  -> (tensor<2x1xf32>, tensor<1xf32>, f32)
```

### Changing the Model

Modify the `@forward` function to implement a different model, e.g., a 2-layer neural network:

```mlir
func.func @forward(%X: tensor<4x2xf32>, %W1: tensor<2x4xf32>, ...) -> tensor<4x1xf32> {
  // Add hidden layer with ReLU activation
  %h = linalg.matmul ins(%X, %W1) ...
  %h_relu = linalg.generic ... // ReLU
  %out = linalg.matmul ins(%h, %W2) ...
  return %out
}
```

### Using Forward-Mode AD

Replace `lagrad.grad` with `lagrad.tangent` for forward-mode differentiation:

```mlir
%dW = lagrad.tangent @forward(%X, %W, %b, %dW_seed)
  {of = [1]} :
  (tensor<4x2xf32>, tensor<2x1xf32>, tensor<1xf32>, tensor<2x1xf32>)
  -> tensor<2x1xf32>
```

## Troubleshooting

### Compilation Errors

If you see errors about missing passes, ensure you're using the correct version of `lagrad-opt`:

```bash
/home/zuo/code/lagrad-fork/build/bin/lagrad-opt --help
```

### Linking Errors

If you see undefined reference errors, make sure all object files are being linked:

```bash
gcc output/main_wrapper.o output/05.o output/print_utils.o -o output/training -lm
```

### Runtime Errors

If the program crashes, check the intermediate MLIR files to see if the transformations are correct:

```bash
cat output/01_preprocessed.mlir  # Check AD transformation
cat output/02_bufferized.mlir    # Check bufferization
```

## References

- LAGrad dialect definition: `include/LAGrad/LAGradOps.td`
- AD implementation: `lib/LAGrad/LowerGrad.cpp` (reverse-mode), `lib/LAGrad/LAGradTransforms.cpp` (forward-mode)
- More examples: `test/LAGrad/nn/` (neural network examples)
