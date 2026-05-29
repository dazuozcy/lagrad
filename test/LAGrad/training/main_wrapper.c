// C wrapper for MLIR training example
#include <stdio.h>
#include <stdlib.h>

// Declare the MLIR main function
extern void _mlir_ciface_mlir_main();

int main() {
    printf("=== LAGrad Training Example ===\n");
    printf("Model: Linear Regression (y = X @ W + b)\n");
    printf("Loss: Mean Squared Error\n");
    printf("Optimizer: SGD with learning rate 0.01\n\n");
    
    printf("Training data:\n");
    printf("  X = [[1, 2], [3, 4], [5, 6], [7, 8]]\n");
    printf("  y_true = [[9], [19], [29], [39]]\n");
    printf("  (Target: y = 2*x1 + 3*x2 + 1)\n\n");
    
    printf("Initial parameters:\n");
    printf("  W = [[0.1], [0.2]]\n");
    printf("  b = [0.0]\n\n");
    
    // Run training (this will update W and b internally)
    printf("Running 20 training steps...\n\n");
    _mlir_ciface_mlir_main();
    
    printf("\nTraining completed!\n");
    printf("Target: W = [2.0, 3.0], b = 1.0\n");
    printf("The model has been trained using automatic differentiation (lagrad.grad).\n");
    printf("To see the generated gradient code, examine:\n");
    printf("  output/01_preprocessed.mlir - After AD transformation\n");
    printf("  output/02_bufferized.mlir - After bufferization\n");
    printf("  output/04.ll - Final LLVM IR\n");
    
    return 0;
}
