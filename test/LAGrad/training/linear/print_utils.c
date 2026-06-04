// C helper functions for MLIR training example
#include <stdio.h>

// This function is called by the MLIR-generated wrapper.
// Signature must match: (i32, float, float, float, float) -> void
void _mlir_ciface_print_training_info(int step, float loss, float w0, float w1, float b0) {
    printf("  [Step %d]  loss = %12.6f  |  W = [%8.5f, %8.5f]  b = %8.5f\n",
           step, loss, w0, w1, b0);
}
