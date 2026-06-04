#include <stdio.h>

void _mlir_ciface_print_loss(int step, float loss) {
    printf("  [Step %d]  loss = %12.6f\n", step, loss);
}
