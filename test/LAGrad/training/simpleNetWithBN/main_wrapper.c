#include <stdio.h>

extern void _mlir_ciface_mlir_main();

int main() {
    printf("=== SimpleNet with BatchNorm Training ===\n");
    printf("Running 4 training steps...\n\n");
    _mlir_ciface_mlir_main();
    printf("\nTraining completed!\n");
    return 0;
}
