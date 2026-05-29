mkdir build
cd build
cmake .. -Wno-dev -DMLIR_DIR=/home/zuo/code/llvm-project/build/lib/cmake/mlir/
cmake --build . -j4

