#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<(d0) -> (d0)>
module {
  func.func private @print_training_info(i32, f32, f32, f32, f32) attributes {llvm.emit_c_interface}
  func.func @forward(%arg0: tensor<4x2xf32>, %arg1: tensor<2x1xf32>, %arg2: tensor<1xf32>) -> tensor<4x1xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<4x1xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<4x1xf32>) -> tensor<4x1xf32>
    %2 = linalg.matmul ins(%arg0, %arg1 : tensor<4x2xf32>, tensor<2x1xf32>) outs(%1 : tensor<4x1xf32>) -> tensor<4x1xf32>
    %3 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%2, %arg2 : tensor<4x1xf32>, tensor<1xf32>) outs(%1 : tensor<4x1xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<4x1xf32>
    return %3 : tensor<4x1xf32>
  }

  func.func @mse_loss(%arg0: tensor<4x1xf32>, %arg1: tensor<4x1xf32>) -> tensor<f32> {
    %cst = arith.constant dense<4.000000e+00> : tensor<f32>
    %0 = arith.subf %arg0, %arg1 : tensor<4x1xf32>
    %1 = arith.mulf %0, %0 : tensor<4x1xf32>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<f32>
    %reduced = linalg.reduce ins(%1 : tensor<4x1xf32>) outs(%cst_0 : tensor<f32>) dimensions = [0, 1] 
      (%in: f32, %init: f32) {
        %3 = arith.addf %in, %init : f32
        linalg.yield %3 : f32
      }
    %2 = arith.divf %reduced, %cst : tensor<f32>
    return %2 : tensor<f32>
  }

  func.func @forward_and_loss(%arg0: tensor<4x2xf32>, %arg1: tensor<4x1xf32>, %arg2: tensor<2x1xf32>, %arg3: tensor<1xf32>) -> tensor<f32> {
    %0 = call @forward(%arg0, %arg2, %arg3) : (tensor<4x2xf32>, tensor<2x1xf32>, tensor<1xf32>) -> tensor<4x1xf32>
    %1 = call @mse_loss(%0, %arg1) : (tensor<4x1xf32>, tensor<4x1xf32>) -> tensor<f32>
    return %1 : tensor<f32>
  }

  func.func @compute_loss_and_gradients(%arg0: tensor<4x2xf32>, %arg1: tensor<4x1xf32>, %arg2: tensor<2x1xf32>, %arg3: tensor<1xf32>) -> (tensor<2x1xf32>, tensor<1xf32>, tensor<f32>) {
    %0:3 = lagrad.grad @forward_and_loss(%arg0, %arg1, %arg2, %arg3) {of = [2, 3], return_primal} : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>) -> (tensor<2x1xf32>, tensor<1xf32>, tensor<f32>)
    return %0#0, %0#1, %0#2 : tensor<2x1xf32>, tensor<1xf32>, tensor<f32>
  }

  func.func @sgd_update(%arg0: tensor<2x1xf32>, %arg1: tensor<1xf32>, %arg2: tensor<2x1xf32>, %arg3: tensor<1xf32>, %arg4: f32) -> (tensor<2x1xf32>, tensor<1xf32>) attributes {no_inline} {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg2 : tensor<2x1xf32>, tensor<2x1xf32>) outs(%arg0 : tensor<2x1xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %arg4, %in_0 : f32
      %3 = arith.subf %in, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<2x1xf32>
    %1 = linalg.generic {indexing_maps = [#map2, #map2, #map2], iterator_types = ["parallel"]} ins(%arg1, %arg3 : tensor<1xf32>, tensor<1xf32>) outs(%arg1 : tensor<1xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %arg4, %in_0 : f32
      %3 = arith.subf %in, %2 : f32
      linalg.yield %3 : f32
    } -> tensor<1xf32>
    return %0, %1 : tensor<2x1xf32>, tensor<1xf32>
  }

  func.func @train_step(%arg0: tensor<4x2xf32>, %arg1: tensor<4x1xf32>, %arg2: tensor<2x1xf32>, %arg3: tensor<1xf32>, %arg4: f32) -> (tensor<2x1xf32>, tensor<1xf32>, tensor<f32>) attributes {no_inline} {
    %0:3 = call @compute_loss_and_gradients(%arg0, %arg1, %arg2, %arg3) : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>) -> (tensor<2x1xf32>, tensor<1xf32>, tensor<f32>)
    %1:2 = call @sgd_update(%arg2, %arg3, %0#0, %0#1, %arg4) : (tensor<2x1xf32>, tensor<1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32) -> (tensor<2x1xf32>, tensor<1xf32>)
    return %1#0, %1#1, %0#2 : tensor<2x1xf32>, tensor<1xf32>, tensor<f32>
  }

  func.func @mlir_main() attributes {llvm.emit_c_interface} {
    %cst = arith.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00], [7.000000e+00, 8.000000e+00]]> : tensor<4x2xf32>
    %cst_0 = arith.constant dense<[[9.000000e+00], [1.900000e+01], [2.900000e+01], [3.900000e+01]]> : tensor<4x1xf32>
    %cst_1 = arith.constant dense<[[1.000000e-01], [2.000000e-01]]> : tensor<2x1xf32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<1xf32>
    %cst_3 = arith.constant 0.00999999977 : f32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c21 = arith.constant 21 : index
    %0:2 = scf.for %arg0 = %c1 to %c21 step %c1 iter_args(%arg1 = %cst_1, %arg2 = %cst_2) -> (tensor<2x1xf32>, tensor<1xf32>) {
      %2:3 = func.call @train_step(%cst, %cst_0, %arg1, %arg2, %cst_3) : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32) -> (tensor<2x1xf32>, tensor<1xf32>, tensor<f32>)
      %extracted_7 = tensor.extract %2#2[] : tensor<f32>
      %extracted_8 = tensor.extract %2#0[%c0, %c0] : tensor<2x1xf32>
      %extracted_9 = tensor.extract %2#0[%c1, %c0] : tensor<2x1xf32>
      %extracted_10 = tensor.extract %2#1[%c0] : tensor<1xf32>
      %3 = arith.index_cast %arg0 : index to i32
      func.call @print_training_info(%3, %extracted_7, %extracted_8, %extracted_9, %extracted_10) : (i32, f32, f32, f32, f32) -> ()
      scf.yield %2#0, %2#1 : tensor<2x1xf32>, tensor<1xf32>
    }
    %1 = call @forward_and_loss(%cst, %cst_0, %0#0, %0#1) : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>) -> tensor<f32>
    %extracted = tensor.extract %1[] : tensor<f32>
    %extracted_4 = tensor.extract %0#0[%c0, %c0] : tensor<2x1xf32>
    %extracted_5 = tensor.extract %0#0[%c1, %c0] : tensor<2x1xf32>
    %extracted_6 = tensor.extract %0#1[%c0] : tensor<1xf32>
    %c21_i32 = arith.constant 21 : i32
    call @print_training_info(%c21_i32, %extracted, %extracted_4, %extracted_5, %extracted_6) : (i32, f32, f32, f32, f32) -> ()
    return
  }
}
