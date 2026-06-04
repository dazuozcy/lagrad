#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1)>
#map2 = affine_map<() -> ()>
#map3 = affine_map<(d0, d1) -> (0, d1)>
#map4 = affine_map<(d0) -> (d0)>
module {
  func.func @forward(%arg0: tensor<10x5xf32>, %arg1: tensor<10xf32>, %arg2: tensor<10xf32>, %arg3: tensor<10xf32>, %arg4: tensor<1x10xf32>, %arg5: tensor<i64>, %arg6: tensor<10xf32>, %arg7: tensor<10xf32>, %arg8: tensor<800x5xf32>) -> (tensor<i64>, tensor<10xf32>, tensor<10xf32>, tensor<800xf32>) {
    %c1_i64 = arith.constant 1 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 0.000000e+00 : f64
    %cst_1 = arith.constant dense<-0.0220023226> : tensor<1xf32>
    %cst_2 = arith.constant 1.0012515644555695 : f64
    %cst_3 = arith.constant 9.000000e-01 : f64
    %cst_4 = arith.constant 1.000000e-01 : f64
    %cst_5 = arith.constant 1.000000e-05 : f64
    %cst_6 = arith.constant 8.000000e+02 : f64
    %cst_7 = arith.constant 8.000000e+02 : f32
    %0 = tensor.empty() : tensor<5x10xf32>
    %transposed = linalg.transpose ins(%arg0 : tensor<10x5xf32>) outs(%0 : tensor<5x10xf32>) permutation = [1, 0]
    %1 = tensor.empty() : tensor<800x10xf32>
    %2 = linalg.fill ins(%cst : f32) outs(%1 : tensor<800x10xf32>) -> tensor<800x10xf32>
    %3 = linalg.matmul ins(%arg8, %transposed : tensor<800x5xf32>, tensor<5x10xf32>) outs(%2 : tensor<800x10xf32>) -> tensor<800x10xf32>
    %4 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%3, %arg1 : tensor<800x10xf32>, tensor<10xf32>) outs(%1 : tensor<800x10xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %42 = arith.addf %in, %in_11 : f32
      linalg.yield %42 : f32
    } -> tensor<800x10xf32>
    %5 = tensor.empty() : tensor<i64>
    %6 = linalg.generic {indexing_maps = [#map2, #map2], iterator_types = []} ins(%arg5 : tensor<i64>) outs(%5 : tensor<i64>) {
    ^bb0(%in: i64, %out: i64):
      %42 = arith.addi %in, %c1_i64 : i64
      linalg.yield %42 : i64
    } -> tensor<i64>
    %7 = tensor.empty() : tensor<800x10xf64>
    %8 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%4 : tensor<800x10xf32>) outs(%7 : tensor<800x10xf64>) {
    ^bb0(%in: f32, %out: f64):
      %42 = arith.extf %in : f32 to f64
      linalg.yield %42 : f64
    } -> tensor<800x10xf64>
    %9 = tensor.empty() : tensor<1x10xf64>
    %10 = linalg.fill ins(%cst_0 : f64) outs(%9 : tensor<1x10xf64>) -> tensor<1x10xf64>
    %11 = linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["reduction", "parallel"]} ins(%8 : tensor<800x10xf64>) outs(%10 : tensor<1x10xf64>) {
    ^bb0(%in: f64, %out: f64):
      %42 = arith.addf %in, %out : f64
      linalg.yield %42 : f64
    } -> tensor<1x10xf64>
    %12 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%11 : tensor<1x10xf64>) outs(%9 : tensor<1x10xf64>) {
    ^bb0(%in: f64, %out: f64):
      %42 = arith.divf %in, %cst_6 : f64
      linalg.yield %42 : f64
    } -> tensor<1x10xf64>
    %13 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%8, %12 : tensor<800x10xf64>, tensor<1x10xf64>) outs(%7 : tensor<800x10xf64>) {
    ^bb0(%in: f64, %in_11: f64, %out: f64):
      %42 = arith.subf %in, %in_11 : f64
      linalg.yield %42 : f64
    } -> tensor<800x10xf64>
    %14 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%13, %13 : tensor<800x10xf64>, tensor<800x10xf64>) outs(%7 : tensor<800x10xf64>) {
    ^bb0(%in: f64, %in_11: f64, %out: f64):
      %42 = arith.mulf %in, %in_11 : f64
      linalg.yield %42 : f64
    } -> tensor<800x10xf64>
    %15 = linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["reduction", "parallel"]} ins(%14 : tensor<800x10xf64>) outs(%10 : tensor<1x10xf64>) {
    ^bb0(%in: f64, %out: f64):
      %42 = arith.addf %in, %out : f64
      linalg.yield %42 : f64
    } -> tensor<1x10xf64>
    %16 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%15 : tensor<1x10xf64>) outs(%9 : tensor<1x10xf64>) {
    ^bb0(%in: f64, %out: f64):
      %42 = arith.divf %in, %cst_6 : f64
      linalg.yield %42 : f64
    } -> tensor<1x10xf64>
    %17 = tensor.empty() : tensor<1x10xf32>
    %18 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%16 : tensor<1x10xf64>) outs(%17 : tensor<1x10xf32>) {
    ^bb0(%in: f64, %out: f32):
      %42 = arith.truncf %in : f64 to f32
      linalg.yield %42 : f32
    } -> tensor<1x10xf32>
    %19 = linalg.fill ins(%cst : f32) outs(%17 : tensor<1x10xf32>) -> tensor<1x10xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map3], iterator_types = ["reduction", "parallel"]} ins(%4 : tensor<800x10xf32>) outs(%19 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %42 = arith.addf %in, %out : f32
      linalg.yield %42 : f32
    } -> tensor<1x10xf32>
    %21 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%20 : tensor<1x10xf32>) outs(%17 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %42 = arith.divf %in, %cst_7 : f32
      linalg.yield %42 : f32
    } -> tensor<1x10xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%18 : tensor<1x10xf32>) outs(%17 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %42 = arith.truncf %cst_5 : f64 to f32
      %43 = arith.addf %in, %42 : f32
      linalg.yield %43 : f32
    } -> tensor<1x10xf32>
    %23 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%22 : tensor<1x10xf32>) outs(%17 : tensor<1x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %42 = math.rsqrt %in : f32
      linalg.yield %42 : f32
    } -> tensor<1x10xf32>
    %24 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%4, %21 : tensor<800x10xf32>, tensor<1x10xf32>) outs(%1 : tensor<800x10xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %42 = arith.subf %in, %in_11 : f32
      linalg.yield %42 : f32
    } -> tensor<800x10xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map3, #map], iterator_types = ["parallel", "parallel"]} ins(%24, %23 : tensor<800x10xf32>, tensor<1x10xf32>) outs(%1 : tensor<800x10xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %42 = arith.mulf %in, %in_11 : f32
      linalg.yield %42 : f32
    } -> tensor<800x10xf32>
    %collapsed = tensor.collapse_shape %21 [[0, 1]] : tensor<1x10xf32> into tensor<10xf32>
    %26 = tensor.empty() : tensor<10xf32>
    %27 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%collapsed : tensor<10xf32>) outs(%26 : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %42 = arith.truncf %cst_4 : f64 to f32
      %43 = arith.mulf %in, %42 : f32
      linalg.yield %43 : f32
    } -> tensor<10xf32>
    %28 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%arg6 : tensor<10xf32>) outs(%26 : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %42 = arith.truncf %cst_3 : f64 to f32
      %43 = arith.mulf %in, %42 : f32
      linalg.yield %43 : f32
    } -> tensor<10xf32>
    %29 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%27, %28 : tensor<10xf32>, tensor<10xf32>) outs(%26 : tensor<10xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %42 = arith.addf %in, %in_11 : f32
      linalg.yield %42 : f32
    } -> tensor<10xf32>
    %collapsed_8 = tensor.collapse_shape %18 [[0, 1]] : tensor<1x10xf32> into tensor<10xf32>
    %30 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%collapsed_8 : tensor<10xf32>) outs(%26 : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %42 = arith.truncf %cst_2 : f64 to f32
      %43 = arith.mulf %in, %42 : f32
      linalg.yield %43 : f32
    } -> tensor<10xf32>
    %31 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%30 : tensor<10xf32>) outs(%26 : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %42 = arith.truncf %cst_4 : f64 to f32
      %43 = arith.mulf %in, %42 : f32
      linalg.yield %43 : f32
    } -> tensor<10xf32>
    %32 = linalg.generic {indexing_maps = [#map4, #map4], iterator_types = ["parallel"]} ins(%arg7 : tensor<10xf32>) outs(%26 : tensor<10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %42 = arith.truncf %cst_3 : f64 to f32
      %43 = arith.mulf %in, %42 : f32
      linalg.yield %43 : f32
    } -> tensor<10xf32>
    %33 = linalg.generic {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]} ins(%31, %32 : tensor<10xf32>, tensor<10xf32>) outs(%26 : tensor<10xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %42 = arith.addf %in, %in_11 : f32
      linalg.yield %42 : f32
    } -> tensor<10xf32>
    %34 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%25, %arg2 : tensor<800x10xf32>, tensor<10xf32>) outs(%1 : tensor<800x10xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %42 = arith.mulf %in, %in_11 : f32
      linalg.yield %42 : f32
    } -> tensor<800x10xf32>
    %35 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%34, %arg3 : tensor<800x10xf32>, tensor<10xf32>) outs(%1 : tensor<800x10xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %42 = arith.addf %in, %in_11 : f32
      linalg.yield %42 : f32
    } -> tensor<800x10xf32>
    %36 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%35 : tensor<800x10xf32>) outs(%1 : tensor<800x10xf32>) {
    ^bb0(%in: f32, %out: f32):
      %42 = arith.cmpf ugt, %in, %cst : f32
      %43 = arith.select %42, %in, %cst : f32
      linalg.yield %43 : f32
    } -> tensor<800x10xf32>
    %37 = tensor.empty() : tensor<10x1xf32>
    %transposed_9 = linalg.transpose ins(%arg4 : tensor<1x10xf32>) outs(%37 : tensor<10x1xf32>) permutation = [1, 0]
    %38 = tensor.empty() : tensor<800x1xf32>
    %39 = linalg.fill ins(%cst : f32) outs(%38 : tensor<800x1xf32>) -> tensor<800x1xf32>
    %40 = linalg.matmul ins(%36, %transposed_9 : tensor<800x10xf32>, tensor<10x1xf32>) outs(%39 : tensor<800x1xf32>) -> tensor<800x1xf32>
    %41 = linalg.generic {indexing_maps = [#map, #map1, #map], iterator_types = ["parallel", "parallel"]} ins(%40, %cst_1 : tensor<800x1xf32>, tensor<1xf32>) outs(%38 : tensor<800x1xf32>) {
    ^bb0(%in: f32, %in_11: f32, %out: f32):
      %42 = arith.addf %in, %in_11 : f32
      linalg.yield %42 : f32
    } -> tensor<800x1xf32>
    %collapsed_10 = tensor.collapse_shape %41 [[0, 1]] : tensor<800x1xf32> into tensor<800xf32>
    return %6, %29, %33, %collapsed_10 : tensor<i64>, tensor<10xf32>, tensor<10xf32>, tensor<800xf32>
  }
}
