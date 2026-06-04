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

  // MSELoss: mean((pred - target)^2)
  func.func @mse_loss(%pred: tensor<800xf32>, %target: tensor<800xf32>) -> tensor<f32> {
    %cst_800 = arith.constant dense<800.0> : tensor<f32>
    %diff = arith.subf %pred, %target : tensor<800xf32>
    %squared = arith.mulf %diff, %diff : tensor<800xf32>
    %zero_scalar = arith.constant dense<0.0> : tensor<f32>
    %sum = linalg.reduce ins(%squared : tensor<800xf32>) outs(%zero_scalar : tensor<f32>) dimensions = [0]
      (%in: f32, %out: f32) { %s = arith.addf %in, %out : f32 linalg.yield %s : f32 }
    %mse = arith.divf %sum, %cst_800 : tensor<f32>
    return %mse : tensor<f32>
  }

  // Combined forward + loss: returns (loss, updated_mean, updated_var)
  // loss is the primal (primal_of=0), BN stats are non-primal pass-through
  func.func @forward_and_loss(%W1: tensor<10x5xf32>, %b1: tensor<10xf32>, %gamma: tensor<10xf32>, %beta: tensor<10xf32>, %W2: tensor<1x10xf32>, %step: tensor<i64>, %mean: tensor<10xf32>, %var: tensor<10xf32>, %X: tensor<800x5xf32>, %target: tensor<800xf32>) -> (tensor<f32>, tensor<10xf32>, tensor<10xf32>) {
    %step_new, %mean_new, %var_new, %pred = func.call @forward(%W1, %b1, %gamma, %beta, %W2, %step, %mean, %var, %X) : (tensor<10x5xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<1x10xf32>, tensor<i64>, tensor<10xf32>, tensor<10xf32>, tensor<800x5xf32>) -> (tensor<i64>, tensor<10xf32>, tensor<10xf32>, tensor<800xf32>)
    %loss = func.call @mse_loss(%pred, %target) : (tensor<800xf32>, tensor<800xf32>) -> tensor<f32>
    return %loss, %mean_new, %var_new : tensor<f32>, tensor<10xf32>, tensor<10xf32>
  }

  // Compute gradients for all 5 trainable params + pass-through BN stats
  func.func @compute_gradients(%W1: tensor<10x5xf32>, %b1: tensor<10xf32>, %gamma: tensor<10xf32>, %beta: tensor<10xf32>, %W2: tensor<1x10xf32>, %step: tensor<i64>, %mean: tensor<10xf32>, %var: tensor<10xf32>, %X: tensor<800x5xf32>, %target: tensor<800xf32>) -> (tensor<10x5xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<1x10xf32>, tensor<f32>, tensor<10xf32>, tensor<10xf32>) {
    %dW1, %db1, %dgamma, %dbeta, %dW2, %loss, %mean_new, %var_new = lagrad.grad @forward_and_loss(%W1, %b1, %gamma, %beta, %W2, %step, %mean, %var, %X, %target)
      {of = [0, 1, 2, 3, 4], return_primal, primal_of = 0 : i64} :
      (tensor<10x5xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<1x10xf32>, tensor<i64>, tensor<10xf32>, tensor<10xf32>, tensor<800x5xf32>, tensor<800xf32>)
      -> (tensor<10x5xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<1x10xf32>, tensor<f32>, tensor<10xf32>, tensor<10xf32>)
    return %dW1, %db1, %dgamma, %dbeta, %dW2, %loss, %mean_new, %var_new : tensor<10x5xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<1x10xf32>, tensor<f32>, tensor<10xf32>, tensor<10xf32>
  }

  func.func private @print_loss(i32, f32) attributes {llvm.emit_c_interface}

  func.func @mlir_main() attributes {llvm.emit_c_interface, no_inline} {
    %c1 = arith.constant 1 : index
    %c21 = arith.constant 40 : index

    %W1_init = arith.constant dense<0.01> : tensor<10x5xf32>
    %b1_init = arith.constant dense<0.0> : tensor<10xf32>
    %gamma_init = arith.constant dense<1.0> : tensor<10xf32>
    %beta_init = arith.constant dense<0.0> : tensor<10xf32>
    %W2_init = arith.constant dense<0.01> : tensor<1x10xf32>
    %step_init = arith.constant dense<0> : tensor<i64>
    %mean_init = arith.constant dense<0.0> : tensor<10xf32>
    %var_init = arith.constant dense<1.0> : tensor<10xf32>

    %X = arith.constant dense<1.0> : tensor<800x5xf32>
    %target = arith.constant dense<0.5> : tensor<800xf32>
    %lr = arith.constant 0.1 : f32

    %W1_f, %b1_f, %gamma_f, %beta_f, %W2_f = scf.for %i = %c1 to %c21 step %c1
        iter_args(%W1 = %W1_init, %b1 = %b1_init, %gamma = %gamma_init, %beta = %beta_init, %W2 = %W2_init)
        -> (tensor<10x5xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<1x10xf32>) {

      %dW1, %db1, %dgamma, %dbeta, %dW2, %loss_t, %mean_new, %var_new = func.call @compute_gradients(%W1, %b1, %gamma, %beta, %W2, %step_init, %mean_init, %var_init, %X, %target)
        : (tensor<10x5xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<1x10xf32>, tensor<i64>, tensor<10xf32>, tensor<10xf32>, tensor<800x5xf32>, tensor<800xf32>)
        -> (tensor<10x5xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<1x10xf32>, tensor<f32>, tensor<10xf32>, tensor<10xf32>)

      %loss = tensor.extract %loss_t[] : tensor<f32>
      %step_i32 = arith.index_cast %i : index to i32
      func.call @print_loss(%step_i32, %loss) : (i32, f32) -> ()

      // SGD update W1
      %W1_new = linalg.generic
        {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%W1, %dW1 : tensor<10x5xf32>, tensor<10x5xf32>)
        outs(%W1 : tensor<10x5xf32>) {
      ^bb0(%p: f32, %g: f32, %out: f32):
        %scaled = arith.mulf %lr, %g : f32
        %new = arith.subf %p, %scaled : f32
        linalg.yield %new : f32
      } -> tensor<10x5xf32>

      // SGD update b1
      %b1_new = linalg.generic
        {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]}
        ins(%b1, %db1 : tensor<10xf32>, tensor<10xf32>)
        outs(%b1 : tensor<10xf32>) {
      ^bb0(%p: f32, %g: f32, %out: f32):
        %scaled = arith.mulf %lr, %g : f32
        %new = arith.subf %p, %scaled : f32
        linalg.yield %new : f32
      } -> tensor<10xf32>

      // SGD update gamma
      %gamma_new = linalg.generic
        {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]}
        ins(%gamma, %dgamma : tensor<10xf32>, tensor<10xf32>)
        outs(%gamma : tensor<10xf32>) {
      ^bb0(%p: f32, %g: f32, %out: f32):
        %scaled = arith.mulf %lr, %g : f32
        %new = arith.subf %p, %scaled : f32
        linalg.yield %new : f32
      } -> tensor<10xf32>

      // SGD update beta
      %beta_new = linalg.generic
        {indexing_maps = [#map4, #map4, #map4], iterator_types = ["parallel"]}
        ins(%beta, %dbeta : tensor<10xf32>, tensor<10xf32>)
        outs(%beta : tensor<10xf32>) {
      ^bb0(%p: f32, %g: f32, %out: f32):
        %scaled = arith.mulf %lr, %g : f32
        %new = arith.subf %p, %scaled : f32
        linalg.yield %new : f32
      } -> tensor<10xf32>

      // SGD update W2
      %W2_new = linalg.generic
        {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%W2, %dW2 : tensor<1x10xf32>, tensor<1x10xf32>)
        outs(%W2 : tensor<1x10xf32>) {
      ^bb0(%p: f32, %g: f32, %out: f32):
        %scaled = arith.mulf %lr, %g : f32
        %new = arith.subf %p, %scaled : f32
        linalg.yield %new : f32
      } -> tensor<1x10xf32>

      scf.yield %W1_new, %b1_new, %gamma_new, %beta_new, %W2_new
        : tensor<10x5xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<1x10xf32>
    }

    // Use final parameters to prevent DCE
    %final_loss_t, %final_mean, %final_var = func.call @forward_and_loss(%W1_f, %b1_f, %gamma_f, %beta_f, %W2_f, %step_init, %mean_init, %var_init, %X, %target)
      : (tensor<10x5xf32>, tensor<10xf32>, tensor<10xf32>, tensor<10xf32>, tensor<1x10xf32>, tensor<i64>, tensor<10xf32>, tensor<10xf32>, tensor<800x5xf32>, tensor<800xf32>)
      -> (tensor<f32>, tensor<10xf32>, tensor<10xf32>)
    %final_loss = tensor.extract %final_loss_t[] : tensor<f32>
    %step_final = arith.constant 999 : i32
    func.call @print_loss(%step_final, %final_loss) : (i32, f32) -> ()

    return
  }
}
