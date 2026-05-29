// Simple Linear Regression Training Example
// Model: y = X @ W + b
// Loss: MSE = mean((y_pred - y_true)^2)
// Optimizer: SGD with learning rate

// External C function for printing training info
// Signature: print_training_info(step: i32, loss: f32, w0: f32, w1: f32, b0: f32)
func.func private @print_training_info(i32, f32, f32, f32, f32) attributes { llvm.emit_c_interface }

// Affine maps for linalg operations
#map_broadcast_add = affine_map<(d0, d1) -> (d0, d1)>
#map_broadcast_add_b = affine_map<(d0, d1) -> (d1)>
#map_sub = affine_map<(d0, d1) -> (d0, d1)>
#map_square = affine_map<(d0, d1) -> (d0, d1)>

// ============================================================================
// Forward Pass: Linear layer y = X @ W + b
// ============================================================================
func.func @forward(
  %X: tensor<4x2xf32>,
  %W: tensor<2x1xf32>,
  %b: tensor<1xf32>
) -> tensor<4x1xf32> {
  %cst_zero = arith.constant 0.0 : f32
  %zero_4x1 = arith.constant dense<0.0> : tensor<4x1xf32>

  // Matrix multiplication: X @ W
  %matmul = linalg.matmul
    ins(%X, %W : tensor<4x2xf32>, tensor<2x1xf32>)
    outs(%zero_4x1 : tensor<4x1xf32>)
    -> tensor<4x1xf32>

  // Add bias: matmul + b (broadcast)
  %result = linalg.generic
    {indexing_maps = [#map_broadcast_add, #map_broadcast_add_b, #map_broadcast_add],
     iterator_types = ["parallel", "parallel"]}
    ins(%matmul, %b : tensor<4x1xf32>, tensor<1xf32>)
    outs(%zero_4x1 : tensor<4x1xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %sum = arith.addf %in1, %in2 : f32
    linalg.yield %sum : f32
  } -> tensor<4x1xf32>

  return %result : tensor<4x1xf32>
}

// ============================================================================
// Loss Function: MSE = mean((y_pred - y_true)^2)
// ============================================================================
func.func @mse_loss(
  %y_pred: tensor<4x1xf32>,
  %y_true: tensor<4x1xf32>
) -> f32 {
  %cst_zero = arith.constant 0.0 : f32
  %zero_4x1 = arith.constant dense<0.0> : tensor<4x1xf32>
  %cst_4 = arith.constant 4.0 : f32

  // Compute difference: y_pred - y_true
  %diff = linalg.generic
    {indexing_maps = [#map_sub, #map_sub, #map_sub],
     iterator_types = ["parallel", "parallel"]}
    ins(%y_pred, %y_true : tensor<4x1xf32>, tensor<4x1xf32>)
    outs(%zero_4x1 : tensor<4x1xf32>) {
  ^bb0(%in1: f32, %in2: f32, %out: f32):
    %d = arith.subf %in1, %in2 : f32
    linalg.yield %d : f32
  } -> tensor<4x1xf32>

  // Square the difference
  %squared = linalg.generic
    {indexing_maps = [#map_square, #map_square],
     iterator_types = ["parallel", "parallel"]}
    ins(%diff : tensor<4x1xf32>)
    outs(%zero_4x1 : tensor<4x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %sq = arith.mulf %in, %in : f32
    linalg.yield %sq : f32
  } -> tensor<4x1xf32>

  // Sum all elements - use tensor<f32> for reduction
  %zero_scalar = arith.constant dense<0.0> : tensor<f32>
  %sum_tensor = linalg.generic
    {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> ()>],
     iterator_types = ["reduction", "reduction"]}
    ins(%squared : tensor<4x1xf32>)
    outs(%zero_scalar : tensor<f32>) {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  } -> tensor<f32>

  // Extract scalar from tensor
  %sum = tensor.extract %sum_tensor[] : tensor<f32>

  // Divide by number of samples (4)
  %mse = arith.divf %sum, %cst_4 : f32

  return %mse : f32
}

// ============================================================================
// Combined Forward + Loss
// ============================================================================
func.func @forward_and_loss(
  %X: tensor<4x2xf32>,
  %y_true: tensor<4x1xf32>,
  %W: tensor<2x1xf32>,
  %b: tensor<1xf32>
) -> f32 {
  %y_pred = func.call @forward(%X, %W, %b) : (tensor<4x2xf32>, tensor<2x1xf32>, tensor<1xf32>) -> tensor<4x1xf32>
  %loss = func.call @mse_loss(%y_pred, %y_true) : (tensor<4x1xf32>, tensor<4x1xf32>) -> f32
  return %loss : f32
}

// ============================================================================
// Compute Loss and Gradients in a single pass
// lagrad.grad with return_primal attribute returns both gradients and the primal result (loss)
// ============================================================================
func.func @compute_loss_and_gradients(
  %X: tensor<4x2xf32>,
  %y_true: tensor<4x1xf32>,
  %W: tensor<2x1xf32>,
  %b: tensor<1xf32>
) -> (tensor<2x1xf32>, tensor<1xf32>, f32) {
  // Compute gradients and loss in one call
  // lagrad.grad with {return_primal} returns: gradients for W and b, plus the primal result (loss)
  %dW, %db, %loss = lagrad.grad @forward_and_loss(%X, %y_true, %W, %b)
    {of = [2, 3], return_primal} :
    (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)

  return %dW, %db, %loss : tensor<2x1xf32>, tensor<1xf32>, f32
}

// ============================================================================
// SGD Optimizer: W = W - lr * dW, b = b - lr * db
// ============================================================================
func.func @sgd_update(
  %W: tensor<2x1xf32>,
  %b: tensor<1xf32>,
  %dW: tensor<2x1xf32>,
  %db: tensor<1xf32>,
  %lr: f32
) -> (tensor<2x1xf32>, tensor<1xf32>) {
  %zero_2x1 = arith.constant dense<0.0> : tensor<2x1xf32>
  %zero_1 = arith.constant dense<0.0> : tensor<1xf32>

  // Update W: W_new = W - lr * dW
  %W_new = linalg.generic
    {indexing_maps = [#map_broadcast_add, #map_broadcast_add, #map_broadcast_add],
     iterator_types = ["parallel", "parallel"]}
    ins(%W, %dW : tensor<2x1xf32>, tensor<2x1xf32>)
    outs(%zero_2x1 : tensor<2x1xf32>) {
  ^bb0(%w: f32, %dw: f32, %out: f32):
    %scaled = arith.mulf %lr, %dw : f32
    %updated = arith.subf %w, %scaled : f32
    linalg.yield %updated : f32
  } -> tensor<2x1xf32>

  // Update b: b_new = b - lr * db
  %b_new = linalg.generic
    {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
     iterator_types = ["parallel"]}
    ins(%b, %db : tensor<1xf32>, tensor<1xf32>)
    outs(%zero_1 : tensor<1xf32>) {
  ^bb0(%bi: f32, %dbi: f32, %out: f32):
    %scaled = arith.mulf %lr, %dbi : f32
    %updated = arith.subf %bi, %scaled : f32
    linalg.yield %updated : f32
  } -> tensor<1xf32>

  return %W_new, %b_new : tensor<2x1xf32>, tensor<1xf32>
}

// ============================================================================
// Single Training Step
// ============================================================================
func.func @train_step(
  %X: tensor<4x2xf32>,
  %y_true: tensor<4x1xf32>,
  %W: tensor<2x1xf32>,
  %b: tensor<1xf32>,
  %lr: f32
) -> (tensor<2x1xf32>, tensor<1xf32>, f32) {
  // Compute loss and gradients in one call
  %dW, %db, %loss = func.call @compute_loss_and_gradients(%X, %y_true, %W, %b)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)

  // Update parameters
  %W_new, %b_new = func.call @sgd_update(%W, %b, %dW, %db, %lr)
    : (tensor<2x1xf32>, tensor<1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>)

  return %W_new, %b_new, %loss : tensor<2x1xf32>, tensor<1xf32>, f32
}

// ============================================================================
// Main Entry Point - Training with progress output
// ============================================================================
func.func @mlir_main() attributes {llvm.emit_c_interface} {
  // Training data: 4 samples, 2 features
  // X = [[1, 2], [3, 4], [5, 6], [7, 8]]
  %X = arith.constant dense<[
    [1.0, 2.0],
    [3.0, 4.0],
    [5.0, 6.0],
    [7.0, 8.0]
  ]> : tensor<4x2xf32>

  // Target: y = 2*x1 + 3*x2 + 1
  // y_true = [[9], [19], [29], [39]]
  %y_true = arith.constant dense<[
    [9.0],
    [19.0],
    [29.0],
    [39.0]
  ]> : tensor<4x1xf32>

  // Initial weights (random initialization)
  %W_init = arith.constant dense<[
    [0.1],
    [0.2]
  ]> : tensor<2x1xf32>

  // Initial bias
  %b_init = arith.constant dense<[0.0]> : tensor<1xf32>

  // Learning rate
  %lr = arith.constant 0.01 : f32

  // Index constants for tensor.extract
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index

  // // Compute initial loss
  // %loss_init = func.call @forward_and_loss(%X, %y_true, %W_init, %b_init)
  //   : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>) -> f32

  // // Extract and print initial parameters
  // %w0_init = tensor.extract %W_init[%c0, %c0] : tensor<2x1xf32>
  // %w1_init = tensor.extract %W_init[%c1, %c0] : tensor<2x1xf32>
  // %b0_init = tensor.extract %b_init[%c0] : tensor<1xf32>
  // %step0 = arith.constant 0 : i32
  // func.call @print_training_info(%step0, %loss_init, %w0_init, %w1_init, %b0_init)
  //   : (i32, f32, f32, f32, f32) -> ()

  // Training step 1
  %W1, %b1, %loss1 = func.call @train_step(%X, %y_true, %W_init, %b_init, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_1 = tensor.extract %W1[%c0, %c0] : tensor<2x1xf32>
  %w1_1 = tensor.extract %W1[%c1, %c0] : tensor<2x1xf32>
  %b0_1 = tensor.extract %b1[%c0] : tensor<1xf32>
  %step1 = arith.constant 1 : i32
  func.call @print_training_info(%step1, %loss1, %w0_1, %w1_1, %b0_1)
    : (i32, f32, f32, f32, f32) -> ()

  // Training step 2
  %W2, %b2, %loss2 = func.call @train_step(%X, %y_true, %W1, %b1, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_2 = tensor.extract %W2[%c0, %c0] : tensor<2x1xf32>
  %w1_2 = tensor.extract %W2[%c1, %c0] : tensor<2x1xf32>
  %b0_2 = tensor.extract %b2[%c0] : tensor<1xf32>
  %step2 = arith.constant 2 : i32
  func.call @print_training_info(%step2, %loss2, %w0_2, %w1_2, %b0_2)
    : (i32, f32, f32, f32, f32) -> ()

  // Training step 3
  %W3, %b3, %loss3 = func.call @train_step(%X, %y_true, %W2, %b2, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_3 = tensor.extract %W3[%c0, %c0] : tensor<2x1xf32>
  %w1_3 = tensor.extract %W3[%c1, %c0] : tensor<2x1xf32>
  %b0_3 = tensor.extract %b3[%c0] : tensor<1xf32>
  %step3 = arith.constant 3 : i32
  func.call @print_training_info(%step3, %loss3, %w0_3, %w1_3, %b0_3)
    : (i32, f32, f32, f32, f32) -> ()

  // Training step 4
  %W4, %b4, %loss4 = func.call @train_step(%X, %y_true, %W3, %b3, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_4 = tensor.extract %W4[%c0, %c0] : tensor<2x1xf32>
  %w1_4 = tensor.extract %W4[%c1, %c0] : tensor<2x1xf32>
  %b0_4 = tensor.extract %b4[%c0] : tensor<1xf32>
  %step4 = arith.constant 4 : i32
  func.call @print_training_info(%step4, %loss4, %w0_4, %w1_4, %b0_4)
    : (i32, f32, f32, f32, f32) -> ()

  // Training step 5
  %W5, %b5, %loss5 = func.call @train_step(%X, %y_true, %W4, %b4, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_5 = tensor.extract %W5[%c0, %c0] : tensor<2x1xf32>
  %w1_5 = tensor.extract %W5[%c1, %c0] : tensor<2x1xf32>
  %b0_5 = tensor.extract %b5[%c0] : tensor<1xf32>
  %step5 = arith.constant 5 : i32
  func.call @print_training_info(%step5, %loss5, %w0_5, %w1_5, %b0_5)
    : (i32, f32, f32, f32, f32) -> ()

  // Training step 6
  %W6, %b6, %loss6 = func.call @train_step(%X, %y_true, %W5, %b5, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_6 = tensor.extract %W6[%c0, %c0] : tensor<2x1xf32>
  %w1_6 = tensor.extract %W6[%c1, %c0] : tensor<2x1xf32>
  %b0_6 = tensor.extract %b6[%c0] : tensor<1xf32>
  %step6 = arith.constant 6 : i32
  func.call @print_training_info(%step6, %loss6, %w0_6, %w1_6, %b0_6)
    : (i32, f32, f32, f32, f32) -> ()

  // Training step 7
  %W7, %b7, %loss7 = func.call @train_step(%X, %y_true, %W6, %b6, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_7 = tensor.extract %W7[%c0, %c0] : tensor<2x1xf32>
  %w1_7 = tensor.extract %W7[%c1, %c0] : tensor<2x1xf32>
  %b0_7 = tensor.extract %b7[%c0] : tensor<1xf32>
  %step7 = arith.constant 7 : i32
  func.call @print_training_info(%step7, %loss7, %w0_7, %w1_7, %b0_7)
    : (i32, f32, f32, f32, f32) -> ()

  // Training step 8
  %W8, %b8, %loss8 = func.call @train_step(%X, %y_true, %W7, %b7, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_8 = tensor.extract %W8[%c0, %c0] : tensor<2x1xf32>
  %w1_8 = tensor.extract %W8[%c1, %c0] : tensor<2x1xf32>
  %b0_8 = tensor.extract %b8[%c0] : tensor<1xf32>
  %step8 = arith.constant 8 : i32
  func.call @print_training_info(%step8, %loss8, %w0_8, %w1_8, %b0_8)
    : (i32, f32, f32, f32, f32) -> ()

  // Training step 9
  %W9, %b9, %loss9 = func.call @train_step(%X, %y_true, %W8, %b8, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_9 = tensor.extract %W9[%c0, %c0] : tensor<2x1xf32>
  %w1_9 = tensor.extract %W9[%c1, %c0] : tensor<2x1xf32>
  %b0_9 = tensor.extract %b9[%c0] : tensor<1xf32>
  %step9 = arith.constant 9 : i32
  func.call @print_training_info(%step9, %loss9, %w0_9, %w1_9, %b0_9)
    : (i32, f32, f32, f32, f32) -> ()

  // Training step 10
  %W10, %b10, %loss10 = func.call @train_step(%X, %y_true, %W9, %b9, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_10 = tensor.extract %W10[%c0, %c0] : tensor<2x1xf32>
  %w1_10 = tensor.extract %W10[%c1, %c0] : tensor<2x1xf32>
  %b0_10 = tensor.extract %b10[%c0] : tensor<1xf32>
  %step10 = arith.constant 10 : i32
  func.call @print_training_info(%step10, %loss10, %w0_10, %w1_10, %b0_10)
    : (i32, f32, f32, f32, f32) -> ()

  // Training step 15 (skip a few for brevity)
  %W11, %b11, %loss11 = func.call @train_step(%X, %y_true, %W10, %b10, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %W12, %b12, %loss12 = func.call @train_step(%X, %y_true, %W11, %b11, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %W13, %b13, %loss13 = func.call @train_step(%X, %y_true, %W12, %b12, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %W14, %b14, %loss14 = func.call @train_step(%X, %y_true, %W13, %b13, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %W15, %b15, %loss15 = func.call @train_step(%X, %y_true, %W14, %b14, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_15 = tensor.extract %W15[%c0, %c0] : tensor<2x1xf32>
  %w1_15 = tensor.extract %W15[%c1, %c0] : tensor<2x1xf32>
  %b0_15 = tensor.extract %b15[%c0] : tensor<1xf32>
  %step15 = arith.constant 15 : i32
  func.call @print_training_info(%step15, %loss15, %w0_15, %w1_15, %b0_15)
    : (i32, f32, f32, f32, f32) -> ()

  // Training step 20
  %W16, %b16, %loss16 = func.call @train_step(%X, %y_true, %W15, %b15, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %W17, %b17, %loss17 = func.call @train_step(%X, %y_true, %W16, %b16, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %W18, %b18, %loss18 = func.call @train_step(%X, %y_true, %W17, %b17, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %W19, %b19, %loss19 = func.call @train_step(%X, %y_true, %W18, %b18, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %W20, %b20, %loss20 = func.call @train_step(%X, %y_true, %W19, %b19, %lr)
    : (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>, f32)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  %w0_20 = tensor.extract %W20[%c0, %c0] : tensor<2x1xf32>
  %w1_20 = tensor.extract %W20[%c1, %c0] : tensor<2x1xf32>
  %b0_20 = tensor.extract %b20[%c0] : tensor<1xf32>
  %step20 = arith.constant 20 : i32
  func.call @print_training_info(%step20, %loss20, %w0_20, %w1_20, %b0_20)
    : (i32, f32, f32, f32, f32) -> ()

  return
}
