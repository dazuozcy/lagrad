# return_primal 属性实现说明

## 概述

为 `lagrad.grad` 操作添加了 `return_primal` 属性，用于控制生成的伴随函数是否返回原始函数的计算结果（如 loss 值）。

## 动机

在训练场景中，通常需要同时获取梯度和 loss 值。之前的实现会在 `train_step` 中分别调用：
1. `forward_and_loss` 计算 loss
2. `lagrad.grad @forward_and_loss` 计算梯度

这导致 `forward` 和 `mse_loss` 被执行了两次。通过 `return_primal` 属性，可以让 `lagrad.grad` 在计算梯度的同时返回 loss 值，避免重复计算。

## 实现细节

### 1. TableGen 定义 (include/LAGrad/LAGradOps.td)

在 `LAGrad_GradOp` 的描述中添加了 `return_primal` 属性说明：

```tablegen
let description = [{
    ...
    Optional attributes:
    - `return_primal`: UnitAttr indicating the primal result should also be returned (default: false)
}];
```

### 2. 验证器 (lib/LAGrad/LAGradOps.cpp)

更新了 `GradOp::verifySymbolUses` 以验证 `return_primal` 属性：

```cpp
bool returnPrimal = (*this)->hasAttrOfType<UnitAttr>("return_primal");
size_t expectedResults = gradientsOf.size() + ((customGradSignal || !returnPrimal) ? 0 : 1);

// 验证最后一个返回值是原始结果（当 return_primal 被设置时）
if (!customGradSignal && returnPrimal) {
  if (fnType.getNumResults() != 1)
    return emitOpError("differentiated function must have exactly 1 result");
  if (getResult(gradientsOf.size()).getType() != fnType.getResult(0)) {
    // 类型不匹配错误
  }
}
```

### 3. 函数签名 (include/LAGrad/Utils.h)

更新了 `differentiateFunction` 的签名：

```cpp
func::FuncOp differentiateFunction(func::FuncOp funcOp, LAGradContext &ctx,
                             ArrayAttr gradientsOf,
                             ConversionPatternRewriter &rewriter, 
                             bool topLevel,
                             bool onehotSparse, 
                             bool returnPrimal = false);
```

### 4. 函数实现 (lib/LAGrad/Utils.cpp)

在 `differentiateFunction` 中添加了逻辑来追加原始结果：

```cpp
// 保存原始结果
Value primalResult;
// ... 在处理 return 操作时保存 ...
primalResult = operand;

// 在构建返回类型时追加原始结果
if (topLevel && returnPrimal && primalResult) {
  returnType.push_back(primalResult.getType());
  returnValue.push_back(primalResult);
}
```

### 5. 降低过程 (lib/LAGrad/LowerGrad.cpp)

在 `GradOpLowering::generateAdjointFunc` 中读取属性并传递给 `differentiateFunction`：

```cpp
bool returnPrimal = gradOp->hasAttrOfType<UnitAttr>("return_primal");

return differentiateFunction(funcOp, lagradctx, gradientsOf, rewriter,
                             /*topLevel=*/!customGradSignal,
                             /*onehotsparse=*/oneHotSparse,
                             /*returnPrimal=*/returnPrimal);
```

## 使用示例

### 不返回原始结果（默认行为）

```mlir
%grad = lagrad.grad @forward_and_loss(%X, %y_true, %W, %b)
  {of = [2, 3]} :
  (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>)
  -> (tensor<2x1xf32>, tensor<1xf32>)
```

生成的伴随函数签名：
```mlir
func.func @__grad_forward_and_loss(...) -> (tensor<2x1xf32>, tensor<1xf32>)
```

### 返回原始结果

```mlir
%dW, %db, %loss = lagrad.grad @forward_and_loss(%X, %y_true, %W, %b)
  {of = [2, 3], return_primal} :
  (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>)
  -> (tensor<2x1xf32>, tensor<1xf32>, f32)
```

生成的伴随函数签名：
```mlir
func.func @__grad_forward_and_loss(...) -> (tensor<2x1xf32>, tensor<1xf32>, f32)
```

## 测试

创建了测试文件 `test_return_primal.mlir` 来验证两种行为：

1. **不带 `return_primal`**：生成的函数只返回梯度
2. **带 `return_primal`**：生成的函数返回梯度和原始结果

测试命令：
```bash
./build/bin/lagrad-opt test/LAGrad/training/test_return_primal.mlir -take-grads
```

## 训练示例更新

更新了 `training.mlir` 中的 `compute_loss_and_gradients` 函数，使用 `return_primal` 属性：

```mlir
func.func @compute_loss_and_gradients(...) -> (tensor<2x1xf32>, tensor<1xf32>, f32) {
  %dW, %db, %loss = lagrad.grad @forward_and_loss(%X, %y_true, %W, %b)
    {of = [2, 3], return_primal} :
    (tensor<4x2xf32>, tensor<4x1xf32>, tensor<2x1xf32>, tensor<1xf32>)
    -> (tensor<2x1xf32>, tensor<1xf32>, f32)
  
  return %dW, %db, %loss : tensor<2x1xf32>, tensor<1xf32>, f32
}
```

这样 `train_step` 只需要调用一次 `compute_loss_and_gradients` 即可获取梯度和 loss，避免了重复计算。

## 性能提升

在训练示例中，每个训练步骤减少了：
- 1 次 `forward` 调用（包含矩阵乘法和偏置加法）
- 1 次 `mse_loss` 调用（包含差值计算、平方和归约）

对于 20 个训练步骤，总共减少了 40 次函数调用。

## 注意事项

1. **函数缓存**：如果对同一个函数使用不同的 `return_primal` 属性值进行微分，会重用第一次生成的伴随函数。这是因为伴随函数名只包含原始函数名，不包含属性信息。如需支持这种情况，需要实现名称修饰（name mangling）。

2. **默认值**：`return_primal` 默认为 `false`，保持向后兼容性。

3. **内部调用**：在 `reverseCallOp` 等内部调用中，`returnPrimal` 始终为 `false`，因为内部调用不需要原始结果。

## 文件修改清单

1. `include/LAGrad/LAGradOps.td` - 添加属性描述
2. `include/LAGrad/Utils.h` - 更新函数签名
3. `lib/LAGrad/LAGradOps.cpp` - 更新验证器
4. `lib/LAGrad/Utils.cpp` - 实现返回原始结果的逻辑
5. `lib/LAGrad/LowerGrad.cpp` - 读取属性并传递
6. `test/LAGrad/training/training.mlir` - 使用新属性
7. `test/LAGrad/training/test_return_primal.mlir` - 新增测试文件
