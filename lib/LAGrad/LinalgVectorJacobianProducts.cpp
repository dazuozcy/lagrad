#include "LAGrad/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {

// Check if the generic op represents a batch matmul pattern
static bool isBatchMatmulPattern(linalg::GenericOp op, int op_index) {
  auto iteratorTypes = op.getIteratorTypesArray();
  
  // Batch matmul requires 4 iterators: [batch, M, K, N] or similar
  if (iteratorTypes.size() != 4) return false;
  
  // Check for pattern: parallel, parallel, reduction, parallel (or similar)
  // This represents: out[b,m,n] = sum_k(A[b,m,k] * B[b,k,n])
  int numParallel = 0, numReduction = 0;
  int reductionDim = -1;
  for (size_t i = 0; i < iteratorTypes.size(); i++) {
    if (iteratorTypes[i] == utils::IteratorType::parallel) numParallel++;
    else if (iteratorTypes[i] == utils::IteratorType::reduction) {
      numReduction++;
      reductionDim = i;
    }
  }
  
  // Batch matmul should have 3 parallel and 1 reduction
  if (numParallel != 3 || numReduction != 1) return false;
  
  // Check indexing maps for batch matmul pattern
  auto indexingMaps = op.getIndexingMapsArray();
  if (indexingMaps.size() < 2) return false;
  
  // Get the shapes
  if (op.getNumDpsInputs() < 2) return false;
  
  auto inputs = op.getDpsInputs();
  auto input0Type = inputs[0].getType().dyn_cast<RankedTensorType>();
  auto input1Type = inputs[1].getType().dyn_cast<RankedTensorType>();
  auto outputType = op->getResult(0).getType().dyn_cast<RankedTensorType>();
  
  if (!input0Type || !input1Type || !outputType) return false;
  
  // Check if it's a 3D tensor operation (batch dimension)
  if (input0Type.getRank() != 3 || input1Type.getRank() != 3 || outputType.getRank() != 3) return false;
  
  // Check the body for mul + add pattern
  auto *body = op.getBody();
  if (!body) return false;
  
  // Look for arith.mulf followed by arith.addf
  bool hasMul = false, hasAdd = false;
  for (auto &bodyOp : *body) {
    if (isa<arith::MulFOp>(bodyOp)) hasMul = true;
    if (isa<arith::AddFOp>(bodyOp)) hasAdd = true;
  }
  
  if (!hasMul || !hasAdd) return false;
  
  // Check indexing maps to ensure this is a matmul pattern
  // For batch matmul: A[b,m,k] * B[b,k,n] -> C[b,m,n]
  // The reduction dimension should appear in both inputs but not in output
  
  auto map0 = indexingMaps[0];
  auto map1 = indexingMaps[1];
  auto mapOut = indexingMaps[2];
  
  // Check if reduction dimension is in both inputs
  bool reductionInInput0 = false, reductionInInput1 = false;
  bool reductionInOutput = false;
  
  for (unsigned i = 0; i < map0.getNumResults(); i++) {
    if (map0.getResult(i).isa<AffineDimExpr>()) {
      unsigned dim = map0.getResult(i).cast<AffineDimExpr>().getPosition();
      if (dim == (unsigned)reductionDim) reductionInInput0 = true;
    }
  }
  
  for (unsigned i = 0; i < map1.getNumResults(); i++) {
    if (map1.getResult(i).isa<AffineDimExpr>()) {
      unsigned dim = map1.getResult(i).cast<AffineDimExpr>().getPosition();
      if (dim == (unsigned)reductionDim) reductionInInput1 = true;
    }
  }
  
  for (unsigned i = 0; i < mapOut.getNumResults(); i++) {
    if (mapOut.getResult(i).isa<AffineDimExpr>()) {
      unsigned dim = mapOut.getResult(i).cast<AffineDimExpr>().getPosition();
      if (dim == (unsigned)reductionDim) reductionInOutput = true;
    }
  }
  
  // Reduction dimension should be in both inputs but not in output
  if (!reductionInInput0 || !reductionInInput1 || reductionInOutput) return false;
  
  return true;
}

// Create batch_matmul from generic op pattern
static Value createBatchMatmulFromGeneric(linalg::GenericOp op, int op_index,
                                         Value vjp_value, Value output,
                                         ConversionPatternRewriter &rewriter) {
  auto loc = op.getLoc();
  
  // Determine which operand to use based on op_index
  Value lhs, rhs;
  if (op_index == 0) {
    lhs = vjp_value;
    rhs = op.getOperand(1);
  } else {
    lhs = op.getOperand(0);
    rhs = vjp_value;
  }
  
  // Create batch_matmul
  auto batchMatmul = rewriter.create<linalg::BatchMatmulOp>(
      loc, TypeRange{output.getType()}, 
      ValueRange{lhs, rhs}, 
      ValueRange{output});
  
  return batchMatmul.getResult(0);
}

Value reverseGenericOp(linalg::GenericOp op, LAGradContext &ctx, Value operand,
                       Value vjp_value, int op_index, Value output,
                       ConversionPatternRewriter &rewriter) {
  // Try to recognize and optimize batch matmul patterns
  if (isBatchMatmulPattern(op, op_index)) {
    return createBatchMatmulFromGeneric(op, op_index, vjp_value, output, rewriter);
  }
  
  // Need to ensure:
  // if (op_index > (size_t)genericOp.getNumDpsInputs() - 1)
  //   continue;
  auto numIterators = op.getIteratorTypesArray().size();
  SmallVector<AffineMap, 6> indexing_maps(
      op->getNumOperands() + 1, rewriter.getMultiDimIdentityMap(numIterators));
  SmallVector<utils::IteratorType, 6> iterator_types(numIterators,
                                                     utils::IteratorType::parallel);

  auto outputShape = output.getType().dyn_cast_or_null<ShapedType>();
  assert(outputShape && outputShape.hasRank() &&
         "output must be a ranked type");
  SmallVector<AffineMap> generic_indexing_maps;
  for (auto attr : op.getIndexingMaps()) {
    generic_indexing_maps.push_back(cast<AffineMapAttr>(attr).getValue());
  }
  unsigned int op_count = op.getNumOperands();
  SmallVector<Value> inputs;
  for (size_t i = 0; i < op_count; i++) {
    if (i == static_cast<size_t>(op_index)) {
      indexing_maps[i] = generic_indexing_maps[i];
      inputs.push_back(op.getOperand(i));
    } else if (i == op_count - 1) {
      if (op_index == -1) {
        // In the case of free variables, the output is assumed to be 0d.
        indexing_maps[i + 1] = indexing_maps[i + 1].getSubMap({});
      } else {
        // The output has to map the shape of the current argument.
        indexing_maps[i + 1] = generic_indexing_maps[op_index];
        // find the dimensions that don't appear in the results
        AffineMap outputMap = generic_indexing_maps[op_index];
        for (size_t idx = 0; idx < outputMap.getNumDims(); idx++) {
          if (!outputMap.isFunctionOfDim(idx)) {
            iterator_types[idx] = utils::IteratorType::reduction;
          }
        }
      }
      // Add the gradient signal as an argument at the end of the
      // inputs.
      inputs.push_back(vjp_value);
      indexing_maps[i] = generic_indexing_maps[op_count - 1];
    } else {
      indexing_maps[i] = generic_indexing_maps[i];
      inputs.push_back(op.getOperand(i));
    }
  }

  DenseMap<Value, Value> bbEnv;
  SmallVector<Value> genericOperands;
  for (Value arg : op.getBodyRegion().getArguments()) {
    genericOperands.push_back(arg);
  }

  Operation *yieldOp = nullptr;

  auto adjoint = rewriter.create<linalg::GenericOp>(
      operand.getLoc(), /*resultTensorType=*/outputShape,
      /*inputs=*/inputs, /*outputs=*/ValueRange({output}),
      /*indexing_maps=*/indexing_maps,
      /*iterator_types=*/iterator_types,
      [&](OpBuilder &builder, Location loc, ValueRange regionArgs) {
        PatternRewriter::InsertionGuard insertionGuard(rewriter);
        SmallVector<mlir::Operation *> genericRegionOps =
            cloneBasicBlock(llvm::make_range(op.getBodyRegion().op_begin(), op.getBodyRegion().op_end()), builder, regionArgs, genericOperands);

        for (auto it = genericRegionOps.rbegin(); it != genericRegionOps.rend();
             it++) {
          auto rop = *it;
          if (rop->getName().getStringRef() == "linalg.yield") {
            bbEnv[rop->getOperand(0)] = regionArgs[regionArgs.size() - 2];
            rewriter.setInsertionPointAfter(rop);
            // rop->erase();
            rewriter.eraseOp(rop);
            yieldOp = rop;
          } else if (rop->getName().getStringRef() == "arith.cmpf") {
            continue;
          } else {
            populateVJP(rop, ctx, bbEnv, rewriter);
          }
        }

        // This add operation is required in the case of undoing
        // reductions. It might be possible to omit this, if the
        // output argument is never used in the primal, or perhaps if
        // the primal iterator types do not include reductions.
        // I'm not entirely sure how best to check if we can omit this.
        auto new_operand = op_index == -1 ? operand : regionArgs[op_index];
        if (!bbEnv[new_operand]) {
          rewriter.create<linalg::YieldOp>(loc,
                                           getZero(loc, new_operand, rewriter));
        } else {
          // Check if we can skip the add operation
          // Skip if: output is not used in primal OR no reduction dimensions
          bool hasReduction = false;
          for (auto iterType : iterator_types) {
            if (iterType == utils::IteratorType::reduction) {
              hasReduction = true;
              break;
            }
          }
          
          // Check if output argument is used in the primal body
          bool outputUsed = false;
          auto outputArg = regionArgs.back();
          for (auto &bodyOp : op.getBodyRegion().getOps()) {
            for (auto operand : bodyOp.getOperands()) {
              if (operand == outputArg) {
                outputUsed = true;
                break;
              }
            }
            if (outputUsed) break;
          }
          
          if (!hasReduction && !outputUsed) {
            // Skip the add operation - just yield the gradient
            rewriter.create<linalg::YieldOp>(loc, bbEnv[new_operand]);
          } else if (outputShape.getRank() !=
                     static_cast<int64_t>(numIterators)) {
            Value add_res = rewriter.create<arith::AddFOp>(
                loc, bbEnv[new_operand], regionArgs[regionArgs.size() - 1]);

            rewriter.create<linalg::YieldOp>(loc, add_res);
          } else {
            Value add_res = rewriter.create<arith::AddFOp>(
                loc, bbEnv[new_operand], regionArgs.back());
            rewriter.create<linalg::YieldOp>(loc, add_res);
          }
        }
      });

  for (auto &bodyOp : adjoint.getBodyRegion().getOps()) {
    // Ugly, but necessary to do some form of cleanup here because non-active
    // primal ops might no longer be in scope if the generic ops are inside a
    // loop. scf.if ops inside the generic body cause this to segfault for some
    // reason.
    for (auto *user : bodyOp.getUsers()) {
      if (bodyOp.getNumResults() > 0 && bodyOp.hasOneUse() && user == yieldOp) {
        rewriter.eraseOp(&bodyOp);
      }
    }
  }
  adjoint->setAttr("adjoint of " + ctx.debug_names[op.getResult(0)],
                   UnitAttr::get(op.getContext()));

  return adjoint.getResult(0);
}
} // namespace mlir
