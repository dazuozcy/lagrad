#include "LAGrad/LAGradDialect.h"
#include "LAGrad/LAGradOps.h"
#include "LAGrad/Transforms.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace lagrad;

using llvm::errs;
namespace {

auto containsEnv(IRMapping &map,
                 ConversionPatternRewriter &rewriter) {
  return [&](Value v) { return map.contains(rewriter.getRemappedValue(v)); };
}

auto lookupEnv(IRMapping &map, ConversionPatternRewriter &rewriter) {
  return [&](Value v) { return map.lookup(rewriter.getRemappedValue(v)); };
}

/// Forward mode results in creating ops in the same region. We need to save
/// them to avoid trying to differentiate generated adjoint ops.
SmallVector<Operation *> savePrimalOps(Region *region) {
  size_t numOps =
      std::distance(region->getOps().begin(), region->getOps().end());
  SmallVector<Operation *> primalOps{numOps - 1};
  llvm::transform(region->front().without_terminator(), primalOps.begin(),
                  [](Operation &op) { return &op; });
  return primalOps;
}

static LogicalResult generateTangent(func::FuncOp tangentFunc, LAGradContext &ctx,
                                     ArrayAttr activeArgsAttr,
                                     ConversionPatternRewriter &rewriter,
                                     bool includePrimal, bool sparseSeed);

// TODO: Organize forward mode functions better.
LogicalResult populateJVP(Operation *op, LAGradContext &ctx,
                          IRMapping &env,
                          ConversionPatternRewriter &rewriter);

void updateActiveValues(LAGradContext &ctx, ValueRange from, ValueRange to) {
  for (auto pair : llvm::zip(from, to)) {
    if (ctx.activeValues.contains(std::get<0>(pair))) {
      ctx.activeValues.insert(std::get<1>(pair));
    }
  }
}

void updateActiveValues(LAGradContext &ctx, Operation *from, Operation *to,
                        IRMapping &map) {
  updateActiveValues(ctx, from->getResults(), to->getResults());

  from->walk([&](Operation *origBodyOp) {
    for (Value result : origBodyOp->getResults()) {
      if (map.contains(result) && ctx.activeValues.contains(result)) {
        ctx.activeValues.insert(map.lookup(result));
      }
    }
  });
}

LogicalResult callJVP(func::CallOp op, LAGradContext &ctx, IRMapping &env,
                      ConversionPatternRewriter &rewriter) {
  auto getTangentFunc = [&]() -> FailureOr<func::FuncOp> {
    std::string tangentFuncName = ("__tangent_" + op.getCallee()).str();
    if (auto existingFunc = ctx.moduleOp.lookupSymbol<func::FuncOp>(tangentFuncName))
      return existingFunc;

    auto originalFuncOp = ctx.moduleOp.lookupSymbol<func::FuncOp>(op.getCallee());
    SmallVector<int64_t> tangentsOf;
    for (auto operand : llvm::enumerate(op.getArgOperands())) {
      if (ctx.activeValues.contains(operand.value()))
        tangentsOf.push_back(operand.index());
    }
    func::FuncOp tangentFunc =
        copyFunctionDeclaration(originalFuncOp, tangentFuncName, rewriter);

    runActivityAnalysis(ctx, tangentFunc, rewriter.getI64ArrayAttr(tangentsOf));
    if (failed(generateTangent(tangentFunc, ctx,
                               rewriter.getI64ArrayAttr(tangentsOf), rewriter,
                               /*includePrimal=*/true, /*sparseSeed=*/false))) {
      return failure();
    }
    return tangentFunc;
  };

  auto tangentFuncResult = getTangentFunc();
  if (failed(tangentFuncResult)) {
    return failure();
  }
  func::FuncOp tangentFunc = tangentFuncResult.value();
  SmallVector<Value> newOperands;
  DenseMap<unsigned, unsigned> dualMapping;
  SmallVector<std::pair<unsigned, unsigned>> remappedValues;

  auto lookupE = lookupEnv(env, rewriter);
  unsigned idx = 0, origIdx = 0;
  for (auto operand : llvm::enumerate(op.getArgOperands())) {
    newOperands.push_back(operand.value());
    if (ctx.activeValues.contains(operand.value())) {
      newOperands.push_back(lookupE(operand.value()));
      idx++;
    }
    idx++;
    origIdx++;
  }

  auto dualCall =
      rewriter.create<func::CallOp>(op.getLoc(), tangentFunc, newOperands);

  idx = 0, origIdx = 0;
  for (auto result : op.getResults()) {
    remappedValues.push_back(std::make_pair(origIdx, idx));
    if (ctx.activeValues.contains(result)) {
      dualMapping[idx] = idx + 1;
      idx++;
    }
    idx++;
    origIdx++;
  }

  SmallVector<Value> replacedResults{static_cast<size_t>(op.getNumResults())};
  for (auto mapping : remappedValues) {
    replacedResults[mapping.first] = dualCall.getResult(mapping.second);
    updateActiveValues(ctx, replacedResults[mapping.first],
                       dualCall.getResult(mapping.second));
  }
  rewriter.replaceOp(op, replacedResults);

  for (auto mapping : dualMapping) {
    env.map(dualCall.getResult(mapping.first),
            dualCall.getResult(mapping.second));
  }

  return success();
}

LogicalResult linalgJVP(linalg::LinalgOp op, LAGradContext &ctx,
                        IRMapping &env,
                        ConversionPatternRewriter &rewriter) {
  SmallVector<Value> inputs, outputs;
  SmallVector<Type> outputTypes;
  SmallVector<AffineMap> indexingMaps;
  DenseMap<unsigned, unsigned> dualMapping;
  SmallVector<std::pair<unsigned, unsigned>> replacedPrimalMapping;

  SmallVector<utils::IteratorType> iteratorTypes = op.getIteratorTypesArray();

  unsigned idx = 0;
  unsigned origIdx = 0;

  auto lookupE = lookupEnv(env, rewriter);
  auto containsE = containsEnv(env, rewriter);
  for (OpOperand *input : op.getDpsInputOperands()) {
    inputs.push_back(input->get());
    indexingMaps.push_back(cast<AffineMapAttr>(op.getIndexingMaps()[input->getOperandNumber()]).getValue());
    replacedPrimalMapping.push_back(std::make_pair(origIdx, idx));
    if (containsE(rewriter.getRemappedValue(input->get()))) {
      inputs.push_back(lookupE(input->get()));
      indexingMaps.push_back(cast<AffineMapAttr>(op.getIndexingMaps()[input->getOperandNumber()]).getValue());

      dualMapping[idx] = idx + 1;
      ++idx;
    }
    ++idx;
    ++origIdx;
  }

  for (int64_t i = 0, e = op.getNumDpsInits(); i < e; ++i) {
    OpOperand *output = op.getDpsInitOperand(i);
    outputs.push_back(output->get());
    outputTypes.push_back(output->get().getType());
    indexingMaps.push_back(cast<AffineMapAttr>(op.getIndexingMaps()[output->getOperandNumber()]).getValue());
    replacedPrimalMapping.push_back(std::make_pair(origIdx, idx));
    if (containsE(output->get())) {
      Value dual = lookupE(output->get());
      outputs.push_back(dual);
      outputTypes.push_back(dual.getType());
      indexingMaps.push_back(cast<AffineMapAttr>(op.getIndexingMaps()[output->getOperandNumber()]).getValue());
      dualMapping[idx] = idx + 1;
      ++idx;
    }
    ++idx;
    ++origIdx;
  }

  auto terminator = cast<linalg::YieldOp>(op.getBlock()->getTerminator());
  IRMapping map;
  bool augmentFailed = false;
  auto newOp = rewriter.create<linalg::GenericOp>(
      op.getLoc(), outputTypes, inputs, outputs, indexingMaps, iteratorTypes,
      [&](OpBuilder &b, Location loc, ValueRange regionArgs) {
        PatternRewriter::InsertionGuard guard(rewriter);
        auto startOfBlock = b.saveInsertionPoint();
        rewriter.restoreInsertionPoint(startOfBlock);
        for (auto mapping : dualMapping) {
          env.map(regionArgs[mapping.first], regionArgs[mapping.second]);
        }

        for (auto pair : replacedPrimalMapping) {
          map.map(op.getBlock()->getArgument(pair.first),
                  regionArgs[pair.second]);
        }

        // Clone over original ops.
        SmallVector<Operation *> bodyOps;
        for (Operation &origOp : op.getBlock()->without_terminator()) {
          Operation *clonedOp = b.clone(origOp, map);
          bodyOps.push_back(clonedOp);
          map.map(origOp.getResults(), clonedOp->getResults());
        }

        // Generate JVPs.
        for (auto *bodyOp : bodyOps) {
          if (failed(populateJVP(bodyOp, ctx, env, rewriter))) {
            augmentFailed = true;
            return;
          }
        }

        // Find yield operands.
        SmallVector<Value> results;
        results.reserve(outputs.size());
        for (auto mapping : replacedPrimalMapping) {
          if (mapping.first >= op.getNumDpsInputs()) {
            Value newYieldOperand = map.lookup(
                terminator.getOperand(mapping.first - op.getNumDpsInputs()));
            results.push_back(newYieldOperand);
            if (dualMapping.count(mapping.second))
              results.push_back(lookupE(newYieldOperand));
          }
        }
        b.create<linalg::YieldOp>(loc, results);
      });
  if (augmentFailed) {
    return failure();
  }

  SmallVector<Value> replacedResults{static_cast<size_t>(op.getNumDpsInits())};
  for (auto mapping : replacedPrimalMapping) {
    if (mapping.first >= op.getNumDpsInputs()) {
      replacedResults[mapping.first - op.getNumDpsInputs()] =
          newOp.getResult(mapping.second - newOp.getNumDpsInputs());
    }
  }
  rewriter.replaceOp(op, replacedResults);

  for (auto mapping : dualMapping) {
    if (mapping.first >= newOp.getNumDpsInputs()) {
      env.map(newOp.getResult(mapping.first - newOp.getNumDpsInputs()),
              newOp.getResult(mapping.second - newOp.getNumDpsInputs()));
    }
  }
  return success();
}

LogicalResult ifJVP(scf::IfOp ifOp, LAGradContext &ctx,
                    IRMapping &env,
                    ConversionPatternRewriter &rewriter) {
  Location loc = ifOp.getLoc();
  SmallVector<Type> resultTypes;
  SmallVector<std::pair<unsigned, unsigned>> replacedPrimalMapping;
  DenseMap<unsigned, unsigned> dualMapping;
  unsigned idx = 0;
  unsigned origIdx = 0;
  for (OpResult result : ifOp.getResults()) {
    resultTypes.push_back(result.getType());
    replacedPrimalMapping.push_back(std::make_pair(origIdx, idx));
    // FIXME: need proper activity analysis
    if (isFloatOrFloatTensor(result.getType())) {
      // if (ctx.activeValues.contains(result)) {
      resultTypes.push_back(result.getType());
      dualMapping[idx] = idx + 1;
      idx++;
    }
    idx++;
    origIdx++;
  }
  auto lookupE = lookupEnv(env, rewriter);
  auto containsE = containsEnv(env, rewriter);

  auto builderFunc = [&](Block &block) {
    return [&](OpBuilder &builder, Location loc) {
      PatternRewriter::InsertionGuard guard(rewriter);

      auto startOfBlock = builder.saveInsertionPoint();
      rewriter.restoreInsertionPoint(startOfBlock);
      IRMapping map;

      // Clone over the original ops.
      SmallVector<Operation *> bodyOps;
      for (Operation &origOp : block.without_terminator()) {
        Operation *clonedOp = builder.clone(origOp, map);
        bodyOps.push_back(clonedOp);
        map.map(origOp.getResults(), clonedOp->getResults());
        for (OpResult r : origOp.getResults()) {
          if (containsE(r)) {
            env.map(clonedOp->getResult(r.getResultNumber()), lookupE(r));
          }
        }
      }

      // Generate JVPs.
      for (auto *bodyOp : bodyOps) {
        if (failed(populateJVP(bodyOp, ctx, env, rewriter))) {
          llvm_unreachable("populate jvp failed inside if op\n");
          return;
        }
      }
      SmallVector<Value> results;
      auto yieldOp = cast<scf::YieldOp>(block.getTerminator());

      for (auto mapping : replacedPrimalMapping) {
        Value newYieldOperand = rewriter.getRemappedValue(
            map.lookupOrDefault(yieldOp.getOperand(mapping.first)));
        results.push_back(newYieldOperand);
        if (dualMapping.count(mapping.second)) {
          results.push_back(lookupE(newYieldOperand));
        }
      }
      rewriter.create<scf::YieldOp>(loc, results);
    };
  };

  auto augmentedIf = rewriter.create<scf::IfOp>(
      loc, resultTypes, ifOp.getCondition());
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(augmentedIf.thenBlock());
    auto thenBuilder = builderFunc(*ifOp.thenBlock());
    thenBuilder(rewriter, augmentedIf.thenBlock()->getParentOp()->getLoc());
  }
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(augmentedIf.elseBlock());
    auto elseBuilder = builderFunc(*ifOp.elseBlock());
    elseBuilder(rewriter, augmentedIf.elseBlock()->getParentOp()->getLoc());
  }
  SmallVector<Value> replacedResults{replacedPrimalMapping.size()};
  for (auto mapping : replacedPrimalMapping) {
    replacedResults[mapping.first] = augmentedIf.getResult(mapping.second);
    ifOp.getResult(mapping.first)
        .replaceAllUsesWith(augmentedIf.getResult(mapping.second));
  }
  rewriter.replaceOp(ifOp, replacedResults);
  for (auto mapping : dualMapping) {
    env.map(augmentedIf.getResult(mapping.first),
            augmentedIf.getResult(mapping.second));
  }
  return success();
}

LogicalResult forLoopJVP(scf::ForOp forOp, LAGradContext &ctx,
                         IRMapping &env,
                         ConversionPatternRewriter &rewriter) {
  SmallVector<Value> iterArgInits;
  DenseMap<unsigned, unsigned> dualMapping;
  SmallVector<std::pair<unsigned, unsigned>> replacedPrimalMapping;
  auto lookupE = lookupEnv(env, rewriter);
  auto containsE = containsEnv(env, rewriter);
  unsigned idx = 0;
  unsigned originalIdx = 0;
  for (Value iterOperand : forOp.getInitArgs()) {
    iterArgInits.push_back(iterOperand);
    replacedPrimalMapping.push_back(std::make_pair(originalIdx, idx));
    if (containsE(iterOperand)) {
      iterArgInits.push_back(lookupE(iterOperand));
      dualMapping[idx] = idx + 1;
      ++idx;
    }
    ++idx;
    ++originalIdx;
  }

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  // Create a new for op because we need to pass additional iteration args
  // with the dual numbers.
  IRMapping map;
  bool augmentFailed = false;
  auto augmentedFor = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),
      iterArgInits,
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange regionArgs) {
        PatternRewriter::InsertionGuard guard(rewriter);
        auto startOfBlock = builder.saveInsertionPoint();
        rewriter.restoreInsertionPoint(startOfBlock);

        // The region arg mapping must match the iter operand mapping.
        for (auto mapping : dualMapping) {
          env.map(regionArgs[mapping.first], regionArgs[mapping.second]);
        }
        for (auto mapping : replacedPrimalMapping) {
          map.map(forOp.getRegionIterArgs()[mapping.first],
                  regionArgs[mapping.second]);
        }
        map.map(forOp.getInductionVar(), iv);

        // Clone over the original ops.
        SmallVector<Operation *> bodyOps;
        for (Operation &origOp : forOp.getBody()->without_terminator()) {
          Operation *clonedOp = builder.clone(origOp, map);
          bodyOps.push_back(clonedOp);
          map.map(origOp.getResults(), clonedOp->getResults());
          updateActiveValues(ctx, &origOp, clonedOp, map);
        }

        // Generate JVPs.
        for (auto *bodyOp : bodyOps) {
          if (failed(populateJVP(bodyOp, ctx, env, rewriter))) {
            augmentFailed = true;
            return;
          }
        }

        SmallVector<Value> results;
        results.reserve(regionArgs.size());
        for (auto mapping : replacedPrimalMapping) {
          Value newYieldOperand = rewriter.getRemappedValue(
              map.lookup(yieldOp.getOperand(mapping.first)));
          results.push_back(newYieldOperand);
          if (dualMapping.count(mapping.second))
            results.push_back(lookupE(newYieldOperand));
        }
        rewriter.create<scf::YieldOp>(loc, results);
      });
  if (augmentFailed) {
    return failure();
  }

  SmallVector<Value> replacedResults{replacedPrimalMapping.size()};
  for (auto mapping : replacedPrimalMapping) {
    replacedResults[mapping.first] = augmentedFor.getResult(mapping.second);
    forOp.getResult(mapping.first)
        .replaceAllUsesWith(augmentedFor.getResult(mapping.second));
  }
  rewriter.replaceOp(forOp, replacedResults);
  for (auto mapping : dualMapping) {
    env.map(augmentedFor.getResult(mapping.first),
            augmentedFor.getResult(mapping.second));
  }
  // rewriter.eraseOp(forOp);
  return success();
}

LogicalResult populateJVP(Operation *op, LAGradContext &ctx,
                          IRMapping &env,
                          ConversionPatternRewriter &rewriter) {
  rewriter.setInsertionPointAfter(op);
  Location loc = op->getLoc();
  Value jvp;
  if (op->getNumResults() == 0) {
    return success();
  }
  // Need to hit these before the linalg op interface.
  if (auto initTensorOp = dyn_cast<tensor::EmptyOp>(op)) {
    jvp = rewriter.create<tensor::EmptyOp>(loc, initTensorOp.getType(),
                                                initTensorOp.getDynamicSizes());
    env.map(initTensorOp.getResult(), jvp);
    return success();
  } else if (auto fillOp = dyn_cast<linalg::FillOp>(op)) {
    auto dualFillOp =
        rewriter.create<linalg::FillOp>(loc, fillOp.value(), fillOp.output());
    env.map(fillOp.getResults(), dualFillOp.getResults());
    return success();
  } else if (auto sitofpOp = dyn_cast<arith::SIToFPOp>(op)) {
    env.map(sitofpOp.getResult(), getZero(loc, sitofpOp.getResult(), rewriter));
    return success();
  }
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    return ifJVP(ifOp, ctx, env, rewriter);
  } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    return forLoopJVP(forOp, ctx, env, rewriter);
  } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    return linalgJVP(linalgOp, ctx, env, rewriter);
  } else if (auto callOp = dyn_cast<func::CallOp>(op)) {
    return callJVP(callOp, ctx, env, rewriter);
  }

  assert(op->getNumResults() == 1);
  if (op->getResult(0).getType().isIntOrIndex()) {
    return success();
  }

  auto lookupE = lookupEnv(env, rewriter);
  auto containsE = containsEnv(env, rewriter);

  if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
    if (isFloatOrFloatTensor(constOp.getType())) {
      jvp = getZero(constOp.getLoc(), constOp.getResult(), rewriter);
    } else if (auto attr = constOp.getValue().dyn_cast<FloatAttr>()) {
      // constOp.getType().getIntOrFloatBitWidth()
      jvp = rewriter.create<arith::ConstantOp>(loc,
                                               rewriter.getF64FloatAttr(0.0));
    } else {
      return success();
    }
  } else if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
    Value trueDual = containsE(selectOp.getTrueValue())
                         ? lookupE(selectOp.getTrueValue())
                         : getZero(loc, selectOp.getTrueValue(), rewriter);
    Value falseDual = containsE(selectOp.getFalseValue())
                          ? lookupE(selectOp.getFalseValue())
                          : getZero(loc, selectOp.getFalseValue(), rewriter);
    jvp = rewriter.create<arith::SelectOp>(loc, selectOp.getCondition(), trueDual,
                                    falseDual);
  } else if (auto addfOp = dyn_cast<arith::AddFOp>(op)) {
    if (!containsE(addfOp.getLhs())) {
      jvp = lookupE(addfOp.getRhs());
    } else if (!containsE(addfOp.getRhs())) {
      jvp = lookupE(addfOp.getLhs());
    } else {
      jvp = rewriter.create<arith::AddFOp>(loc, lookupE(addfOp.getLhs()),
                                           lookupE(addfOp.getRhs()));
    }
  } else if (auto subfOp = dyn_cast<arith::SubFOp>(op)) {
    jvp = rewriter.create<arith::SubFOp>(loc, lookupE(subfOp.getLhs()),
                                         lookupE(subfOp.getRhs()));
  } else if (auto mulfOp = dyn_cast<arith::MulFOp>(op)) {
    if (!containsE(mulfOp.getLhs())) {
      jvp = rewriter.create<arith::MulFOp>(loc, mulfOp.getLhs(),
                                           lookupE(mulfOp.getRhs()));
    } else if (!containsE(mulfOp.getRhs())) {
      jvp = rewriter.create<arith::MulFOp>(loc, mulfOp.getRhs(),
                                           lookupE(mulfOp.getLhs()));
    } else {
      jvp = rewriter.create<arith::AddFOp>(
          loc,
          rewriter.create<arith::MulFOp>(loc, mulfOp.getRhs(),
                                         lookupE(mulfOp.getLhs())),
          rewriter.create<arith::MulFOp>(loc, mulfOp.getLhs(),
                                         lookupE(mulfOp.getRhs())));
    }
  } else if (auto negfOp = dyn_cast<arith::NegFOp>(op)) {
    jvp = rewriter.create<arith::NegFOp>(loc, lookupE(negfOp.getOperand()));
  } else if (auto divfOp = dyn_cast<arith::DivFOp>(op)) {
    Value lhsDual = rewriter.create<arith::DivFOp>(loc, lookupE(divfOp.getLhs()),
                                                   divfOp.getRhs());

    // RHS
    jvp = rewriter.create<arith::MulFOp>(op->getLoc(), lookupE(divfOp.getRhs()),
                                         divfOp.getLhs());
    jvp = rewriter.create<arith::NegFOp>(op->getLoc(), jvp);
    Value denom = rewriter.create<arith::MulFOp>(op->getLoc(), divfOp.getRhs(),
                                                 divfOp.getRhs());
    Value rhsDual = rewriter.create<arith::DivFOp>(op->getLoc(), jvp, denom);

    jvp = rewriter.create<arith::AddFOp>(loc, lhsDual, rhsDual);
  } else if (auto sqrtOp = dyn_cast<math::SqrtOp>(op)) {
    auto half = constLike(loc, sqrtOp.getOperand(), 0.5, rewriter);
    jvp = rewriter.create<arith::DivFOp>(
        loc,
        rewriter.create<arith::MulFOp>(loc, lookupE(sqrtOp.getOperand()), half),
        sqrtOp.getResult());
  } else if (auto sinOp = dyn_cast<math::SinOp>(op)) {
    jvp = rewriter.create<arith::MulFOp>(
        loc, lookupE(sinOp.getOperand()),
        rewriter.create<math::CosOp>(loc, sinOp.getOperand()));
  } else if (auto cosOp = dyn_cast<math::CosOp>(op)) {
    jvp = rewriter.create<arith::MulFOp>(
        loc, lookupE(cosOp.getOperand()),
        rewriter.create<arith::NegFOp>(
            loc, rewriter.create<math::SinOp>(loc, cosOp.getOperand())));
  } else if (auto expOp = dyn_cast<math::ExpOp>(op)) {
    jvp = rewriter.create<arith::MulFOp>(
        loc, lookupE(expOp.getOperand()),
        rewriter.create<math::ExpOp>(loc, expOp.getOperand()));
  } else if (auto logOp = dyn_cast<math::LogOp>(op)) {
    jvp = rewriter.create<arith::DivFOp>(loc, lookupE(logOp.getOperand()),
                                         logOp.getOperand());
  } else if (auto tanhOp = dyn_cast<math::TanhOp>(op)) {
    auto exp = rewriter.create<math::ExpOp>(loc, tanhOp.getOperand());
    auto negexp = rewriter.create<math::ExpOp>(
        loc, rewriter.create<arith::NegFOp>(loc, tanhOp.getOperand()));
    auto numerator = rewriter.create<arith::AddFOp>(loc, exp, negexp);
    auto half = constLike(loc, tanhOp.getOperand(), 0.5, rewriter);
    auto cosh = rewriter.create<arith::MulFOp>(loc, numerator, half);
    auto coshsquared = rewriter.create<arith::MulFOp>(loc, cosh, cosh);
    jvp = rewriter.create<arith::DivFOp>(loc, lookupE(tanhOp.getOperand()),
                                         coshsquared);
  } else if (auto insertOp = dyn_cast<tensor::InsertOp>(op)) {
    jvp = rewriter.create<tensor::InsertOp>(loc, lookupE(insertOp.getScalar()),
                                            lookupE(insertOp.getDest()),
                                            insertOp.getIndices());
  } else if (auto extractOp = dyn_cast<tensor::ExtractOp>(op)) {
    jvp = rewriter.create<tensor::ExtractOp>(
        loc, lookupE(rewriter.getRemappedValue(extractOp.getTensor())),
        extractOp.getIndices());
  } else if (auto insertSliceOp = dyn_cast<tensor::InsertSliceOp>(op)) {
    jvp = rewriter.create<tensor::InsertSliceOp>(
        loc, lookupE(rewriter.getRemappedValue(insertSliceOp.getSource())),
        lookupE(insertSliceOp.getDest()), insertSliceOp.getMixedOffsets(),
        insertSliceOp.getMixedSizes(), insertSliceOp.getMixedStrides());
  } else if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(op)) {
    jvp = rewriter.create<tensor::ExtractSliceOp>(
        loc, extractSliceOp.getType(),
        lookupE(rewriter.getRemappedValue(extractSliceOp.getSource())),
        extractSliceOp.getMixedOffsets(), extractSliceOp.getMixedSizes(),
        extractSliceOp.getMixedStrides());
  } else {
    op->emitOpError() << "unhandled op";
    return failure();
  }
  env.map(op->getResult(0), jvp);
  return success();
}

static LogicalResult generateTangent(func::FuncOp tangentFunc, LAGradContext &ctx,
                                     ArrayAttr activeArgsAttr,
                                     ConversionPatternRewriter &rewriter,
                                     bool includePrimal, bool sparseSeed) {
  PatternRewriter::InsertionGuard guard(rewriter);
  Region *region = tangentFunc.getCallableRegion();
  if (!region) {
    tangentFunc.emitOpError("Cannot perform AD on op without body");
    return failure();
  }
  assert(region->hasOneBlock());
  rewriter.setInsertionPointToStart(&region->front());

  // env maps primal values to their dual values.
  IRMapping env;
  size_t idx = 0;
  // Modify the function signature
  DenseSet<size_t> activeArgs;
  if (!activeArgsAttr || activeArgsAttr.empty()) {
    activeArgs.insert(0);
  }
  for (APInt attr : activeArgsAttr.getAsValueRange<IntegerAttr>()) {
    activeArgs.insert(attr.getSExtValue());
  }
  SmallVector<BlockArgument> originalArgs{tangentFunc.getArguments().begin(),
                                          tangentFunc.getArguments().end()};
  for (auto pair : llvm::enumerate(originalArgs)) {
    BlockArgument arg = pair.value();
    if (activeArgs.contains(pair.index())) {
      ++idx;
      Type argType = arg.getType();
      if (sparseSeed) {
        assert(activeArgs.size() == 1 &&
               "Expected one active argument when 'sparse' attribute is set");
        auto tensorType = argType.cast<RankedTensorType>();
        argType = RankedTensorType::get(tensorType.getShape(),
                                        tensorType.getElementType(),
                                        rewriter.getStringAttr("onehot"));
        ctx.sparseValues.insert(arg);
      }
      tangentFunc.insertArgument(idx, argType, DictionaryAttr(), tangentFunc.getLoc());
      env.map(arg, tangentFunc.getArgument(idx));
    } else if (isFloatOrFloatTensor(arg.getType())) {
      // TODO: modify populateJVP to not need these dummy zero values.
      env.map(arg, getZero(tangentFunc.getLoc(), arg, rewriter));
    }
    ++idx;
  }

  if (includePrimal) {
    SmallVector<Type> results(tangentFunc.getFunctionType().getResults().begin(),
                              tangentFunc.getFunctionType().getResults().end());
    idx = 0;
    for (Type resultType : results) {
      ++idx;
      tangentFunc.insertResult(idx, resultType, {});
    }
    ++idx;
  }

  SmallVector<Operation *> primalOps = savePrimalOps(region);
  for (Operation *op : primalOps) {
    if (failed(populateJVP(op, ctx, env, rewriter))) {
      return failure();
    }
  }

  Operation *terminator = region->front().getTerminator();
  SmallVector<Value> results;
  auto lookupE = lookupEnv(env, rewriter);
  for (Value operand : terminator->getOperands()) {
    if (includePrimal)
      results.push_back(operand);
    results.push_back(lookupE(rewriter.getRemappedValue(operand)));
  }
  rewriter.create<func::ReturnOp>(terminator->getLoc(), results);
  rewriter.eraseOp(terminator);
  return success();
}

class ForwardModeAD : public OpConversionPattern<TangentOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(TangentOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto tangentFunc = generateTangentFunc(op, rewriter);
    if (!tangentFunc)
      return failure();

    rewriter.replaceOpWithNewOp<func::CallOp>(op, tangentFunc, op.getOperands());
    return success();
  }

private:
  static func::FuncOp generateTangentFunc(TangentOp op,
                                    ConversionPatternRewriter &rewriter) {
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto originalFuncOp = moduleOp.lookupSymbol<func::FuncOp>(op.getFAttr());
    std::string tangentFuncName =
        ("__tangent_" + originalFuncOp.getName()).str();
    if (auto existingFunc = moduleOp.lookupSymbol<func::FuncOp>(tangentFuncName))
      return existingFunc;

    func::FuncOp tangentFunc =
        copyFunctionDeclaration(originalFuncOp, tangentFuncName, rewriter);
    auto tangentOf = op->getAttrOfType<ArrayAttr>("of");
    bool includePrimal = op->hasAttrOfType<UnitAttr>("include_primal");
    bool sparseSeed = op->hasAttrOfType<UnitAttr>("sparse");
    if (!tangentOf) {
      tangentOf = rewriter.getArrayAttr({});
    }
    LAGradContext lagradctx{moduleOp};
    DEBUGpopulateFunc(lagradctx.debug_names, tangentFunc);
    runActivityAnalysis(lagradctx, tangentFunc, tangentOf);
    if (failed(generateTangent(tangentFunc, lagradctx, tangentOf, rewriter,
                               includePrimal, sparseSeed))) {
      return nullptr;
    }
    return tangentFunc;
  }
};
} // namespace

void mlir::lagrad::populateLAGradTransforms(RewritePatternSet &patterns,
                                            MLIRContext *ctx) {
  patterns.add<ForwardModeAD>(ctx);
}
