/**
 * A custom pass meant to fill gaps in bufferizing tensor ops.
 * Motivated by a lack of bufferization for the tensor.insert op.
 */
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "LAGrad/Analysis.h"
#include "LAGrad/Logger.h"
#include "LAGrad/Passes.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using llvm::errs;
constexpr bool force_copy = false;

namespace {
class BufferizeTypeConverter : public TypeConverter {
public:
  BufferizeTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
  }
};
} // namespace

namespace {

void bufferizeLinalgOp(linalg::LinalgOp linalgOp, ValueRange outputBuffers,
                       ValueRange results, const TypeConverter &typeConverter,
                       ConversionPatternRewriter &rewriter) {
  PatternRewriter::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(linalgOp);

  SmallVector<Value> newOperands;
  newOperands.reserve(linalgOp.getNumDpsInputs() + linalgOp.getNumDpsInits());
  for (OpOperand *inputOperand : linalgOp.getDpsInputOperands()) {
    newOperands.push_back(rewriter.create<bufferization::ToMemrefOp>(
        linalgOp.getLoc(),
        typeConverter.convertType(inputOperand->get().getType()).cast<MemRefType>(),
        inputOperand->get()));
  }
  newOperands.append(outputBuffers.begin(), outputBuffers.end());

  assert(linalgOp->getNumRegions() == 1 &&
         "expected linalg op to have 1 region");
  OperationState state(linalgOp.getLoc(), linalgOp->getName());
  state.addOperands(newOperands);
  state.addTypes(TypeRange{});
  state.addAttributes(linalgOp->getAttrs());
  auto newOp = cast<linalg::LinalgOp>(rewriter.create(state));
  newOp->getRegion(0).takeBody(linalgOp->getRegion(0));

  // The number of output buffers should always be 1 for now.
  rewriter.replaceOp(linalgOp, results);
}

class BufferizeInsertExtractPair : public ConversionPattern {
public:
  BufferizeInsertExtractPair(const InsertExtractAnalysis &ieAnalysis,
                             TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), /*benefit=*/3,
                          ctx),
        ieAnalysis{ieAnalysis} {}
  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (!isa<tensor::ExtractSliceOp>(op)) {
      return failure();
    }
    auto extractSliceOp = cast<tensor::ExtractSliceOp>(op);
    if (!ieAnalysis.isPairedExtractSlice(extractSliceOp)) {
      return failure();
    }
    auto insertSliceOp = ieAnalysis.getPairedInsertSlice(extractSliceOp);
    auto sliceType = extractSliceOp.getType();
    auto sourceType = getTypeConverter()
                          ->convertType(extractSliceOp.getSourceType())
                          .cast<MemRefType>();

    auto resultType =
        memref::SubViewOp::inferRankReducedResultType(
            sliceType.getRank(), sourceType, extractSliceOp.getMixedOffsets(),
            extractSliceOp.getMixedSizes(), extractSliceOp.getMixedStrides())
            .cast<MemRefType>();
    auto identityResultType =
        getTypeConverter()->convertType(sliceType).cast<MemRefType>();
    auto source = rewriter.create<bufferization::ToMemrefOp>(
        op->getLoc(), sourceType, extractSliceOp.getSource());
    auto subview = rewriter.create<memref::SubViewOp>(
        op->getLoc(), resultType, source, extractSliceOp.getMixedOffsets(),
        extractSliceOp.getMixedSizes(), extractSliceOp.getMixedStrides());
    auto casted = rewriter.create<memref::CastOp>(
        op->getLoc(), identityResultType, subview.getResult());
    auto loaded = rewriter.create<bufferization::ToTensorOp>(op->getLoc(), casted);
    rewriter.replaceOp(op, loaded.getResult());
    rewriter.replaceOp(insertSliceOp, extractSliceOp.getSource());

    // Need to ensure linalg ops that write to the extractSliceOp are bufferized
    // in-place.
    for (auto &use : extractSliceOp.getResult().getUses()) {
      if (ieAnalysis.isLinalgMarkedForBufferization(use.getOwner())) {
        auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner());
        if (linalgOp.isInitTensor(&use)) {
          bufferizeLinalgOp(linalgOp, /*outputBuffers=*/subview.getResult(),
                            /*results=*/loaded.getResult(), *getTypeConverter(),
                            rewriter);
        }
      }
    }
    return success();
  }

private:
  const InsertExtractAnalysis &ieAnalysis;
};

static bool hasUseAfter(Value dest, Value result) {
  DominanceInfo dom;
  bool hasUseAfter = false;

  for (auto user : dest.getUsers()) {
    if (dom.properlyDominates(result, user)) {
      hasUseAfter = true;
    }
  }
  return hasUseAfter;
}

class BufferizeFillOp : public OpConversionPattern<linalg::FillOp> {
public:
  BufferizeFillOp(TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1) {}

  LogicalResult
  matchAndRewrite(linalg::FillOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1) {
      return failure();
    }

    Value outputVal = op.getDpsInitOperand(0)->get();
    bool mustCopy = !hasUseAfter(outputVal, op.getResult(0));
    if (mustCopy) {
      auto outputType = outputVal.getType().cast<ShapedType>();
      if (!outputType.hasStaticShape()) {
        return failure();
      }
      // SmallVector<Value> dynamicSizes{
      //     static_cast<size_t>(outputType.getNumDynamicDims())};
      // for (auto pair : llvm::enumerate(outputType.getShape())) {
      //   if (ShapedType::isDynamic(pair.value())) {
      //     dynamicSizes.push_back(rewriter.create<memref::DimOp>(
      //         op.getLoc(), operands[0], pair.index()));
      //   }
      // }

      Value copy = rewriter.create<memref::AllocOp>(
          op.getLoc(), getTypeConverter()
                           ->convertType(outputVal.getType())
                           .cast<MemRefType>());
      rewriter.create<linalg::FillOp>(op.getLoc(), adaptor.getInputs()[0], copy);
      rewriter.replaceOp(op, copy);
    } else {
      rewriter.create<linalg::FillOp>(op.getLoc(), adaptor.getInputs()[0], adaptor.getOutputs()[0]);
      rewriter.replaceOp(op, adaptor.getOutputs()[0]);
    }
    return success();
  }
};

class BufferizeInsertOp : public OpConversionPattern<tensor::InsertOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(tensor::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    bufferization::ToTensorOp loaded;

    // This is very naïve.
    DominanceInfo dom;
    bool hasUseAfter = false;

    for (auto user : op.getDest().getUsers()) {
      if (dom.properlyDominates(op.getResult(), user)) {
        hasUseAfter = true;
      }
    }
    // constexpr bool bufferize_insert_in_place = false;
    // if (bufferize_insert_in_place) {
    if (!hasUseAfter && !force_copy) {
      // This first implementation updates the tensor in place rather than
      // returning a copy per the spec. Watch out for bugs this may cause.
      rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.getScalar(),
                                       adaptor.getDest(), adaptor.getIndices());
      loaded =
          rewriter.create<bufferization::ToTensorOp>(op.getLoc(), adaptor.getDest());
    } else {
      auto space = rewriter.create<memref::AllocOp>(
          op.getLoc(), adaptor.getDest().getType().cast<MemRefType>());
      rewriter.create<linalg::CopyOp>(op.getLoc(), ValueRange{adaptor.getDest()}, ValueRange{space});
      rewriter.create<memref::StoreOp>(op.getLoc(), adaptor.getScalar(), space,
                                       adaptor.getIndices());
      loaded = rewriter.create<bufferization::ToTensorOp>(op.getLoc(), space);
    }

    op.replaceAllUsesWith(loaded.getResult());
    rewriter.eraseOp(op);
    return success();
  }
};

/**
 * Replace tensor.extract_slice ops with a combination of memref.subview and
 * memref.cast.
 */
class BufferizeExtractSliceOp
    : public OpConversionPattern<tensor::ExtractSliceOp> {
private:
  const InsertExtractAnalysis &ieAnalysis;

public:
  using OpConversionPattern::OpConversionPattern;
  BufferizeExtractSliceOp(const InsertExtractAnalysis &ieAnalysis,
                          TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1),
        ieAnalysis{ieAnalysis} {}
  LogicalResult
  matchAndRewrite(tensor::ExtractSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getDroppedDims().empty() || ieAnalysis.isPairedExtractSlice(op)) {
      // Only deal with rank reduced cases
      // If in the ieAnalysis, the dedicated pass will deal with it.
      return failure();
    }
    auto resultTensorType =
        op.getResult().getType().dyn_cast_or_null<RankedTensorType>();
    auto resultRank = resultTensorType.getRank();
    // Curently only deal with 2D -> 1D and 3D -> 2D rank reduction cases.
    if (!resultTensorType ||
        op.getOffsetSizeAndStrideStartOperandIndex() != 1) {
      return failure();
    }
    auto sourceType = adaptor.getSource().getType().cast<MemRefType>();
    Value source = adaptor.getSource();
    // Ugly, brittle hack to get extract_slice with compressed GMMs working.
    // This results in the source having a fully dynamic layout map, which is
    // okay, but partial bufferization appears to expect an identity layout
    // map at the end of this transformation.
    if (sourceType.getDimSize(sourceType.getRank() - 1) == 1) {
      source = rewriter.create<memref::CastOp>(
          op.getLoc(), sourceType, source);
    }

    auto resultType =
        memref::SubViewOp::inferRankReducedResultType(
            resultRank, sourceType, op.getMixedOffsets(),
            op.getMixedSizes(), op.getMixedStrides())
            .cast<MemRefType>();
    auto identityResultType =
        getTypeConverter()->convertType(resultTensorType).cast<MemRefType>();

    auto subview = rewriter.create<memref::SubViewOp>(
        op.getLoc(), resultType, source, op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());

    DominanceInfo dom;
    bool hasWriteAfter = false;

    for (auto user : op.getSource().getUsers()) {
      if (dom.properlyDominates(op.getResult(), user)) {
        hasWriteAfter = true;
      }
    }

    // Need to consider writes to tensors that alias this one.
    // This causes a slow-down with GMM/main term, have yet to look into why.

    // This is a coarse-grained heuristic.
    // SmallVector<Value> frontier{op.getSource()};
    // ValueSet derivedFromSource;
    // runTopDownDFS(frontier, derivedFromSource);
    // for (Value derivedValue : derivedFromSource) {
    //   Operation *definingOp = derivedValue.getDefiningOp();
    //   if (definingOp && dom.properlyDominates(op.getResult(), definingOp) &&
    //       ((dyn_cast<tensor::InsertSliceOp>(definingOp) ||
    //         dyn_cast<tensor::InsertOp>(definingOp)))) {
    //     hasWriteAfter = true;
    //   }
    // }
    if (!hasWriteAfter && !force_copy) {
      // rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op,
      //                                                   subview.getResult());
      rewriter.replaceOpWithNewOp<memref::CastOp>(op, identityResultType,
                                                  subview.getResult());
    } else {
      auto dest =
          rewriter.create<memref::AllocOp>(op.getLoc(), identityResultType);
      rewriter.create<linalg::CopyOp>(op.getLoc(), ValueRange{subview.getResult()}, ValueRange{dest});
      rewriter.replaceOp(op, dest.getResult());
    }
    return success();
  }
};

class BufferizeInsertSliceOp
    : public OpConversionPattern<tensor::InsertSliceOp> {
private:
  const InsertExtractAnalysis &ieAnalysis;

public:
  using OpConversionPattern::OpConversionPattern;
  BufferizeInsertSliceOp(const InsertExtractAnalysis &ieAnalysis,
                         TypeConverter &typeConverter, MLIRContext *ctx)
      : OpConversionPattern(typeConverter, ctx, /*benefit=*/1),
        ieAnalysis{ieAnalysis} {}
  LogicalResult
  matchAndRewrite(tensor::InsertSliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (ieAnalysis.isPairedInsertSlice(op)) {
      // This op will be dealt with in the dedicated extract/insert slice pair
      // pass.
      return failure();
    }

    DominanceInfo dom;
    bool hasUseAfter = false;

    for (auto user : op.getDest().getUsers()) {
      if (dom.properlyDominates(op.getResult(), user)) {
        hasUseAfter = true;
      }
    }

    auto sliceType =
        memref::SubViewOp::inferRankReducedResultType(
            op.getSourceType().getRank(),
            adaptor.getDest().getType().cast<MemRefType>(), op.getMixedOffsets(),
            op.getMixedSizes(), op.getMixedStrides())
            .cast<MemRefType>();
    if (!hasUseAfter) {
      auto subview = rewriter.create<memref::SubViewOp>(
          op.getLoc(), sliceType, adaptor.getDest(), op.getMixedOffsets(),
          op.getMixedSizes(), op.getMixedStrides());
      rewriter.create<linalg::CopyOp>(op.getLoc(), ValueRange{adaptor.getSource()}, ValueRange{subview.getResult()});
      rewriter.replaceOp(op, adaptor.getDest());
    } else {
      auto dest = rewriter.create<memref::AllocOp>(
          op.getLoc(), adaptor.getDest().getType().dyn_cast<MemRefType>());
      rewriter.create<linalg::CopyOp>(op.getLoc(), ValueRange{adaptor.getDest()}, ValueRange{dest});
      auto subview = rewriter.create<memref::SubViewOp>(
          op.getLoc(), sliceType, dest, op.getMixedOffsets(),
          op.getMixedSizes(), op.getMixedStrides());

      rewriter.create<linalg::CopyOp>(op.getLoc(), ValueRange{adaptor.getSource()}, ValueRange{subview.getResult()});
      rewriter.replaceOp(op, dest.getResult());
    }
    return success();
  }
};

class BufferizeLinalgOp
    : public OpInterfaceConversionPattern<linalg::LinalgOp> {
public:
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;
  LogicalResult
  matchAndRewrite(linalg::LinalgOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.hasPureBufferSemantics()) {
      return failure();
    }
    DominanceInfo dom;
    bool hasUseAfter = false;
    for (auto user : op.getDpsInitOperand(0)->get().getUsers()) {
      if (dom.properlyDominates(op->getResult(0), user)) {
        hasUseAfter = true;
      }
    }
    SmallVector<Value> newOperands{operands.begin(), operands.end()};
    Operation *parentOp =
        op.getDpsInitOperand(0)->get().getDefiningOp()
            ?: op.getDpsInitOperand(0)->get().getParentRegion()->getParentOp();
    bool isImmutableArg = false;
    if (auto funcOp = dyn_cast<func::FuncOp>(parentOp)) {
      auto blockArg = op.getDpsInitOperand(0)->get().cast<BlockArgument>();
      // Function arguments are immutable by default.
      isImmutableArg = !static_cast<bool>(
          funcOp.getArgAttr(blockArg.getArgNumber(), "linalg.mutable"));
    }
    bool isConstantMemory = isa_and_nonnull<arith::ConstantOp>(parentOp);
    bool mustCopy = hasUseAfter || isConstantMemory || isImmutableArg;

    assert(op.getNumDpsInits() == 1);
    if (mustCopy) {
      errs() << "Analysis determined we must copy: hasUseAfter " << hasUseAfter
             << " isConstantMemory: " << isConstantMemory
             << " isImmutableArg: " << isImmutableArg << "\n";
      auto space = rewriter.create<memref::AllocOp>(
          op.getLoc(), newOperands.back().getType().cast<MemRefType>());
      rewriter.create<linalg::CopyOp>(op.getLoc(), ValueRange{newOperands.back()}, ValueRange{space.getResult()});
      newOperands[newOperands.size() - 1] = space.getResult();
    } else {
      op.emitRemark() << "Made in-place update";
    }
    SmallVector<Value> results;
    results.reserve(op->getNumResults());
    Operation *unknownOp = op;
    auto linalgOp = cast<linalg::LinalgOp>(unknownOp);
    assert(op->getNumRegions() == 1 && "expected linalg op to have 1 region");
    OperationState state(op.getLoc(), op->getName());
    state.addOperands(newOperands);
    state.addTypes(TypeRange{});
    state.addAttributes(op->getAttrs());
    auto newOp = cast<linalg::LinalgOp>(rewriter.create(state));
    newOp->getRegion(0).takeBody(op->getRegion(0));
    for (int64_t i = 0, e = newOp.getNumDpsInits(); i < e; ++i) {
      results.push_back(rewriter.create<bufferization::ToTensorOp>(
          op.getLoc(), newOp.getDpsInitOperand(i)->get()));
    }

    rewriter.replaceOp(op, results);
    return success();
  }
};

struct StandaloneBufferizePass : public OperationPass<ModuleOp> {
  StandaloneBufferizePass()
      : ::mlir::OperationPass<ModuleOp>(
            ::mlir::TypeID::get<StandaloneBufferizePass>()) {}
  StandaloneBufferizePass(const StandaloneBufferizePass &other)
      : ::mlir::OperationPass<ModuleOp>(other) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect, linalg::LinalgDialect>();
  }
  StringRef getArgument() const override { return "standalone-bufferize"; }
  StringRef getDescription() const override {
    return "Bufferize tensor ops required by the standalone dialect that "
           "aren't bufferized elsewhere.";
  }

  /// Returns the derived pass name.
  StringRef getName() const override {
    return llvm::getTypeName<StandaloneBufferizePass>();
  }

  /// A clone method to create a copy of this pass.
  std::unique_ptr<Pass> clonePass() const override {
    return std::make_unique<StandaloneBufferizePass>(
        *static_cast<const StandaloneBufferizePass *>(this));
  }

  void runOnOperation() final {
    auto *context = &getContext();
    BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    auto &ieAnalysis = getAnalysis<InsertExtractAnalysis>();
    if (disableIEAnalysis)
      ieAnalysis.disableAnalysis();

    target.addLegalDialect<memref::MemRefDialect>();
    target.addDynamicallyLegalDialect<arith::ArithDialect,
                                      func::FuncDialect>([&](Operation *op) {
      return typeConverter.isLegal(op) || isa<arith::ConstantOp>(op);
    });
    target.addDynamicallyLegalDialect<linalg::LinalgDialect>(
        [&](Operation *op) {
          return typeConverter.isLegal(op) ||
                 !ieAnalysis.isLinalgMarkedForBufferization(op);
        });
    target.addDynamicallyLegalOp<linalg::FillOp>([](linalg::FillOp op) {
      return op.hasPureBufferSemantics() || op.getNumResults() != 1 ||
             !op.getDpsInitOperand(0)->get().getType().cast<ShapedType>().hasStaticShape();
    });
    target.addLegalOp<func::CallOp>();
    target.addLegalOp<func::ReturnOp>();
    target.addLegalOp<linalg::CopyOp>();
    target.addLegalOp<linalg::YieldOp>();
    target.addLegalOp<tensor::EmptyOp>();
    target.addIllegalOp<tensor::InsertOp>();
    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<bufferization::BufferizationDialect>();

    patterns.add<BufferizeInsertOp>(typeConverter, patterns.getContext());
    patterns.add<BufferizeInsertExtractPair>(ieAnalysis, typeConverter,
                                             patterns.getContext());
    patterns.add<BufferizeExtractSliceOp>(ieAnalysis, typeConverter,
                                          patterns.getContext());
    patterns.add<BufferizeInsertSliceOp>(ieAnalysis, typeConverter,
                                         patterns.getContext());
    patterns.add<BufferizeFillOp>(typeConverter, patterns.getContext());
    // patterns.add<BufferizeLinalgOp>(typeConverter,
    // patterns.getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {

      signalPassFailure();
    }
  };

protected:
  Pass::Option<bool> disableIEAnalysis{
      *this, "disable-ie", llvm::cl::desc("Disables Insert-Extract Analysis"),
      llvm::cl::init(false)};
};
} // namespace

std::unique_ptr<Pass> mlir::lagrad::createBufferizePass() {
  return std::make_unique<StandaloneBufferizePass>();
}
