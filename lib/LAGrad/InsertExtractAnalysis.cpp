#include "LAGrad/Analysis.h"
#include "LAGrad/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Dominance.h"

namespace mlir {
InsertExtractAnalysis::InsertExtractAnalysis(Operation *op) {
  DominanceInfo dom;
  op->walk([&](tensor::ExtractSliceOp op) {
    auto maybeInsertSliceOp = getMatchingInsertSlice(op, dom);
    if (maybeInsertSliceOp.has_value()) {
      extract_to_insert[op] = maybeInsertSliceOp.value();
      matchingInserts.insert(maybeInsertSliceOp.value());
      for (auto &use : op.getResult().getUses()) {
        if (auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner())) {
          // Only handle the case where the linalg op has one output for now.
          if (linalgOp.isInitTensor(&use) && linalgOp.getNumDpsInits() == 1 &&
              linalgOp.hasPureTensorSemantics()) {
            linalgInPlaceOps.insert(linalgOp);
          }
        }
      }
    }
  });
}

void InsertExtractAnalysis::disableAnalysis() { disabled = true; }
bool InsertExtractAnalysis::isPairedExtractSlice(
    tensor::ExtractSliceOp op) const {
  if (disabled)
    return false;
  return extract_to_insert.count(op) > 0;
}

bool InsertExtractAnalysis::isPairedInsertSlice(
    tensor::InsertSliceOp op) const {
  if (disabled)
    return false;
  return matchingInserts.contains(op);
}

tensor::InsertSliceOp
InsertExtractAnalysis::getPairedInsertSlice(tensor::ExtractSliceOp op) const {
  return cast<tensor::InsertSliceOp>(extract_to_insert.lookup(op));
}

bool InsertExtractAnalysis::isLinalgMarkedForBufferization(
    Operation *op) const {
  if (disabled)
    return false;
  return linalgInPlaceOps.count(op) > 0;
}

std::optional<tensor::InsertSliceOp>
InsertExtractAnalysis::getMatchingInsertSlice(tensor::ExtractSliceOp op,
                                              const DominanceInfo &dom) const {
  if (disabled)
    return std::nullopt;
  // The source of the extract slice should have exactly one use besides the
  // insert slice op.
  size_t domUseCount = 0;
  tensor::InsertSliceOp insertSliceOp;
  for (Operation *user : op.getSource().getUsers()) {
    if (dom.properlyDominates(op.getResult(), user)) {
      if (auto iso = dyn_cast<tensor::InsertSliceOp>(user)) {
        if (iso.getDest() == op.getSource()) {
          insertSliceOp = iso;
        }
      }
      domUseCount++;
    }
  }
  if (domUseCount != 1) {
    return std::nullopt;
  }

  // We ultimately remove the insertSliceOp, so we must ensure that the
  // subsequent ops write in-place.
  bool isWrittenInPlace = false;
  for (OpOperand &use : op.getResult().getUses()) {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(use.getOwner())) {
      if (linalgOp.isInitTensor(&use)) {
        isWrittenInPlace = true;
      }
    } else if (auto forOp = dyn_cast<scf::ForOp>(use.getOwner())) {
      // This is potentially a dangerous assumption.
      isWrittenInPlace = true;
    } else if (auto addfOp = dyn_cast<arith::AddFOp>(use.getOwner())) {
      isWrittenInPlace = true;
    }
  }

  bool linked = false;
  if (insertSliceOp) {
    SmallVector<Value> frontier{op.getResult()};
    ValueSet derivedFromResult;
    runTopDownDFS(frontier, derivedFromResult);
    linked = derivedFromResult.contains(insertSliceOp.getSource());
  }

  if (linked && isWrittenInPlace && insertSliceOp &&
      (insertSliceOp.getMixedOffsets() == op.getMixedOffsets()) &&
      (insertSliceOp.getMixedSizes() == op.getMixedSizes()) &&
      (insertSliceOp.getMixedStrides() == op.getMixedStrides())) {
    return insertSliceOp;
  }
  return std::nullopt;
}
} // namespace mlir
