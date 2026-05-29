/**
 * @file LLFastMath.cpp
 * @brief Add fast math flags to supported instructions
 *
 */

#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace {
struct LLFastMathPass : PassInfoMixin<LLFastMathPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &) {
    bool modified = false;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (isa<FPMathOperator>(*I)) {
        I->setFast(true);
        modified = true;
      }
    }
    return modified ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LLFastMath", LLVM_VERSION_STRING,
          [](PassBuilder &PB) {
            PB.registerOptimizerEarlyEPCallback(
                [](ModulePassManager &MPM, OptimizationLevel,
                   ThinOrFullLTOPhase) {
                  FunctionPassManager FPM;
                  FPM.addPass(LLFastMathPass());
                  MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
                });
          }};
}
