//===- LAGradOps.td - LAGrad dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LAGRAD_OPS
#define LAGRAD_OPS

include "LAGradDialect.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def LAGrad_DiffOp : LAGrad_Op<"diff", [Pure]> {
    let summary = "Annotate a function for automatic differentiation via Enzyme";
    let description = [{
        The `lagrad.diff` operation annotates a function for automatic differentiation via Enzyme.
    }];

    let arguments = (ins FunctionType:$input);
    let results = (outs FunctionType:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input) `,` type($res)
    }];
}

def LAGrad_GradOp : LAGrad_Op<"grad", [
        Pure,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
    let summary = "Reverse-mode autodiff a standard function.";
    let description = [{
        The `lagrad.grad` operation performs a source transformation to compute the gradient of a func op.
        
        Optional attributes:
        - `of`: ArrayAttr of integers specifying which arguments to differentiate with respect to
        - `grad_signal`: UnitAttr indicating a custom gradient signal is provided
        - `sparse`: UnitAttr indicating the gradient signal is one-hot sparse
        - `return_primal`: UnitAttr indicating the primal result should also be returned (default: false)
        - `primal_of`: IntegerAttr specifying which return value index is the "real" output for
          backpropagation when the function has multiple return values (default: 0). Other return
          values are treated as pass-through data (e.g., updated running statistics) and do not
          receive gradient signals.
    }];

    let arguments = (ins FlatSymbolRefAttr:$F, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$results);

    let assemblyFormat = [{
        $F `(` $operands `)` attr-dict `:` functional-type($operands, $results)
    }];
}

def LAGrad_TangentOp : LAGrad_Op<"tangent", [
        Pure,
        DeclareOpInterfaceMethods<SymbolUserOpInterface>
    ]> {
    let summary = "Forward-mode autodiff a standard function.";
    let description = [{
        The `lagrad.tangent` operation performs a source transformation to compute the forward-mode (tangent) gradient of a function.
    }];

    let arguments = (ins FlatSymbolRefAttr:$F, Variadic<AnyType>:$operands);
    let results = (outs Variadic<AnyType>:$results);

    let assemblyFormat = [{
        $F `(` $operands `)` attr-dict `:` functional-type($operands, $results)
    }];
}

def LAGrad_PackOp : LAGrad_Op<"pack", [Pure]> {
    let summary = "convert fully materialized triangular tensor to packed";
    let description = [{
        The `lagrad.pack` operation denotes a conversion from a fully materialized strictly lower triangular tensor to a packed triangular tensor.
    }];

    let arguments = (ins AnyRankedTensor:$source);
    let results = (outs AnyRankedTensor:$dest);

    let assemblyFormat = [{
        $source attr-dict `:` type($source) `to` type($dest)
    }];

    let hasVerifier = 1;
}

#endif // LAGRAD_OPS
