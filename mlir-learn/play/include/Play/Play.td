#ifndef PLAY_DIALECT
#define PLAY_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Play_Dialect : Dialect {
    let name = "play";
    let summary = "A play out-of-tree MLIR dialect.";
    let description = [{
        This dialect is an example of an out-of-tree MLIR dialect designed to
        illustrate the basic setup required to develop MLIR-based tools without
        working inside of the LLVM source tree.
    }];
    let cppNamespace = "::mlir::play";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}


class Play_Op<string mnemonic, list<Trait> traits = []> :
        Op<Play_Dialect, mnemonic, traits>;


def Play_FooOp : Play_Op<"foo", [Pure, SameOperandsAndResultType]> {
    let summary = "Illustrates how to define an operation.";
    let description = [{
        The `play.foo` operation illustrates how to define a new
        operation in a dialect. It uses an operation trait to declare that it
        has no side effects.

        This operation takes an integer argument and returns an integer.

        Example:

        ```mlir
        %0 = arith.constant 2 : i32
        // Apply the foo operation to %0
        %1 = play.foo %0 : i32
        ```
    }];

    let arguments = (ins I32:$input);
    let results = (outs I32:$res);

    let assemblyFormat = [{
        $input attr-dict `:` type($input)
    }];
}

class Play_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Play_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Play_CustomType : Play_Type<"Custom", "custom"> {
    let summary = "Play custom type";
    let description = "Custom type in play dialect";
    let parameters = (ins StringRefParameter<"the custom value">:$value);
    let assemblyFormat = "`<` $value `>`";
}

#endif // !PLAY_DIALECT