#ifndef PLAY_DIALECT
#define PLAY_DIALECT

include "mlir/IR/OpBase.td"


def PlayDialect : Dialect {
    let name = "play";
    let cppNamespace = "::mlir::play";
}

#endif // PLAY_DIALECT