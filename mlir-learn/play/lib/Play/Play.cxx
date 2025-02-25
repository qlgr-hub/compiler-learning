#include "Play/Play.hxx"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"


using namespace mlir;
using namespace mlir::play;

#include "Play/PlayDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Play/Play.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Play/PlayTypes.cpp.inc"

void PlayDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Play/Play.cpp.inc"
      >();
  registerTypes();
}

void PlayDialect::registerTypes() {
    addTypes<
  #define GET_TYPEDEF_LIST
  #include "Play/PlayTypes.cpp.inc"
        >();
  }
