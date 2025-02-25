#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Play/PlayDialect.h.inc"

#define GET_OP_CLASSES
#include "Play/Play.h.inc"

#define GET_TYPEDEF_CLASSES
#include "Play/PlayTypes.h.inc"
