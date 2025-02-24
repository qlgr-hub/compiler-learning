#include "Play-c/Dialects.hxx"

#include "Play/Play.hxx"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Play, play, mlir::play::PlayDialect)