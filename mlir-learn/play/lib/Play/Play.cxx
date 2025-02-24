/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Definitions                                                        *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Play.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "Play/Play.hxx"

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::play::PlayDialect)
namespace mlir {
namespace play {

PlayDialect::PlayDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context, ::mlir::TypeID::get<PlayDialect>()) {
        initialize();
    }

PlayDialect::~PlayDialect() = default;
} // namespace play
} // namespace mlir
