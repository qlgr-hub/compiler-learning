#ifndef PLAY_HXX
#define PLAY_HXX
/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: Play.td                                                              *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include <mlir/IR/Dialect.h>

namespace mlir {
  namespace play {
  
  class PlayDialect : public ::mlir::Dialect {
    explicit PlayDialect(::mlir::MLIRContext *context);
  
    void initialize();
    friend class ::mlir::MLIRContext;
  public:
    ~PlayDialect() override;
    static constexpr ::llvm::StringLiteral getDialectNamespace() {
      return ::llvm::StringLiteral("play");
    }
  };
  } // namespace play
} // namespace mlir
MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::play::PlayDialect);
  
#endif // PLAY_HXX