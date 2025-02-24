set(LLVM_TARGET_DEFINITIONS Play.td)
mlir_tablegen(Play.hxx.inc --gen-dialect-decls)
mlir_tablegen(Play.cxx.inc --gen-dialect-defs)
add_public_tablegen_target(MLIRPlayIncGen)