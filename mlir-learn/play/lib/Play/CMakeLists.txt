add_mlir_dialect_library(MLIRPlay
        Play.cxx

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/Play

        DEPENDS
        MLIRPlayIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
