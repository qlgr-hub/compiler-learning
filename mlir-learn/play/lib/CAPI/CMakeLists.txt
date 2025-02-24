add_mlir_public_c_api_library(PlayCAPI
  Dialects.cxx
  LINK_LIBS PUBLIC
  MLIRPlay
)