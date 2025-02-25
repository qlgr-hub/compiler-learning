include(AddMLIRPython)

# Specifies that all MLIR packages are co-located under the `mlir_standalone`
# top level package (the API has been embedded in a relocatable way).
# TODO: Add an upstream cmake param for this vs having a global here.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=mlir_play.")


################################################################################
# Sources
################################################################################

declare_mlir_python_sources(PlayPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT PlayPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir_play"
  TD_FILE dialects/Play.td
  SOURCES
    dialects/play_pybind11.py
    dialects/play_nanobind.py
  DIALECT_NAME play)


declare_mlir_python_extension(PlayPythonSources.Pybind11Extension
  MODULE_NAME _playDialectsPybind11
  ADD_TO_PARENT PlayPythonSources
  SOURCES
    PlayExtensionPybind11.cxx
  EMBED_CAPI_LINK_LIBS
    PlayCAPI
  PYTHON_BINDINGS_LIBRARY pybind11
)

declare_mlir_python_extension(PlayPythonSources.NanobindExtension
  MODULE_NAME _playDialectsNanobind
  ADD_TO_PARENT PlayPythonSources
  SOURCES
    PlayExtensionNanobind.cxx
  EMBED_CAPI_LINK_LIBS
    PlayCAPI
  PYTHON_BINDINGS_LIBRARY nanobind
)


################################################################################
# Common CAPI
################################################################################

add_mlir_python_common_capi_library(PlayPythonCAPI
  INSTALL_COMPONENT PlayPythonModules
  INSTALL_DESTINATION python_packages/play/mlir_play/_mlir_libs
  OUTPUT_DIRECTORY "${MLIR_BINARY_DIR}/python_packages/play/mlir_play/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    PlayPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources.Core
)

################################################################################
# Instantiation of all Python modules
################################################################################

add_mlir_python_modules(PlayPythonModules
  ROOT_PREFIX "${MLIR_BINARY_DIR}/python_packages/play/mlir_play"
  INSTALL_PREFIX "python_packages/play/mlir_play"
  DECLARED_SOURCES
    PlayPythonSources
    # TODO: Remove this in favor of showing fine grained registration once
    # available.
    MLIRPythonExtension.RegisterEverything
    MLIRPythonSources
  COMMON_CAPI_LINK_LIBS
    PlayPythonCAPI
  )
