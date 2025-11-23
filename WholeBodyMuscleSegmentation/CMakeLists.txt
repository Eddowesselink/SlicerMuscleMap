cmake_minimum_required(VERSION 3.13.4)

#-----------------------------------------------------------------------------
# Extension meta-data
#-----------------------------------------------------------------------------

project(MuscleMap)

set(EXTENSION_NAME "MuscleMap")

set(EXTENSION_DESCRIPTION
  "MuscleMap provides automated whole-body muscle segmentation directly inside 3D Slicer."
  )

set(EXTENSION_CATEGORY "Segmentation")

set(EXTENSION_CONTRIBUTORS
  "Eddo Wesselink; Kenneth A. Weber; MuscleMap Consortium"
  )

set(EXTENSION_HOMEPAGE
  "https://github.com/Eddowesselink/SlicerMuscleMap"
  )

set(EXTENSION_ICONURL
  "https://raw.githubusercontent.com/Eddowesselink/Slicer3D_MuscleMap/main/MuscleMap.png"
  )

set(EXTENSION_SCREENSHOTURLS
  ""
  )

set(EXTENSION_DEPENDS
  "Pytorch"
  )

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

set(MODULE_NAME WholeBodyMuscleSegmentation)

set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  )

slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

if(BUILD_TESTING)

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()

#-----------------------------------------------------------------------------
# Packaging
#-----------------------------------------------------------------------------

include(${Slicer_EXTENSION_CPACK})
