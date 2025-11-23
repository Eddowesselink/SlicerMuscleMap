cmake_minimum_required(VERSION 3.13.4)

#-----------------------------------------------------------------------------
# Extension meta-data
#-----------------------------------------------------------------------------

project(MuscleMap)

# Naam zoals hij in de Extensions Manager zal verschijnen
set(EXTENSION_NAME "MuscleMap")

# Korte beschrijving
set(EXTENSION_DESCRIPTION
  "MuscleMap provides automated whole-body muscle segmentation directly inside 3D Slicer."
  )

# Categorie in de Extensions Manager
set(EXTENSION_CATEGORY "Segmentation")

# Wie er als contributors vermeld worden
set(EXTENSION_CONTRIBUTORS
  "Eddo Wesselink; Kenneth A. Weber; MuscleMap Consortium"
  )

# Homepage van de extensie (jouw GitHub repo)
set(EXTENSION_HOMEPAGE
  "https://github.com/Eddowesselink/SlicerMuscleMap"
  )

# Icoon in de Extensions Manager (PNG moet openbaar bereikbaar zijn)
# Zorg dat dit pad klopt – deze wijst naar MuscleMap.png in je repo root (main branch)
set(EXTENSION_ICONURL
  "https://raw.githubusercontent.com/Eddowesselink/Slicer3D_MuscleMap/main/MuscleMap.png"
  )

# Screenshots (optioneel, kun je later bijv. PNG's in de repo zetten en hier linken)
set(EXTENSION_SCREENSHOTURLS
  ""
  )

# Afhankelijkheden van andere Slicer-extensies (build/install afhankelijkheden)
# Als je SlicerPyTorch officieel wilt aangeven:
# set(EXTENSION_DEPENDS "SlicerPyTorch")
# Voor nu laten we het leeg zodat build niet faalt als die ontbreekt:
set(EXTENSION_DEPENDS
  ""
  )

#-----------------------------------------------------------------------------
# Slicer setup
#-----------------------------------------------------------------------------

find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Scripted module build
#-----------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------
# Tests
#-----------------------------------------------------------------------------

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
