cmake_minimum_required(VERSION 3.20.6...3.22.6 FATAL_ERROR)

project(MuscleMap)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/MuscleMap")
set(EXTENSION_CONTRIBUTORS "Eddo Wesselink (Stanford University), Kenneth Arnold Weber II (Stanford University)")
set(EXTENSION_DESCRIPTION "A 3D Slicer interface to the MuscleMap whole-body segmentation model.")
set(EXTENSION_ICONURL "https://github.com/Eddowesselink/SlicerMuscleMap/blob/main/MuscleMap.png")
set(EXTENSION_SCREENSHOTURLS "https://github.com/Eddowesselink/SlicerMuscleMap/blob/main/MuscleMap_Slicer3D.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(WholeBodyMuscleSegmentation)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
