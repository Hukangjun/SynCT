cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(SynCT)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/SynCT")
set(EXTENSION_CONTRIBUTORS "Kangjun Hu (Zhejiang University)")
set(EXTENSION_DESCRIPTION "This is an example of a SynCT extension")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/SynCT.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/SynCT/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SynCT)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
