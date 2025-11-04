# Install script for directory: D:/Coding/CG/dandelion/deps/assimp/code

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "C:/Program Files (x86)/dandelion")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "C:/Program Files/JetBrains/CLion 2024.1.6/bin/mingw/bin/objdump.exe")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "libassimp6.0.2-dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "D:/Coding/CG/dandelion/cmake-build-debug/deps/assimp/lib/libassimpd.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "assimp-dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/assimp" TYPE FILE FILES
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/anim.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/aabb.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/ai_assert.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/camera.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/color4.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/color4.inl"
    "D:/Coding/CG/dandelion/cmake-build-debug/deps/assimp/code/../include/assimp/config.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/ColladaMetaData.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/commonMetaData.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/defs.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/cfileio.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/light.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/material.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/material.inl"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/matrix3x3.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/matrix3x3.inl"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/matrix4x4.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/matrix4x4.inl"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/mesh.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/ObjMaterial.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/pbrmaterial.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/GltfMaterial.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/postprocess.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/quaternion.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/quaternion.inl"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/scene.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/metadata.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/texture.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/types.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/vector2.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/vector2.inl"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/vector3.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/vector3.inl"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/version.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/cimport.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/AssertHandler.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/importerdesc.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Importer.hpp"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/DefaultLogger.hpp"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/ProgressHandler.hpp"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/IOStream.hpp"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/IOSystem.hpp"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Logger.hpp"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/LogStream.hpp"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/NullLogger.hpp"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/cexport.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Exporter.hpp"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/DefaultIOStream.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/DefaultIOSystem.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/ZipArchiveIOSystem.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/SceneCombiner.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/fast_atof.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/qnan.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/BaseImporter.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Hash.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/MemoryIOWrapper.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/ParsingUtils.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/StreamReader.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/StreamWriter.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/StringComparison.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/StringUtils.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/SGSpatialSort.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/GenericProperty.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/SpatialSort.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/SkeletonMeshBuilder.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/SmallVector.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/SmoothingGroups.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/SmoothingGroups.inl"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/StandardShapes.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/RemoveComments.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Subdivision.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Vertex.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/LineSplitter.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/TinyFormatter.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Profiler.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/LogAux.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Bitmap.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/XMLTools.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/IOStreamBuffer.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/CreateAnimMesh.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/XmlParser.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/BlobIOSystem.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/MathFunctions.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Exceptional.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/ByteSwapper.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Base64.hpp"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "assimp-dev" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/assimp/Compiler" TYPE FILE FILES
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Compiler/pushpack1.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Compiler/poppack1.h"
    "D:/Coding/CG/dandelion/deps/assimp/code/../include/assimp/Compiler/pstdint.h"
    )
endif()

