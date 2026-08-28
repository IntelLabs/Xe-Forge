# orbit_mini SYCL extension (Xe-Orbit plan §11.3, §11.4, §13, §15.2)
#
# THIS BUILD IS OPTIONAL AND MUST STAY OPTIONAL.
# orbit_mini runs on a CPU-only machine with no oneAPI, no icpx and no XPU
# runtime. Nothing in the Python package requires the artifact this produces;
# `kernels/sycl_op.py` probes for the op and falls back to torch when it is
# absent. Do not add this to a default build.
#
# WHY IT IS HERE AT ALL
# ---------------------
# For a compiled kernel, dependency closure is not an AST walk — it is the build
# graph (§11.3). `compile_commands.json` gives the exact compile line for the
# translation unit: every include path, every define, every flag. That file is
# the extraction input for a SYCL kernel the way a `@triton.jit` function object
# is for a Triton one, so CMAKE_EXPORT_COMPILE_COMMANDS is on below and is not
# incidental.
#
# It also gives the compiler-option sweep (§11.4) something real to sweep:
# ORBIT_MINI_SYCL_FLAGS below is the knob a candidate varies.
#
# BUILD
#   cmake -S . -B build \
#         -DCMAKE_CXX_COMPILER=icpx \
#         -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
#   cmake --build build -j
#   export ORBIT_MINI_SYCL_LIB=$PWD/build/liborbit_mini_sycl.so
#
# Then `python -m examples.orbit_mini --device xpu` picks the op up through the
# dispatcher, and §13's P1 verification can assert that it appears in the trace.

cmake_minimum_required(VERSION 3.20)
project(orbit_mini_sycl LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# The extraction input for a compiled kernel (§11.3). Keep this ON.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Torch REQUIRED)

add_library(orbit_mini_sycl SHARED orbit_mini_rmsnorm.cpp)

target_link_libraries(orbit_mini_sycl PRIVATE ${TORCH_LIBRARIES})
target_include_directories(orbit_mini_sycl PRIVATE ${TORCH_INCLUDE_DIRS})

# ---------------------------------------------------------------------------
# SYCL detection. Absent icpx, the translation unit still compiles and still
# registers the op — it just refuses to execute. That keeps the schema and the
# dispatch-key registration testable on a machine with no oneAPI, while making
# an accidental "it silently computed something" impossible.
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" OR CMAKE_CXX_COMPILER MATCHES "icpx")
  message(STATUS "orbit_mini: SYCL compiler detected, building the real kernel")
  target_compile_definitions(orbit_mini_sycl PRIVATE ORBIT_MINI_HAS_SYCL)

  # The compiler-option sweep surface (§11.4). A candidate varies these; the
  # baseline must record whichever set produced it, or the comparison is between
  # two different builds and says nothing about the source change.
  set(ORBIT_MINI_SYCL_FLAGS
      -fsycl
      -fsycl-targets=spir64
      -O3
      CACHE STRING "SYCL compile/link flags swept by the Xe-Orbit compiler-option pass")

  target_compile_options(orbit_mini_sycl PRIVATE ${ORBIT_MINI_SYCL_FLAGS})
  target_link_options(orbit_mini_sycl PRIVATE ${ORBIT_MINI_SYCL_FLAGS})
else()
  message(WARNING
    "orbit_mini: no SYCL compiler (need icpx). Building a registration-only stub; "
    "the op will be registered but will refuse to run. This is expected on CPU-only CI.")
endif()

set_target_properties(orbit_mini_sycl PROPERTIES
  OUTPUT_NAME orbit_mini_sycl
  PREFIX "lib")
