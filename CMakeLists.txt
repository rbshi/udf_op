cmake_minimum_required(VERSION 3.0)

project(udf_op)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Vitis)

# Platform
set(DSA_NAME "xilinx_u280_xdma_201920_3" CACHE STRING "DSA string for v++/xocc.")
set(HOST_NAME "udf_selection" CACHE STRING "Host executable name.")
set(KERNEL_NAME krnl_${HOST_NAME} CACHE STRING "Kernel name default krnl_HOST_NAME.")

# From Step in v++
set(VPP_FROM_STEP "system_link" CACHE STRING "From step in v++ compilation.")

# Common src and head that comes with Xilinx VITIS example
set(OCL_COMMON_SRC ${CMAKE_CURRENT_SOURCE_DIR}/common/includes/xcl2/xcl2.cpp)
set(OCL_COMMON_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/common/includes/xcl2)


#FIXME: should VIVADO_INCLUDE be added in to FindVitis?
set(CXXFLAGS -I${Vitis_INCLUDE_DIRS} -I{OCL_COMMON_INCLUDE})
set(LDFLAGS -L${Vitis_LIBRARIES} -lOpenCL -pthread)
set(HOST_SRC ${OCL_COMMON_SRC} ${CMAKE_CURRENT_SOURCE_DIR}/src/${HOST_NAME}.cpp)
set(KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/${KERNEL_NAME}.cpp)

include_directories(${Vitis_INCLUDE_DIRS} ${OCL_COMMON_INCLUDE})
# Host
add_executable(${HOST_NAME} ${HOST_SRC})
target_link_libraries(${HOST_NAME} ${Vitis_LIBRARIES} pthread)


#FIXME: only support one kernel now
function(binary_container_build TARGET)
    set(TEMP_DIR ${CMAKE_BINARY_DIR}/_x.${TARGET}.${DSA_NAME})
    set(VPP_FLAGS -t ${TARGET} --platform ${DSA_NAME} --save-temps)
    if(NOT ${TARGET} EQUAL "hw")
        set(VPP_FLAGS ${VPP_FLAGS} -g)
    endif()

    add_custom_command(
        OUTPUT ${TEMP_DIR}/${KERNEL_NAME}.xo
        COMMENT "Compile kernel code (.xo)."
        COMMAND ${Vitis_COMPILER} ${VPP_FLAGS} -c -k ${KERNEL_NAME} --temp_dir ${TEMP_DIR} 
                --log_dir ${TEMP_DIR}/logs
                -I ${CMAKE_CURRENT_SOURCE_DIR}/src
                -o ${TEMP_DIR}/${KERNEL_NAME}.xo
                ${KERNEL_SRC}
        DEPENDS ${KERNEL_SRC})
    set(BUILD_DIR ${CMAKE_BINARY_DIR}/build_dir.${TARGET}.${DSA_NAME})

    # check if kernel config file (.cfg) exist
    if (EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_NAME}.cfg)
        set(VPP_LDFLAGS_KERNEL --config ${CMAKE_CURRENT_SOURCE_DIR}/${KERNEL_NAME}.cfg)
    endif()

    add_custom_command(
        OUTPUT ${BUILD_DIR}/${KERNEL_NAME}.xclbin
        COMMENT "Link kernel code (.xclbin)"
        COMMAND ${Vitis_COMPILER} ${VPP_FLAGS} -l ${VPP_LDFLAGS} --temp_dir ${TEMP_DIR} ${VPP_LDFLAGS_KERNEL} 
                --log_dir ${TEMP_DIR}/logs --from_step ${VPP_FROM_STEP}
                # -o ${BUILD_DIR}/${KERNEL_NAME}.link.xclbin ${TEMP_DIR}/${KERNEL_NAME}.xo
                -o${BUILD_DIR}/${KERNEL_NAME}.xclbin ${TEMP_DIR}/${KERNEL_NAME}.xo
        # Package feature starts from 2020.2
        # COMMAND ${Vitis_COMPILER} -p ${BUILD_DIR}/${KERNEL_NAME}.link.xclbin -t ${TARGET} --platform ${DSA_NAME} 
        #         --log_dir ${TEMP_DIR}/logs
        #         --package.out_dir ${CMAKE_BINARY_DIR}/package.${TARGET} -o ${BUILD_DIR}/${KERNEL_NAME}.xclbin
        DEPENDS ${TEMP_DIR}/${KERNEL_NAME}.xo)
endfunction()

set(BUILD_MODE "sw_emu" "hw_emu" "hw")
foreach(mode IN LISTS BUILD_MODE)
    message(STATUS "Add MODE ${mode}")
    binary_container_build(${mode})
endforeach()


add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/emconfig.json
    COMMENT "Generating emconfig.json file for hardware emulation."
    COMMAND ${VITIS_ROOT}/bin/emconfigutil --platform ${DSA_NAME})

add_custom_target(sw_emu
    COMMENT "Build for software simulation."
    DEPENDS ${CMAKE_BINARY_DIR}/emconfig.json
            ${CMAKE_BINARY_DIR}/build_dir.sw_emu.${DSA_NAME}/${KERNEL_NAME}.xclbin)
add_custom_target(hw_emu
    COMMENT "Build for hardware simulation."
    DEPENDS ${CMAKE_BINARY_DIR}/emconfig.json
            ${CMAKE_BINARY_DIR}/build_dir.hw_emu.${DSA_NAME}/${KERNEL_NAME}.xclbin)
add_custom_target(hw
    COMMENT "Build hardware."
    DEPENDS ${CMAKE_BINARY_DIR}/build_dir.hw.${DSA_NAME}/${KERNEL_NAME}.xclbin)

add_custom_target(clean_sw_emu
    COMMENT "Cleaning all sw_emu files."
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/emconfig.json
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/_x.sw_emu.${DSA_NAME}
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/build_dir.sw_emu.${DSA_NAME})

add_custom_target(clean_hw_emu
    COMMENT "Cleaning all hw_emu files."
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/emconfig.json
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/_x.hw_emu.${DSA_NAME}
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/build_dir.hw_emu.${DSA_NAME})

add_custom_target(clean_hw
    COMMENT "Cleaning all hw files."
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/_x.hw.${DSA_NAME}
    COMMAND rm -rf ${CMAKE_BINARY_DIR}/build_dir.hw.${DSA_NAME})















