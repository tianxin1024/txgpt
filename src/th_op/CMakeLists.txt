
add_definitions(-DTORCH_CUDA=1)


add_library(th_utils STATIC th_utils.cu)
set_property(TARGET th_utils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET th_utils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(th_utils PUBLIC "${TORCH_LIBRARIES}" -lcublas -lcudart -lcurand tensor)

add_subdirectory(gpt)
