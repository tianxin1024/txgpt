add_library(th_gpt SHARED GptOp.cc)
target_link_libraries(th_gpt PRIVATE "${TORCH_LIBRARIES}" th_utils)
