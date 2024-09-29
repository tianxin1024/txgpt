#include "th_op/gpt/GptOp.h"

namespace th = torch;
namespace ft = fastertransformer;
namespace torch_ext {

} // namespace torch_ext

static auto fasterTransformerGptTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::GptOp>("FasterTransformerGptOp")
#else
    torch::jit::class_<torch_ext::GptOp>("FasterTransformer", "GptOp")
#endif
        .def(torch::jit::init<int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              int64_t,
                              bool,
                              double,
                              std::string,
                              std::string,
                              bool,
                              bool,
                              bool,
                              bool,
                              int64_t,
                              bool,
                              std::vector<th::Tensor>>())
        .def("forward", &torch_ext::GptOp::forward);
