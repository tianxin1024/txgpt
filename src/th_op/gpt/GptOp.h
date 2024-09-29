#pragma once

#include "th_op/th_utils.h"
#include "utils/cuda_utils.h"

namespace ft = fastertransformer;
namespace th = torch;

namespace torch_ext {
using std::vector;

class IFGpt {
public:
    virtual ~IFGpt() {
    }
    virtual void forward(th::Tensor &input_ids,
                         th::Tensor &input_lengths,
                         th::Tensor &output_ids,
                         th::Tensor &sequence_lengths,
                         th::Tensor &cum_log_probs,
                         const size_t request_output_len,
                         const size_t beam_width,
                         th::optional<th::Tensor> top_k_opt,
                         th::optional<th::Tensor> top_p_opt,
                         th::optional<th::Tensor> beam_search_diversity_rate_opt,
                         th::optional<th::Tensor> temperature_opt,
                         th::optional<th::Tensor> len_penalty_opt,
                         th::optional<th::Tensor> repetition_penalty_opt,
                         th::optional<th::Tensor> random_seed_opt,
                         th::optional<int64_t> return_cum_log_probs_opt) = 0;
};

class GptOp : public th::jit::CustomClassHolder {
public:
    GptOp(const int64_t head_num,
          const int64_t size_per_head,
          const int64_t inter_size,
          const int64_t layer_num,
          const int64_t vocab_size,
          const int64_t start_id,
          const int64_t end_id,
          const bool sparse,
          const double layernorm_eps,
          const std::string layernorm_type,
          const std::string activation_type,
          const bool has_positional_encoding,
          const bool has_pre_decoder_layernorm,
          const bool has_post_decoder_layernorm,
          const bool has_adapters,
          const int64_t adapter_inter_size,
          const bool use_attention_linear_bias,
          const vector<th::Tensor> weights);

    ~GptOp();

    vector<th::Tensor> forward(th::Tensor input_ids,
                               th::Tensor input_lengths,
                               const int64_t output_len,
                               th::optional<int64_t> beam_width_opt,
                               th::optional<th::Tensor> top_k_opt,
                               th::optional<th::Tensor> top_p_opt,
                               th::optional<th::Tensor> beam_search_diversity_rate_opt,
                               th::optional<th::Tensor> temperature_opt,
                               th::optional<th::Tensor> len_penalty_opt,
                               th::optional<th::Tensor> repetition_penalty_opt,
                               th::optional<th::Tensor> random_seed_opt,
                               th::optional<int64_t> return_cum_log_probs_opt);

private:
    const at::ScalarType st_;
    IFGpt *ftgpt;
    std::vector<th::Tensor> weights;
};

} // namespace torch_ext
