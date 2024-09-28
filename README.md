# txgpt
Example gpt from the FasterTransformer project for learning

## download vocab_file

* Download vocab and merge table

They can be used in both OpenAI GPT-2 and Megatron.

```bash
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -P ../models
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -P ../models
```

## download ckpt_path

```bash
wget --content-disposition https://api.ngc.nvidia.com/v2/models/nvidia/megatron_lm_345m/versions/v0.0/zip -O megatron_lm_345m_v0.0.zip
mkdir -p ../models/megatron-models/345m
unzip megatron_lm_345m_v0.0.zip -d ../models/megatron-models/345m
```

