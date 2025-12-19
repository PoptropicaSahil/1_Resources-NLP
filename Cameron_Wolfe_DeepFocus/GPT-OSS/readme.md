# Notes from Cameron Wolfe's substack titled 'GPT-oss from the Ground Up'

<[https://cameronrwolfe.substack.com/p/gpt-oss](https://cameronrwolfe.substack.com/p/gpt-oss)>

## At a Glance

| Model | Layers | Total Params | Active Params per token | Total Experts | Active Experts per Token | Context Size | Size if MXFP4 (~4-bit) |
|-------|--------|--------------|-------------------------|---------------|--------------------------|--------------| -----------------------|
| gpt-oss-120b | 36 | 117B | **5.1B** | 128 | 4 | 128k | **80 GB** |
| gpt-oss-20b  | 24 | 21B | **3.6B** | 32| 4 | 128k | **16 GB** |

OpenAI released the Harmony prompt format (chat template) for handling function calling, tool use, reasoning etc. Apparently it is found to be overly complex!

> OpenAI also released HealthBench, for health-related tasks.

GPT-oss models obey inference-time scaling laws (wrt reasoning) - as longer reasoning traces generated, performance improves, *and therefore consume more compute* (during inference).

<img src="readme-images/inference_scaling.png" alt="drawing" width="700"/>

## Model Architecture

### Transformer Structure

Embeddings layer gives 2880-dim vectors.

<img src="readme-images/archi1.png" alt="drawing" width="700"/>

*pretty standard*

Even if pre-norm structures are common, there is no clear answer if pre-norm or post-norm is superior. Some work shows post-norm benefits training stability

<img src="readme-images/norm1.png" alt="drawing" width="700"/>

Although a pre-normalization structure is most common, there is no clear answer in terms of whether pre or post-normalization is superior. In fact, recent work has even shown that post-normalization benefits training stability

### Attention Mechanism

<img src="readme-images/attn1.png" alt="drawing" width="700"/>

**Masked self-attention:**  Each self-attention layer has 64 parallel attention heads, each with dimension 64. So the key, query, value projections transform embedding vectors from 2880-dim to 64-dims

<img src="readme-images/gqa1.png" alt="drawing" width="700"/>

**Grouped Query Attention (GQA):** Best tradeoff because at inference time, because fewer keys/values to be retrieved from KV cache. MQA has been shown to quality degradation and training instability

<img src="readme-images/attn2.png" alt="drawing" width="500"/>

**Sparse Attention (aka Locally Bandeded Sparse Attention):** One diagonal is masked as usual causal attention. Also masking tokens that are *sufficiently far in the past*. Sparse because **every other masked self-attention module is replaced with sliding window attention**. All to avoid $O(n^2)$ complexity.

> 1:1 ratio of dense and sparse attention layers is still conservative. E.g. Gemma-3 uses 5:1 ratio

<img src="readme-images/attn3.png" alt="drawing" width="500"/>

> Self attention is essentially a weighted sum of the value vectors. Where weights are given by the attention scores

**Attention Sinks:** Since softmax forces a probability distribution - *it is IMPOSSIBLE for the model to NOT pay attention to any tokens.* It has been found that LLMs assign spuriously high attention scores to (usually) the first token in a sequence - commonly referred to as **"attention sinks"**. Super high scores assigned by LLMs to attention sinks can lead to practical issues, e.g., outlier attention values make quantization more difficult.

> Qualcomm AI researchers found that 97%+ of outlier activations in LLMs occur in whitespace and punctuation positions

<img src="readme-images/attn-sink.png" alt="drawing" width="700"/>

***NOTE:*** Expand this image from Evan Miller's blog and read the highlights. It is lovely!

<img src="readme-images/evan-miller1.png" alt="drawing" width="100"/>

To solve this issue, authors use an approach simlar to (not exactly same though) the one proposed by Evan Miller. *"Each attention head has a learned bias in the denominator of the softmax, similar to off-by-one attenion and attention sinks, which enables the attention mechanism to pay no attention to any tokens"*


