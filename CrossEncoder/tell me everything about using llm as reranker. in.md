<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# tell me everything about using llm as reranker. in super technical detail. Start with basic rerankers (non lllms) and how they work. then move into llms and how they can be leveraged as given in literature.

I have now conducted comprehensive research gathering over 80+ sources on reranking techniques, covering both traditional methods and LLM-based approaches. Let me now create a detailed technical report.

# Using LLMs as Rerankers: A Comprehensive Technical Deep Dive

Before diving into how Large Language Models function as rerankers, it's essential to understand the foundation: traditional reranking methods that don't rely on LLMs. Modern retrieval-augmented generation (RAG) pipelines and information retrieval systems have evolved through decades of research, starting with statistical methods and progressing to sophisticated neural architectures.[^1][^2][^3]

## Traditional Reranking: Non-LLM Approaches

### Statistical and Lexical Rerankers

The earliest reranking systems relied on statistical methods that analyze term frequencies and document characteristics. **BM25 (Best Matching 25)**, introduced as an enhancement to TF-IDF, remains one of the most influential algorithms in information retrieval. BM25 calculates relevance scores using term frequency (TF), inverse document frequency (IDF), and document length normalization. The scoring function is:[^4][^1]

\$ score(D,Q) = \sum_{i=1}^{n} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})} \$

where $f(q_i, D)$ is the frequency of query term $q_i$ in document $D$, $|D|$ is document length, $\text{avgdl}$ is average document length, and $k_1$ and $b$ are tunable parameters (typically $k_1 = 1.2$ to $2.0$ and $b = 0.75$)[^4][^2].

While BM25 excels at exact keyword matching and handles rare terms effectively, it struggles with semantic understanding. It cannot recognize synonyms, paraphrases, or conceptual similarity—"automobile" and "car" are treated as completely different terms. This limitation motivated the development of neural reranking approaches.[^1][^4]

### Neural Rerankers: The Cross-Encoder Revolution

Neural rerankers represent a paradigm shift from lexical to semantic matching. The most influential architecture is the **cross-encoder**, which processes query-document pairs jointly through a transformer network.[^5][^6][^7]

#### Cross-Encoder Architecture

A cross-encoder, typically built on BERT or similar transformers, concatenates the query and document into a single input sequence: `[CLS] query [SEP] document [SEP]`. This concatenated sequence flows through the transformer's self-attention layers, allowing every token in the query to interact with every token in the document before producing a relevance score.[^2][^8][^5]

The self-attention mechanism is central to the cross-encoder's power. For each token, the model computes three vectors: **query** ($\mathbf{Q}$), **key** ($\mathbf{K}$), and **value** ($\mathbf{V}$) through learned weight matrices $\mathbf{W}_Q$, $\mathbf{W}_K$, and $\mathbf{W}_V$. The attention score between tokens is calculated as:[^9][^10][^11]

\$ Attention(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = softmax\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V} \$

where $d_k$ is the dimensionality of the key vectors, and the scaling factor $\sqrt{d_k}$ prevents extremely large dot products that would push softmax into regions with vanishingly small gradients.[^11][^12][^9]

The final relevance score is extracted from the `[CLS]` token's representation after passing through multiple transformer layers. Typically, a linear projection layer maps this representation to a scalar score: $s = \mathbf{v}_p^T \cdot \text{cls}(\text{BERT}(\text{concat}(q, d)))$, where $\mathbf{v}_p$ is a learned projection vector.[^7][^8]

#### Training Cross-Encoders

Cross-encoders are typically trained with **binary cross-entropy (BCE) loss** or **pairwise ranking losses**. For BCE, the model learns to predict whether a document is relevant (1) or not (0):[^8][^7]

\$ \mathcal{L}_{BCE} = -\left[y \log(\sigma(s)) + (1-y)\log(1-\sigma(s))\right] \$

where $\sigma$ is the sigmoid function and $y$ is the binary relevance label.[^13][^8]

More sophisticated approaches use **contrastive losses** like the one in monoBERT, which samples positive and negative documents for each query. A recent advancement is **Localized Contrastive Estimation (LCE)**, which groups documents from the same initial ranking to ensure negatives are genuinely hard examples rather than easy-to-distinguish irrelevant documents.[^7][^8]

Modern training often incorporates **lambda loss**, a metric-driven approach that weights pairwise comparisons by their impact on ranking metrics like NDCG (Normalized Discounted Cumulative Gain). The gradient for each document pair is:[^14][^15][^16]

\$ \lambda_{ij} = \frac{\partial \mathcal{L}}{\partial s_i} = -\frac{\sigma'(s_i - s_j)}{1 + \sigma(s_i - s_j)} \cdot |\Delta NDCG_{ij}| \$

where $|\Delta \text{NDCG}_{ij}|$ is the change in NDCG if documents $i$ and $j$ were swapped[^14][^15]. This ensures the model focuses optimization effort where it maximally improves the target ranking metric.

#### Cross-Encoder Performance and Limitations

Cross-encoders achieve state-of-the-art effectiveness on benchmarks like MS MARCO and TREC Deep Learning, with models like **monoBERT** and **monoT5** significantly outperforming BM25. However, they come with severe computational costs: each query-document pair requires a full forward pass through the entire transformer, making them impractical for first-stage retrieval over millions of documents.[^17][^18][^19][^5][^8][^7]

For example, ranking 1000 documents for a single query with a BERT-based cross-encoder (340M parameters) requires 1000 forward passes, each processing up to 512 tokens. This computational burden restricts cross-encoders to **second-stage reranking**, where they refine an initial candidate set of 100-1000 documents retrieved by faster methods like BM25 or dense retrievers.[^18][^5][^17][^8][^1]

### Late Interaction Models: ColBERT

**ColBERT (Contextualized Late Interaction over BERT)** introduces an elegant middle ground between cross-encoders and bi-encoders. Unlike cross-encoders that jointly encode query and document, ColBERT independently encodes them into **multi-vector representations**—one embedding per token—and delays interaction until a final scoring step.[^20][^21][^22][^23]

#### ColBERT Architecture

ColBERT uses a shared BERT encoder with special tokens: `[Q]` for queries and `[D]` for documents. Query tokens are padded to a fixed length $N_q$ with `[MASK]` tokens, while documents are processed without padding. Each token is encoded into a contextualized embedding and L2-normalized.[^21][^23][^24]

The **late interaction mechanism** computes relevance via the **MaxSim** operation:[^25][^23][^21]

\$ S(q, d) = \sum_{i \in |E_q|} \max_{j \in |E_d|} (E_{q_i} \cdot E_{d_j}^T) \$

For each query token embedding $E_{q_i}$, the model finds the most similar document token embedding using dot product, then sums these maximum similarities across all query tokens. This captures fine-grained semantic matching—each query term finds its best match in the document, regardless of position.[^23][^24][^21][^25]

#### ColBERT's Efficiency-Effectiveness Trade-off

ColBERT achieves **near-cross-encoder effectiveness** with **significantly better efficiency**. Documents can be pre-encoded offline and indexed, with only the query requiring online encoding. The late interaction step is a simple matrix multiplication that's orders of magnitude faster than cross-encoder inference.[^24][^20][^21]

However, ColBERT's storage requirements are substantial: storing one embedding per token means the index size scales linearly with the number of tokens in the corpus. For the MS MARCO passage collection (8.8M passages), this can require hundreds of gigabytes. Compression techniques like **residual quantization** in ColBERTv2 and **token pruning** reduce this footprint by 50-70% with minimal accuracy loss.[^26][^27][^28][^20]

Recent work shows ColBERT's late interaction internally implements a **semantic variant of BM25**. Mechanistic interpretability studies reveal specialized attention heads that compute soft term frequency, weight by IDF-like signals, and perform length normalization—mirroring BM25's components but operating on contextualized embeddings rather than exact lexical matches.[^2]

## Large Language Models as Rerankers

The emergence of instruction-tuned Large Language Models like GPT-3.5, GPT-4, FLAN-T5, and LLaMA has opened new possibilities for zero-shot and few-shot reranking. LLM-based rerankers leverage the models' strong language understanding, reasoning capabilities, and ability to follow natural language instructions to assess document relevance without task-specific fine-tuning.[^29][^30][^31][^32][^33]

### Prompting Paradigms for LLM Reranking

LLM-based reranking can be categorized into three main prompting strategies: **pointwise**, **pairwise**, and **listwise** approaches.[^31][^34][^35][^36][^29]

#### Pointwise Reranking

**Pointwise methods** prompt the LLM to independently score each document's relevance to the query. Two common approaches are:[^34][^29][^31]

1. **Relevance Generation**: The LLM is asked a yes/no question like "Does the passage answer the query?" and the relevance score is derived from the probability of generating "Yes" or "No":[^29][^31]

\$ s_i = $$
\begin{cases} 1 + p(\text{Yes}), & \text{if output is Yes} \\ 1 - p(\text{No}), & \text{if output is No} \end{cases}
$$ \$
2. **Query Generation**: The LLM generates a hypothetical query given a document, and relevance is measured by the probability of generating the actual query.[^31][^29]

**Limitations**: Pointwise approaches suffer from critical calibration issues. LLMs are not trained to produce calibrated probabilities across different prompts, making scores incomparable between documents. Furthermore, generation-only APIs like GPT-4 don't expose logits, preventing score extraction entirely. Empirically, pointwise methods significantly underperform fine-tuned cross-encoders.[^35][^29][^31]

#### Pairwise Ranking Prompting (PRP)

**Pairwise Ranking Prompting**, introduced by Qin et al., simplifies the ranking task by asking the LLM to compare two documents at a time. The prompt template is:[^29][^31]

```
Given a query {query}, which of the following two passages is more relevant to the query?
Passage A: {passage_a}
Passage B: {passage_b}
Output Passage A or Passage B:
```

The LLM produces either "Passage A" or "Passage B" (generation mode) or assigns log-likelihood scores to these outputs (scoring mode). To mitigate position bias, each pair is queried twice with swapped order: $u(q, d_1, d_2)$ and $u(q, d_2, d_1)$. If both queries agree, a local ordering $d_1 > d_2$ is established; otherwise, the documents are considered equal.[^31][^29]

**PRP Variants**: Three strategies aggregate pairwise comparisons into global rankings:[^29][^31]

1. **PRP-Allpair**: Compares all $\binom{N}{2}$ document pairs and assigns each document a score based on win counts:
\$ s_i = 1 \cdot \sum_{j \neq i} \mathbb{I}_{d_i > d_j} + 0.5 \cdot \sum_{j \neq i} \mathbb{I}_{d_i = d_j} \$
Complexity: $O(N^2)$ LLM calls, but highly robust to input order.[^31][^29]
2. **PRP-Sorting**: Uses pairwise comparisons as the comparator in Heapsort, achieving $O(N \log N)$ complexity while maintaining order-independence.[^29][^31]
3. **PRP-Sliding-K**: A sliding window approach similar to Bubble Sort. One pass compares adjacent pairs and swaps disagreements with the initial ranking. Running $K$ passes ensures accurate top-K ranking with $O(KN)$ complexity.[^31][^29]

**Empirical Results**: On TREC-DL 2019/2020, PRP with FLAN-UL2 (20B parameters) achieves competitive or superior performance to GPT-4-based listwise approaches, outperforming InstructGPT (175B) by over 10% on all metrics. PRP-Sliding-10 maintains effectiveness while being dramatically more efficient than allpair comparisons.[^29][^31]

#### Listwise Reranking

**Listwise approaches** directly prompt the LLM to generate a complete ranked permutation of candidate documents. The documents are assigned identifiers (e.g., `[^1]`, `[^2]`, ..., `[N]`), and the LLM outputs an ordered sequence like `[^2] > [^5] > [^1] > ...`.[^30][^32][^36][^37][^38]

**RankGPT**, developed by Sun et al., is the most prominent listwise reranker. It uses a **sliding window strategy** to handle token length constraints: a window of size $W$ (typically 20) slides through the candidate list, with the LLM reranking documents within each window. The process continues until the ranking stabilizes.[^32][^36][^38][^30]

**Permutation Self-Consistency** addresses RankGPT's sensitivity to input order. The method shuffles the candidate list multiple times, generates rankings from each permutation, and aggregates them using techniques like Borda count or positional scoring to produce a final ranking robust to position bias.[^39][^38]

**Challenges**: Listwise methods face several failure modes:[^36][^30][^29]

- **Missing Documents**: The LLM outputs only a partial ranking.
- **Repetition**: Document identifiers appear multiple times.
- **Rejection**: The LLM refuses to rank or produces irrelevant outputs.
- **Inconsistency**: Different input orders yield wildly different rankings.

For moderate-sized LLMs (< 50B parameters), failure rates can exceed 12%, forcing fallback to the initial ranking and causing severe order sensitivity. Only the largest models like GPT-4 handle listwise prompting reliably.[^30][^36][^29]

### Attention-Based Reranking: In-Context Re-ranking (ICR)

A fundamentally different approach, **In-Context Re-ranking (ICR)**, bypasses generation entirely and directly uses LLM attention weights for scoring. Proposed by Chen et al., ICR aggregates attention from all layers and heads to compute document relevance without requiring text generation.[^40][^41][^42][^43]

#### ICR Mechanism

ICR constructs a prompt containing all candidate documents followed by the query. For each document $d_i$, the relevance score is the sum of attention weights from query tokens to document tokens, averaged across all heads and layers:[^43][^40]

\$ s_{d_{i,j};q} = \frac{1}{|\mathcal{T}_q|} \sum_{l=1}^{L} \sum_{h=1}^{H} \sum_{t \in \mathcal{T}_q} a_{j,t}^{l,h} \$

where $a_{j,t}^{l,h}$ is the attention weight from query token $t$ to document token $j$ in layer $l$, head $h$. The document score is the sum of token-level scores: $s_{d_i;q} = \sum_j s_{d_{i,j};q}$.[^40][^43]

**Complexity**: ICR requires only $O(1)$ forward passes (typically 2: one for scoring, one for calibration with a content-free query), dramatically reducing latency compared to generative methods that need $O(N)$ or more passes. Experiments show ICR outperforms RankGPT while cutting latency by over 60%.[^41][^42][^40]

**Contrastive Retrieval (CoRe) Heads**: Recent work identifies that not all attention heads contribute equally to reranking. CoRe heads are detected using a **contrastive scoring metric**:[^43]

\$ S_{CoRe}^{l,h} = \frac{1}{|Q|} \sum_{q \in Q} \left( \frac{1}{|D_q^+|} \sum_{d \in D_q^+} a_q^{l,h}(d) - \frac{1}{|D_q^-|} \sum_{d \in D_q^-} a_q^{l,h}(d) \right) \$

This rewards heads with high attention to relevant documents and low attention to irrelevant ones, capturing relative ranking ability. Aggregating attention from just the top 1% CoRe heads (8 heads in a 32-layer model) achieves state-of-the-art performance while enabling 40% memory reduction and 20% latency improvement through layer pruning.[^43]

### Fine-Tuning LLMs for Reranking

While zero-shot LLM reranking is appealing, **fine-tuning LLMs on ranking tasks** yields significant improvements. Two main approaches exist: **full fine-tuning** and **knowledge distillation**.[^44][^45][^46]

#### Direct Fine-Tuning with Ranking Losses

**RankT5** and similar models fine-tune T5 with listwise ranking losses. The model is trained to predict relevance scores for all documents in a list jointly, using losses like **softmax cross-entropy** over the list or **listwise lambda loss**. The training objective aligns the model's score distribution with the ground truth ranking:[^45][^47][^44]

\$ \mathcal{L}_{listwise} = -\sum_{i=1}^{N} y_i \log \frac{\exp(s_i)}{\sum_{j=1}^{N} \exp(s_j)} \$

where $y_i$ indicates document $i$'s relevance.[^47][^44]

**RankLLaMA** extends this to decoder-only architectures like LLaMA, using pairwise and listwise prompts during fine-tuning. By training on diverse passage reranking datasets (MS MARCO, BEIR), these models learn strong cross-domain ranking capabilities that generalize to unsupervised settings.[^48][^44]

#### Knowledge Distillation from Teacher Rerankers

**Knowledge distillation** transfers ranking knowledge from powerful teacher models (e.g., cross-encoders, GPT-4) to smaller student models. The student learns not just from ground-truth labels but from the teacher's nuanced relevance assessments.[^49][^50][^51][^52][^53][^45]

**Distillation Loss**: The student minimizes a combination of two losses:[^50][^52]

1. **Hard Loss**: Standard cross-entropy with ground-truth labels
\$ \mathcal{L}_{hard} = -\sum_{i} y_i \log p_{student}(i) \$
2. **Soft Loss**: KL divergence between student and teacher score distributions
\$ \mathcal{L}_{soft} = KL(p_{teacher} \| p_{student}) = \sum_{i} p_{teacher}(i) \log \frac{p_{teacher}(i)}{p_{student}(i)} \$

The total loss is: $\mathcal{L} = \alpha \mathcal{L}_{\text{hard}} + \beta \mathcal{L}_{\text{soft}}$, with $\alpha$ and $\beta$ balancing the two objectives (commonly $\alpha = \beta = 0.5$).[^52][^54]

**Temperature Scaling** softens probability distributions to expose richer knowledge:[^54][^52]

\$ p_i^T = \frac{\exp(s_i / T)}{\sum_j \exp(s_j / T)} \$

where $T > 1$ (typically 2-5). Higher temperatures spread probability mass more evenly, revealing subtle similarities the teacher perceives between documents.[^52][^54]

**Contrastive Partial Ranking Distillation (CPRD)** specifically targets cross-encoder to dual-encoder distillation. Rather than matching all scores, CPRD focuses on preserving the relative order among **hard negatives**—documents the teacher ranks highly but incorrectly. This is formulated as a contrastive loss that maximizes similarity to valid hard negatives and minimizes it to lower-ranked negatives:[^49][^50]

\$ \mathcal{L}_{CPRD} = -\sum_{i} \sum_{j < J_i^*} \log \frac{\exp(v_i^T t_{c_{ij}})}{\sum_{k=j}^{K} \exp(v_i^T t_{c_{ik}})} \$

where $J_i^*$ is the index of the first invalid hard negative. This coordinates with the contrastive learning objective used to train dual-encoders, improving knowledge transfer effectiveness.[^50][^49]

### Reasoning-Enhanced Reranking

**ReasoningRank** introduces dual reasoning into LLM reranking: **explicit reasoning** explaining how a document addresses the query, and **comparison reasoning** justifying why one document is more relevant than another. A teacher LLM (e.g., GPT-4) generates these rationales, which are then used to train smaller student models via distillation.[^51][^53]

The student learns from three signals simultaneously:[^51]

1. Final relevance scores
2. Explicit reasoning chains
3. Comparison reasoning between document pairs

This **rationale distillation** approach enables student models to achieve strong performance with less training data, as the reasoning provides richer supervision than scores alone. Experiments show students trained with reasoning outperform those trained on scores only, particularly on reasoning-intensive retrieval tasks.[^53][^54][^51]

## Practical Considerations and System Design

### Hybrid Retrieval Pipelines

State-of-the-art retrieval systems employ **multi-stage pipelines** combining complementary retrieval methods:[^3][^55][^56][^1]

1. **First Stage - Initial Retrieval**: Fast methods like BM25 or dense retrievers (e.g., DPR, Contriever) retrieve 1000-10000 candidates from the full corpus.[^56][^3][^1]
2. **Second Stage - Neural Reranking**: Cross-encoders or ColBERT rerank the top 100-1000 candidates, improving precision.[^55][^56][^1]
3. **Third Stage - LLM Reranking** (optional): For critical applications, LLM-based rerankers perform a final refinement on the top 10-50 documents.[^33][^57][^56]

**Why Hybrid?** Each stage exploits different strengths:[^58][^55][^2]

- **BM25**: Exact keyword matching, handles rare terms, extremely fast
- **Dense Retrieval**: Semantic similarity, handles synonyms and paraphrases
- **Cross-Encoders**: Fine-grained token-level interactions, contextualized relevance
- **LLM Rerankers**: Complex reasoning, zero-shot generalization, instruction-following

The cross-encoder's semantic BM25 behavior complements lexical BM25, providing coverage across both exact match and semantic dimensions.[^2]

### Cost-Performance Trade-offs

LLM reranking introduces significant computational costs:[^57][^35][^33]

**Pointwise LLM Reranking**: For 100 candidates, requires 100 LLM forward passes. With GPT-4 at ~\$0.03/1k tokens and ~500 tokens per prompt, this costs ~\$1.50 per query.[^35][^33]

**Pairwise Reranking**: PRP-Allpair requires $\binom{100}{2} = 4,950$ comparisons. Even with parallel batching, this is impractical for high query volumes.[^35][^29]

**Listwise Reranking**: RankGPT with sliding windows processes ~5-10 windows, requiring 5-10 LLM calls per query, but using generation APIs that charge per output token makes this expensive for long rankings.[^57][^30][^35]

**Optimization Strategies**:[^33][^35]

- Use **efficient variants**: PRP-Sliding-K or ICR reduce calls to $O(KN)$ or $O(1)$
- **Parallel batching**: Process multiple queries simultaneously
- **Model selection**: Open-source models like FLAN-UL2 avoid per-token API costs
- **Selective reranking**: Only invoke LLM reranking for ambiguous queries or high-value searches

**Latency Considerations**: LLM reranking adds 0.5-2 seconds of latency. For interactive search, this is often acceptable as a final refinement step. For real-time applications, caching frequently queried results or using smaller, distilled models reduces latency.[^53][^33][^57][^35]

### Quality-Efficiency Frontiers

Different reranking approaches occupy distinct points on the quality-efficiency frontier:


| Method | Relative Effectiveness | Relative Speed | Index Size | When to Use |
| :-- | :-- | :-- | :-- | :-- |
| BM25 | 1.0x | 1000x | Minimal | High-volume, keyword-focused queries |
| Dense Retrieval | 1.2-1.5x | 100x | Moderate | Semantic search, cross-lingual |
| ColBERT | 1.4-1.7x | 50x | Large | Multi-vector semantic matching |
| Cross-Encoder | 1.6-2.0x | 1x | None (no index) | Second-stage reranking |
| LLM (Zero-shot) | 1.5-1.8x | 0.1x | None | Zero-shot domains, reasoning tasks |
| LLM (Fine-tuned) | 1.8-2.2x | 0.1x | None | Task-specific, high-value queries |

*Effectiveness is relative to BM25 baseline; speed is approximate queries per second per core*.[^55][^1][^40][^29]

### Failure Modes and Mitigation

**Positional Bias**: LLMs exhibit strong recency bias, favoring documents appearing later in prompts. Mitigation strategies include:[^59][^36][^39]

- **Permutation self-consistency**: Average rankings over multiple shuffled inputs[^39]
- **Calibration queries**: Subtract attention from content-free queries[^40]
- **Position-aware prompts**: Explicitly instruct models to ignore position[^59][^36]

**Hallucination in Reasoning**: When generating explanations, LLMs may fabricate details not present in documents. Use **structured reasoning templates** and **citation requirements** to ground explanations in document content.[^51]

**Domain Shift**: Zero-shot LLM rerankers can underperform on highly specialized domains (medical, legal). Fine-tuning on domain-specific data or using domain-adapted models (e.g., BioBERT for biomedical) improves robustness.[^60][^30]

## Conclusion

Reranking technology has evolved from statistical methods like BM25 through neural cross-encoders and late interaction models to sophisticated LLM-based approaches. Traditional methods provide the foundation—BM25's term frequency and IDF weighting, cross-encoders' joint query-document attention, ColBERT's efficient multi-vector matching—each addressing specific limitations of its predecessors.[^20][^21][^1][^2]

LLM-based rerankers represent the latest frontier, leveraging instruction-following and reasoning capabilities to achieve strong zero-shot performance across diverse domains. Pairwise prompting offers practical effectiveness with moderate-sized models, listwise approaches excel with the largest LLMs, and attention-based methods like ICR provide efficient alternatives to generation. Fine-tuning and distillation techniques further enhance performance, enabling smaller models to match or exceed larger zero-shot counterparts.[^44][^32][^45][^30][^49][^50][^40][^29]

The future of reranking likely lies not in a single dominant approach but in intelligent hybrid systems that select the appropriate method based on query characteristics, latency requirements, and computational budgets. As LLMs continue to improve in efficiency and reasoning capabilities, their role in reranking will expand, particularly for complex, reasoning-intensive retrieval tasks where traditional methods struggle. However, the fundamental principles of term matching, semantic similarity, and contextualized interaction discovered through decades of research remain essential building blocks, now augmented rather than replaced by large-scale language models.[^55][^2][^51]
<span style="display:none">[^100][^101][^102][^103][^104][^105][^106][^107][^108][^109][^110][^111][^112][^113][^114][^115][^116][^117][^118][^119][^120][^121][^122][^123][^124][^125][^126][^127][^128][^129][^130][^131][^132][^133][^134][^135][^136][^137][^138][^139][^140][^141][^142][^143][^144][^145][^146][^147][^148][^149][^150][^151][^152][^153][^154][^155][^156][^157][^158][^159][^160][^161][^162][^163][^164][^165][^166][^167][^168][^169][^170][^171][^172][^173][^174][^175][^176][^177][^178][^179][^180][^181][^182][^183][^184][^185][^186][^187][^188][^189][^190][^191][^192][^193][^194][^195][^196][^197][^198][^199][^200][^201][^202][^203][^204][^205][^206][^207][^208][^209][^210][^211][^212][^213][^214][^215][^216][^217][^218][^219][^220][^61][^62][^63][^64][^65][^66][^67][^68][^69][^70][^71][^72][^73][^74][^75][^76][^77][^78][^79][^80][^81][^82][^83][^84][^85][^86][^87][^88][^89][^90][^91][^92][^93][^94][^95][^96][^97][^98][^99]</span>

<div align="center">⁂</div>

[^1]: https://zilliz.com/learn/what-are-rerankers-enhance-information-retrieval

[^2]: https://www.shaped.ai/blog/cross-encoder-rediscovers-a-semantic-variant-of-bm25

[^3]: https://machinelearningmastery.com/understanding-rag-iii-fusion-retrieval-and-reranking/

[^4]: https://www.allganize.ai/en/blog/innovating-ai-based-search-with-rerankers-understanding-context-and-reasoning

[^5]: https://training.continuumlabs.ai/disruption/search/bert-as-a-reranking-engine

[^6]: https://python.langchain.com/docs/integrations/document_transformers/cross_encoder_reranker/

[^7]: https://par.nsf.gov/servlets/purl/10273588

[^8]: https://cs.uwaterloo.ca/~jimmylin/publications/Pradeep_etal_ECIR2022.pdf

[^9]: https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html

[^10]: https://www.geeksforgeeks.org/nlp/self-attention-in-nlp/

[^11]: https://www.ibm.com/think/topics/self-attention

[^12]: https://jalammar.github.io/illustrated-transformer/

[^13]: https://sbert.net/docs/package_reference/cross_encoder/losses.html

[^14]: https://www.tdcommons.org/context/dpubs_series/article/2281/viewcontent/LAMBDALOSS_METRIC_DRIVEN_LOSS_FOR_LEARNING_TO_RANK.pdf

[^15]: https://marc.najork.org/papers/cikm2018.pdf

[^16]: https://www.shaped.ai/blog/lambdamart-explained-the-workhorse-of-learning-to-rank

[^17]: https://galileo.ai/blog/mastering-rag-how-to-select-a-reranking-model

[^18]: https://arxiv.org/html/2403.20222v1

[^19]: https://downloads.webis.de/publications/papers/schlatt_2024a.pdf

[^20]: https://arxiv.org/pdf/2112.01488.pdf

[^21]: http://arxiv.org/pdf/2004.12832.pdf

[^22]: https://jina.ai/news/what-is-colbert-and-late-interaction-and-why-they-matter-in-search/

[^23]: https://www.lancedb.com/documentation/studies/late-interaction-colbert.html

[^24]: https://zilliz.com/learn/explore-colbert-token-level-embedding-and-ranking-model-for-similarity-search

[^25]: https://weaviate.io/blog/late-interaction-overview

[^26]: https://dl.acm.org/doi/10.1145/3477495.3531835

[^27]: https://dl.acm.org/doi/10.1145/3726302.3730100

[^28]: https://arxiv.org/pdf/2112.06540.pdf

[^29]: https://arxiv.org/abs/2306.17563

[^30]: https://arxiv.org/abs/2312.02969

[^31]: http://arxiv.org/pdf/2306.17563.pdf

[^32]: https://aclanthology.org/2024.emnlp-main.250/

[^33]: https://fin.ai/research/using-llms-as-a-reranker-for-rag-a-practical-guide/

[^34]: https://arxiv.org/pdf/2311.01555.pdf

[^35]: https://www.zeroentropy.dev/articles/should-you-use-llms-for-reranking-a-deep-dive-into-pointwise-listwise-and-cross-encoders

[^36]: https://blog.reachsumit.com/posts/2023/12/prompting-llm-for-ranking/

[^37]: https://arxiv.org/html/2312.02969v1

[^38]: https://aclanthology.org/2023.emnlp-main.923.pdf

[^39]: https://aclanthology.org/2024.naacl-long.129.pdf

[^40]: https://arxiv.org/abs/2410.02642

[^41]: https://github.com/OSU-NLP-Group/In-Context-Reranking

[^42]: http://arxiv.org/pdf/2410.02642.pdf

[^43]: https://arxiv.org/html/2510.02219v1

[^44]: https://dl.acm.org/doi/10.1145/3626772.3657951

[^45]: https://arxiv.org/pdf/2407.02485.pdf

[^46]: https://arxiv.org/pdf/2310.04407.pdf

[^47]: https://arxiv.org/html/2403.19181v2

[^48]: https://arxiv.org/pdf/2312.02969.pdf

[^49]: https://arxiv.org/abs/2505.19274

[^50]: https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_How_to_Make_Cross_Encoder_a_Good_Teacher_for_Efficient_CVPR_2024_paper.pdf

[^51]: https://arxiv.org/html/2410.05168v1

[^52]: https://labelyourdata.com/articles/machine-learning/knowledge-distillation

[^53]: https://arxiv.org/html/2312.15842v1

[^54]: https://wandb.ai/byyoung3/ML_NEWS3/reports/Knowledge-distillation-Teaching-LLM-s-with-synthetic-data--Vmlldzo5MTMyMzA2

[^55]: https://www.chitika.com/neural-reranking-rag/

[^56]: https://developer.nvidia.com/blog/enhancing-rag-pipelines-with-re-ranking/

[^57]: https://www.datacamp.com/tutorial/rankgpt-rag-reranking-agent

[^58]: https://superlinked.com/vectorhub/articles/optimizing-rag-with-hybrid-search-reranking

[^59]: https://arxiv.org/abs/2509.11353

[^60]: https://openreview.net/forum?id=7bCwJebwag

[^61]: https://ieeexplore.ieee.org/document/9282636/

[^62]: https://ieeexplore.ieee.org/document/11086677/

[^63]: http://ieeexplore.ieee.org/document/7546332/

[^64]: https://arxiv.org/abs/2406.01197

[^65]: https://journals.rta.lv/index.php/ETR/article/view/8028

[^66]: https://www.semanticscholar.org/paper/dfc3a70c3d9de9aa1bc0c68198c3ab2eca650405

[^67]: https://arxiv.org/abs/2308.12022

[^68]: https://cad-journal.net/files/vol_21/Vol21NoS13.html

[^69]: https://ieeexplore.ieee.org/document/10317470/

[^70]: http://telkomnika.uad.ac.id/index.php/TELKOMNIKA/article/view/16734

[^71]: http://arxiv.org/pdf/2502.02464.pdf

[^72]: https://arxiv.org/pdf/2402.02764.pdf

[^73]: http://arxiv.org/pdf/2502.00709.pdf

[^74]: https://arxiv.org/pdf/2410.05168.pdf

[^75]: http://arxiv.org/pdf/2406.11720.pdf

[^76]: http://arxiv.org/pdf/2007.07101.pdf

[^77]: http://arxiv.org/pdf/2504.06276.pdf

[^78]: http://arxiv.org/pdf/2410.20286.pdf

[^79]: http://arxiv.org/pdf/2102.11903v1.pdf

[^80]: https://aclanthology.org/2022.emnlp-main.614.pdf

[^81]: https://arxiv.org/pdf/2305.11744.pdf

[^82]: https://arxiv.org/abs/2304.12570

[^83]: https://arxiv.org/pdf/2312.12430.pdf

[^84]: http://arxiv.org/pdf/2405.04844.pdf

[^85]: http://arxiv.org/pdf/2501.09186.pdf

[^86]: http://arxiv.org/pdf/2402.12997.pdf

[^87]: https://arxiv.org/html/2509.07163v1

[^88]: https://www.youtube.com/watch?v=V58mPkLB95o

[^89]: https://www.azion.com/en/learning/ai/what-are-rerankers/

[^90]: https://www.chitika.com/re-ranking-in-retrieval-augmented-generation-how-to-use-re-rankers-in-rag/

[^91]: https://ranjankumar.in/🔎-a-deep-dive-into-cross-encoders-and-how-they-work/

[^92]: https://pmc.ncbi.nlm.nih.gov/articles/PMC7148061/

[^93]: https://www.elastic.co/search-labs/blog/elastic-semantic-reranker-part-1

[^94]: https://samiranama.com/posts/what-are-rerankers/

[^95]: https://osanseviero.github.io/hackerllama/blog/posts/sentence_embeddings2/

[^96]: https://arxiv.org/abs/2503.03064

[^97]: https://dl.acm.org/doi/10.1145/3696410.3714717

[^98]: https://arxiv.org/abs/2504.10509

[^99]: https://arxiv.org/abs/2409.14744

[^100]: https://arxiv.org/abs/2507.05880

[^101]: https://arxiv.org/abs/2506.10859

[^102]: https://arxiv.org/pdf/2403.19181.pdf

[^103]: https://arxiv.org/pdf/2307.06857.pdf

[^104]: https://arxiv.org/pdf/2406.12433.pdf

[^105]: http://arxiv.org/pdf/2411.00142.pdf

[^106]: http://arxiv.org/pdf/2404.18185.pdf

[^107]: http://arxiv.org/pdf/2406.15657.pdf

[^108]: https://arxiv.org/pdf/2409.17711.pdf

[^109]: https://arxiv.org/pdf/2310.09497.pdf

[^110]: https://arxiv.org/pdf/2501.16302.pdf

[^111]: http://arxiv.org/pdf/2503.06034.pdf

[^112]: http://arxiv.org/pdf/2411.04602.pdf

[^113]: https://aicompetence.org/llamarank-vs-gpt-4-bard-and-claude/

[^114]: https://www.pinecone.io/learn/series/rag/rerankers/

[^115]: https://simonwillison.net/2024/Aug/11/using-gpt-4o-mini-as-a-reranker/

[^116]: https://arxiv.org/pdf/2508.16757.pdf

[^117]: https://unstructured.io/blog/improving-retrieval-in-rag-with-reranking?modal=try-for-free

[^118]: https://customgpt.ai/rag-reranking-techniques/

[^119]: https://aclanthology.org/2024.acl-short.59/

[^120]: https://arxiv.org/html/2501.09186v1

[^121]: https://www.reddit.com/r/Rag/comments/1ftmfc7/git_code_exploring_contextual_retrieval_with/

[^122]: https://arxiv.org/abs/2312.16159

[^123]: https://github.com/ielab/llm-rankers

[^124]: https://ai.gopubby.com/reranking-models-guide-2025-5-types-that-boost-search-results-by-40-f14ef217cd40

[^125]: https://gmd.copernicus.org/articles/17/8873/2024/

[^126]: http://www.scitepress.org/DigitalLibrary/Link.aspx?doi=10.5220/0001192100820093

[^127]: https://advanced.onlinelibrary.wiley.com/doi/10.1002/aelm.202200909

[^128]: https://ieeexplore.ieee.org/document/9723186/

[^129]: https://disk.yandex.ru/d/s6xGiq66bSI68A

[^130]: https://www.semanticscholar.org/paper/821d18c147ba83908ac3bfdb819e57d3c49d167c

[^131]: https://www.mdpi.com/2227-7102/14/6/563

[^132]: http://www.emerald.com/jeim/article/27/2/180-196/195543

[^133]: https://www.nature.com/articles/s41598-023-48378-w

[^134]: https://arxiv.org/abs/2211.16648

[^135]: http://arxiv.org/pdf/2406.14848.pdf

[^136]: https://arxiv.org/pdf/2206.13974.pdf

[^137]: http://arxiv.org/pdf/2502.03417.pdf

[^138]: http://arxiv.org/pdf/2109.07383.pdf

[^139]: http://arxiv.org/pdf/2405.05606.pdf

[^140]: https://aclanthology.org/2021.emnlp-main.191.pdf

[^141]: http://arxiv.org/pdf/2407.12385.pdf

[^142]: http://arxiv.org/pdf/2504.03965.pdf

[^143]: https://arxiv.org/pdf/2404.02587.pdf

[^144]: https://huggingface.co/antoinelouis/crossencoder-mt5-base-mmarcoFR

[^145]: https://arxiv.org/html/2505.04180v1

[^146]: https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/rankGPT/

[^147]: https://arxiv.org/html/2503.22672v1

[^148]: http://www.d2l.ai/chapter_attention-mechanisms-and-transformers/index.html

[^149]: https://adasci.org/a-hands-on-guide-to-enhance-rag-with-re-ranking/

[^150]: https://www.youtube.com/watch?v=eMlx5fFNoYc\&vl=en

[^151]: https://openaccess.thecvf.com/content/CVPR2021/papers/Yu_Landmark_Regularization_Ranking_Guided_Super-Net_Training_in_Neural_Architecture_Search_CVPR_2021_paper.pdf

[^152]: https://www.machinelearningmastery.com/the-transformer-attention-mechanism/

[^153]: https://www.techrxiv.org/users/845749/articles/1243437/master/file/data/ChatGPT-Is-All-You-Need-Architecture-Training-Procedure-Capabilities,-Limitations-Applications-NAIK/ChatGPT-Is-All-You-Need-Architecture-Training-Procedure-Capabilities,-Limitations-Applications-NAIK.pdf

[^154]: https://januverma.substack.com/p/llm-based-cross-encoder-for-recommendation

[^155]: https://www.sciencedirect.com/science/article/pii/S2214581823001258

[^156]: https://python.plainenglish.io/when-precision-demands-context-the-case-for-cross-encoder-rerankers-in-ai-retrieval-systems-99ca56a85911

[^157]: https://dl.acm.org/doi/10.1145/3539618.3591977

[^158]: https://arxiv.org/abs/2508.03555

[^159]: https://www.semanticscholar.org/paper/e22bcab2c4258d4c54a23d4238219a498bd8c4bb

[^160]: https://dl.acm.org/doi/10.1145/3639818

[^161]: https://arxiv.org/abs/2408.16672

[^162]: https://dl.acm.org/doi/10.1145/3626772.3657968

[^163]: https://www.semanticscholar.org/paper/1102a741dd0d43d347cb2d584e5c450bfe612eec

[^164]: https://ieeexplore.ieee.org/document/11094542/

[^165]: https://arxiv.org/pdf/2302.06587.pdf

[^166]: https://arxiv.org/pdf/2403.13291.pdf

[^167]: https://arxiv.org/pdf/2408.16672.pdf

[^168]: https://arxiv.org/pdf/2203.13088.pdf

[^169]: https://arxiv.org/pdf/2205.09707.pdf

[^170]: https://arxiv.org/pdf/2406.17968.pdf

[^171]: https://arxiv.org/pdf/2409.14683.pdf

[^172]: https://arxiv.org/pdf/2203.15328.pdf

[^173]: https://www.aclweb.org/anthology/2020.conll-1.17.pdf

[^174]: https://arxiv.org/pdf/2304.01982.pdf

[^175]: https://arxiv.org/pdf/2310.19488.pdf

[^176]: https://arxiv.org/pdf/2211.10411.pdf

[^177]: http://arxiv.org/pdf/2502.08818.pdf

[^178]: https://arxiv.org/pdf/2506.10859.pdf

[^179]: https://arxiv.org/abs/2105.01447

[^180]: https://qdrant.tech/articles/late-interaction-models/

[^181]: https://www.ibm.com/think/topics/attention-mechanism

[^182]: https://arxiv.org/html/2403.13291v1

[^183]: https://papers.dice-research.org/2024/KIAM-Reranking/public.pdf

[^184]: https://developer.ibm.com/articles/how-colbert-works/

[^185]: https://cs.uwaterloo.ca/~jimmylin/publications/Zhang_etal_ECIR2025.pdf

[^186]: https://trec.nist.gov/pubs/trec29/papers/UoB.DL.pdf

[^187]: https://huggingface.co/papers?q=listwise

[^188]: https://ieeexplore.ieee.org/document/10446621/

[^189]: https://ieeexplore.ieee.org/document/10550674/

[^190]: https://ieeexplore.ieee.org/document/10640279/

[^191]: https://arxiv.org/abs/2404.03323

[^192]: https://ieeexplore.ieee.org/document/10534280/

[^193]: https://ieeexplore.ieee.org/document/10145137/

[^194]: https://dl.acm.org/doi/10.1145/3708657.3708716

[^195]: https://dl.acm.org/doi/10.1145/3583780.3615001

[^196]: https://ieeexplore.ieee.org/document/10943638/

[^197]: https://arxiv.org/pdf/2012.06985.pdf

[^198]: http://arxiv.org/pdf/2402.14551.pdf

[^199]: http://arxiv.org/pdf/2203.05119.pdf

[^200]: https://arxiv.org/pdf/2110.08872.pdf

[^201]: http://arxiv.org/pdf/2211.03646.pdf

[^202]: https://arxiv.org/html/2407.07479v1

[^203]: https://arxiv.org/pdf/2308.14893.pdf

[^204]: https://arxiv.org/pdf/2305.10675.pdf

[^205]: https://arxiv.org/pdf/2207.01417.pdf

[^206]: https://arxiv.org/pdf/2107.07608.pdf

[^207]: https://aclanthology.org/2023.acl-long.819.pdf

[^208]: http://arxiv.org/pdf/2503.17526.pdf

[^209]: http://arxiv.org/pdf/2312.08520.pdf

[^210]: https://arxiv.org/html/2309.14580

[^211]: https://arxiv.org/pdf/2307.09614.pdf

[^212]: http://arxiv.org/pdf/2412.00101.pdf

[^213]: https://encord.com/blog/guide-to-contrastive-learning/

[^214]: https://lilianweng.github.io/posts/2021-05-31-contrastive/

[^215]: https://www.tensorflow.org/recommenders/examples/listwise_ranking

[^216]: https://dl.acm.org/doi/10.1145/3269206.3271784

[^217]: https://www.newline.co/@zaoyang/knowledge-distillation-for-llms-techniques-explained--7f55591b

[^218]: https://towardsdatascience.com/introduction-to-ranking-algorithms-4e4639d65b8/

[^219]: https://www.ibm.com/think/topics/knowledge-distillation

[^220]: https://towardsdatascience.com/a-practical-guide-to-contrastive-learning-26e912c0362f/

