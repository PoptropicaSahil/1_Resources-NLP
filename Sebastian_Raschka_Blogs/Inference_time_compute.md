# Inference-time scaling

> Ref: [Sebastian Raschka's Magazine](https://magazine.sebastianraschka.com/p/state-of-llm-reasoning-and-inference-scaling?trk=feed_main-feed-card_feed-article-content )

Main techniques in the blog

## Wait Tokens (Budget Forcing)

**The "s1: Simple Test-Time Scaling" technique introduces "wait" tokens and using an end-of-thinking token delimiter ("Final Answer:")** to control the length of responses and encourage longer, more thoughtful outputs. This method allows the model to generate longer responses, self-verify, and correct itself without modifying the underlying model weights.

## Thought Switching Penalty (TIP)

The TIP method addresses the "underthinking" issue by **modifying the logits of thought-switching tokens to discourage premature reasoning path transitions.** This technique improves accuracy across challenging test sets without requiring model fine-tuning.

### Chain-of-Associated-Thoughts (CoAT)

CoAT combines Monte Carlo Tree Search with an "associative memory" that serves as the LLM's knowledge base during the exploration of reasoning pathways. This allows the model to consider earlier reasoning paths and use dynamically evolving information during response generation.

### Self-Backtracking

This mechanism allows LLMs to improve their reasoning by learning when and where to backtrack during inference. It uses a tree-based search that explores alternative solutions without relying on external reward models, improving the model's ability to correct suboptimal reasoning paths.

### Adaptive Token Routing

**The Inner Thinking Transformer (ITT) employs Adaptive Token Routing to allocate more compute to difficult tokens.** This technique allows certain tokens to pass through the same layer multiple times for additional processing, effectively increasing the inference-compute budget for challenging parts of the input.
