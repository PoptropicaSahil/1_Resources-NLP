# The other parts from the blog

> The blog seems long and contains a lot of little knowledge bits about how can a system be improved. I felt that I should make the code file with only code and make this file seperately with all the knowledge parts \
> (All from the blog) The code here will be for educational purpose only, to see the whole code, visit this [repo](https://github.com/goyalpramod/paper_implementations/blob/main/AI_agents_from_first_principles.ipynb). The code in this section of the blog was heavily inspired from [OpenAI Cookbook](https://cookbook.openai.com/examples/orchestrating_agents)

Multiple frameworks have been developed, like Langchain, LLamaIndex, Langflow. But if you are anything like me, *you hate abstraction layers which add needless complexity during debugging.* So in this blog I would like to breakdown how to **build AI agents from first principles purely using Python and the core libraries.**

## Embedding Models

Examples [BGE](https://huggingface.co/BAAI/bge-large-en-v1.5), [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings), [Jina embeddings](https://huggingface.co/jinaai/jina-embeddings-v2-base-en)

> You can compare the latest best embedding model by going to the [mteb leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

## Vector DB

Databases that efficiently store and search through embeddings using algorithms like **Approximate Nearest Neighbors (ANN)**. They solve the problem of finding similar vectors in high-dimensional spaces quickly.

Popular options with their strengths:

- **Qdrant: Rust-based, excellent for production deployments**
- **Weaviate: GraphQL-native, good for multi-modal data**
- Pinecone: Fully managed, automatic scaling
- Chroma: Python-native, perfect for development
- Milvus: Cloud-native, handles billions of vectors

## Different RAG Approaches for Different Needs

- **Basic RAG**: Single retrieval-injection cycle  
  - Best for: Simple queries, quick implementation  
  - Limitation: Context window size  

- **Iterative RAG**: Multiple retrieval rounds  
  - Best for: Complex queries, follow-up questions  
  - Limitation: Higher latency, more expensive  

- **Hybrid RAG**: Combined search methods  
  - Best for: Large diverse datasets  
  - Limitation: Complex implementation  

- **Re-ranker RAG**: Two-stage retrieval  
  - Best for: High accuracy requirements  
  - Limitation: Higher compute costs  

## Performance Optimization

Critical optimizations for production:

- **Caching**: Store frequent query results  
- **Batching**: Process embeddings in groups  
- **Filtering**: Use metadata before semantic search  
- **Index Optimization**: Implement HNSW algorithms  
- **Query Preprocessing**: Rewrite queries for better retrieval  

## Evaluation Metrics

Monitor these metrics for RAG performance:

- **Retrieval Precision**: Relevance of retrieved contexts  
- **Answer Correctness**: Accuracy of generated responses  
- **Latency**: Response time per query  
- **Cost**: Tokens used per response  

Consider checking out [TruLens](https://www.trulens.org/) and [DeepEval](https://github.com/confident-ai/deepeval).

## Common Failure Patterns

- Hallucination from irrelevant context  
- Missing key information due to poor chunking  
- High latency from inefficient retrieval  

We can dive deeper into each of these topics. I recommend the following [Pinecone Engineering Blogs](https://www.pinecone.io/blog/), [Qdrant Engineering Blogs](https://qdrant.tech/blog/)

---

## Evaluation Framework

Set up clear metrics before building:

1. Task Success Rate: Can your agent complete the assigned tasks?
2. Response Quality: Are the responses accurate and relevant?
3. Operational Metrics: Latency, cost per request, error rates
4. Safety Metrics: Rate of harmful or inappropriate responses

### Development Workflow

**Start Simple**

- Build a basic agent that handles the core task
- Use minimal tools and straightforward prompts
- Test with real scenarios, not just ideal cases

**Measure Everything**

- Log all interactions and their outcomes
- Track token usage and response times
- Monitor error patterns and edge cases

HuggingFace has a nice [blog](https://huggingface.co/docs/smolagents/en/tutorials/inspect_runs) on using OpenTelemetry for inspecting runs.

**Iterate Intelligently**

- Add complexity only when metrics show need
- A/B test prompt changes and tool additions
- Document why each feature was added

**Optimize Systematically**

- Cache frequent queries and their results
- Batch similar operations when possible
- Use cheaper models for simpler tasks

> Anthropic has a nice [cookbook](https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/evaluator_optimizer.ipynb) on creating a testing framework.

### Key Takeaways

- **Minimize LLM Usage**: Each call costs time and money
- **Start Simple**: Add complexity only when needed
- **Measure Everything**: You can’t improve what you don’t measure
- **Test Thoroughly**: Include edge cases and error scenarios
- **Document Decisions**: Keep track of why each feature exists
