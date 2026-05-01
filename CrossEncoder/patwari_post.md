# Nice post from Puneet Patwari

> [link!](https://www.linkedin.com/posts/puneet-patwari_im-a-principal-engineer-with-12-years-of-share-7455235080731480064-GubO?utm_source=share&utm_medium=member_desktop&rcm=ACoAADAIBeMB-rMyiC5OyyimNp7leHf-DdUyUBE). Nothing spectacular, but a nice chain of thought!

If I were asked this in a system design round, I would treat it less like an “LLM question” and more like a **retrieval, ranking, and evidence-selection problem**.

The issue is simple:
You have 300 relevant documents.
You have 2 million tokens of possible context.
But the model can only see 128k tokens at once.

## [1] First, reduce the search space before retrieval

For a 10,000-person enterprise, every query should not search every document equally.

**Before semantic search, I would apply hard filters**:

- user permissions
- team or department
- source system
- document type
- time range
- project, customer, or ticket metadata
- freshness requirements

If the user asks, “What did we decide about Project Falcon pricing last quarter?” I should not be searching HR docs, old onboarding pages, or random Slack threads from 2021.

**Good retrieval starts by removing what should never be considered**.

## [2] Use hybrid retrieval

Vector search is good at meaning.
Keyword search is good at exact matches.
**Enterprise search needs both**.

Acronyms, Jira IDs, customer names, invoice numbers, internal project codenames, and release versions often fail with pure semantic search.

So I would run:

- **keyword** search for exact terms
- **vector search** for semantic meaning
- **metadata filters for narrowing**
- **permission filtering** before results reach the LLM

At this stage, 300 documents may still look relevant.
But 300 documents are not context.

They are only candidates.

## [3] Rerank before you let anything enter context

The first retrieval pass is built for recall. The reranker is built for precision.

The reranker should consider:

- direct relevance to the query
- freshness
- document authority
- source reliability
- whether the chunk actually contains an answer
- whether the document is outdated
- whether multiple chunks are repeating the same thing

I would rather send 30 high-signal chunks than 300 mediocre ones.

## [4] Pack evidence, not documents

**A common mistake is putting full documents into context.**

That burns tokens quickly and still may miss the key paragraph.

Instead, I would chunk documents intelligently and preserve useful context:

- document title
- section heading
- nearby paragraphs
- timestamp
- source link
- owner or team
- permission metadata

Then I would build the context window like an evidence packet:

- direct answer chunks
- supporting chunks
- recent updates
- authoritative decision docs
- conflicting evidence, if it exists

The goal is not to fill 128k tokens.
The goal is to give the model enough grounded evidence to answer safely.

## [5] Make “I don’t know” a valid system output

In enterprise search, **a confident wrong answer is worse than no answer.**

So the system should not always force completion.

The prompt should say:
Use only provided context.
Cite sources.
Mention uncertainty.
Ask for clarification when the query is ambiguous.
Do not invent missing details.

**If retrieval confidence is low, the product should say: “I found related documents, but not enough evidence to answer confidently.”**

## [6] Monitor retrieval quality, not just LLM quality

For this system, I would track:

- **retrieval recall**
- **reranker precision**
- source click-through
- **citation accuracy**
- **hallucination reports**
- **stale-document usage**
- **permission leakage**
- no-answer rate
- **query latency and cost**
