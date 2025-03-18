from typing import List

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


class DocumentStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
        self.documents = []
        self.index = None
        self.dimension = 384

    def add_documents(self, url: str) -> str:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find(id="mw-content-text")
        if content:
            paragraphs = content.find_all("p")
            self.documents = [p.text for p in paragraphs if len(p.text.split()) > 20]

        embeddings = self.embedder.encode(self.documents)
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(np.array(embeddings).astype("float32"))
        return f"Processed {len(self.documents)} documents"

    def search(self, query: str, k: int = 3) -> List[str]:
        query_vector = self.embedder.encode([query])
        D, I = self.index.search(np.array(query_vector).astype("float32"), k)
        return [self.documents[i] for i in I[0]]


# Initialize document store
doc_store = DocumentStore()


def retrieve_documents(url: str):
    """Tool function for document retrieval"""
    return doc_store.add_documents(url)


def search_context(query: str):
    """Tool function for searching documents"""
    return doc_store.search(query)


def check_relevance(context: List[str]) -> bool:
    """
    Tool function to check if retrieved context is relevant using an LLM.

    Args:
        context: List of context strings to evaluate

    Returns:
        bool: True if context is relevant, False otherwise
    """
    if not context:
        return False

    system_message = """You are a relevance checking assistant.
    Evaluate if the given context is relevant and substantial enough to answer questions.
    Return only 'true' or 'false'."""

    prompt = f"""Evaluate if this context is relevant and substantial (contains meaningful information):

    Context: {context}

    Return only 'true' or 'false'."""

    messages = run_llm(content=prompt, system_message=system_message)

    # Get the last message which contains the LLM's response
    result = messages[-1].content.lower().strip()
    return result == "true"


def rewrite_query(query: str, context: List[str]) -> str:
    """
    Tool function to rewrite a query based on context using an LLM.

    Args:
        query: Original query to rewrite
        context: List of context strings to use for rewriting

    Returns:
        str: Rewritten query incorporating context
    """
    system_message = """You are a query rewriting assistant.
    Your task is to rewrite the original query to incorporate relevant context.
    Maintain the original intent while making it more specific based on the context."""

    prompt = f"""Original Query: {query}

Available Context: {context}

Rewrite the query to be more specific using the context.
Maintain the original intent but make it more precise."""

    messages = run_llm(content=prompt, system_message=system_message)

    # Get the last message which contains the rewritten query
    return messages[-1].content.strip()


rag_agent = Agent(
    name="RAG Agent",
    system_message="""You are an intelligent RAG agent that follows a specific workflow:
    1. First, determine if you need to retrieve documents
    2. If yes, use the retrieval tool
    3. Check the relevance of retrieved documents
    4. Either rewrite the query or generate an answer
    5. Always cite your sources from the context

    Be explicit about each step you're taking.""",
    tools=[retrieve_documents, search_context, check_relevance, rewrite_query],
    llm="gpt-4o-mini",
)


def run_rag_agent(agent, messages):
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:
        # Get tool schemas and tool map
        tool_schemas = [function_to_schema(tool) for tool in agent.tools]
        tools = {tool.__name__: tool for tool in agent.tools}

        # Make API call with system message and history
        response = client.chat.completions.create(
            model=agent.llm,
            messages=[{"role": "system", "content": agent.system_message}] + messages,
            tools=tool_schemas,
        )

        # Get and append the assistant's message
        message = response.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": message.tool_calls
                if hasattr(message, "tool_calls")
                else None,
            }
        )

        if message.content:
            print(f"{agent.name}:", message.content)

        if not hasattr(message, "tool_calls") or not message.tool_calls:
            break

        # Handle tool calls
        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, agent.name)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result),
                }
            )

    return messages[num_init_messages:]


# Main execution
messages = []

# First, index a document
url = "https://en.wikipedia.org/wiki/Alan_Turing"
messages.append(
    {"role": "user", "content": f"Please retrieve and index this document: {url}"}
)

while True:
    try:
        response = run_rag_agent(rag_agent, messages)
        messages.extend(response)

        user_input = input("\nUser (type 'quit' to exit): ")
        if user_input.lower() == "quit":
            break

        messages.append({"role": "user", "content": user_input})

    except Exception as e:
        print(f"Error occurred: {e}")
        break
