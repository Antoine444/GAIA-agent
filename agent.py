import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from supabase import create_client, Client
from langchain_community.vectorstores import SupabaseVectorStore

load_dotenv()

# Enable debug logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers.
    Args:
        a: first int
        b: second int
    """
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Add two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a - b

@tool
def divide(a: int, b: int) -> int:
    """Divide two numbers.
    
    Args:
        a: first int
        b: second int
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

@tool
def modulus(a: int, b: int) -> int:
    """Get the modulus of two numbers.
    
    Args:
        a: first int
        b: second int
    """
    return a % b

@tool
def web_search(query: str) -> dict[str, str]:
    """Search DuckDuckGo for a query and return maximum 3 results."""
    logger.info(f"Searching DuckDuckGo for: {query}")

    search_docs = DuckDuckGoSearchResults(max_results=3).invoke(query=query)

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata.get("source", "unknown")}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"web_results": formatted_search_docs}

@tool
def wikipedia_search(query: str) -> dict[str, str]:
    """Search Wikipedia for a query and returns a maximum of 2 results."""
    logger.info(f"Searching Wikipedia for: {query}")

    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata.get("source", "unknown")}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"wikipedia_results": formatted_search_docs}

@tool
def arxiv_search(query: str) -> dict[str, str]:
    """Search Arxiv for a query and returns a maximum of 3 results."""
    logger.info(f"Searching Arxiv for: {query}")

    search_docs = ArxivLoader(query=query, load_max_docs=3).load()

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata.get("source", "unknown")}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arxiv_results": formatted_search_docs}


# Load system prompt
with open("system_prompt.txt", "r") as f:
    system_prompt = f.read()

system_message = SystemMessage(content=system_prompt)

# Initialize embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

supabase: Client = create_client(
    os.environ.get("SUPABASE_URL"), 
    os.environ.get("SUPABASE_SERVICE_KEY"))
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=hf_embeddings,
    table_name="documents",
    query_name="match_documents_langchain",
)
create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)

tools = [
    web_search,
    wikipedia_search,
    arxiv_search,
    add,
    subtract,
    multiply,
    divide,
    modulus
]

def build_graph(provider: str = "groq"):
    """Build the graph"""
    if provider == "groq":
        llm = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0.0
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")
    
    llm_with_tools = llm.bind_tools(tools)

    # Nodes
    def assistant(state: MessagesState):
        """Assistant node"""
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    def retriever(state: MessagesState):
        """Retriever node"""
        similar_question = vector_store.similarity_search(state["messages"][0].content)

        if similar_question:  # Check if the list is not empty
            example_msg = HumanMessage(
                content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
            )
            return {"messages": [system_message] + state["messages"] + [example_msg]}
        else:
            # Handle the case when no similar questions are found
            return {"messages": [system_message] + state["messages"]}

    builder = StateGraph(MessagesState)
    builder.add_node("retriever", retriever)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    logger.info("Successfully built graph")

    return builder.compile()


# Test case
if __name__ == "__main__":
    try:
        logger.info("Starting test case...")
        question = "When was a picture of St. Thomas Aquinas first added to the Wikipedia page on the Principle of double effect?"

        # Build the graph
        graph = build_graph(provider="groq")
        logger.info("Graph built successfully")

        # Run the graph
        logger.info(f"Asking question: {question}")
        messages = [HumanMessage(content=question)]
        result = graph.invoke({"messages": messages})

        logger.info("Response received:")
        for message in result["messages"]:
            if isinstance(message, HumanMessage):
                logger.info(f"Human: {message.content}")
            elif isinstance(message, SystemMessage):
                logger.info(f"System: {message.content}")
            else:
                logger.info(f"Message: {message.content}")

    except Exception as e:
        logger.error(f"Error during test execution: {e}")
