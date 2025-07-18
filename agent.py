import os
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.tools.ddg_search.tool import DuckDuckGoSearchResults
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.document_loaders import ArxivLoader
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore


load_dotenv()

@tool
def web_search(query: str) -> dict[str, str]:
    """Search Tavily for a query and return maximum 3 results.
    
    Args:
        query: The search query."""
    search_docs = DuckDuckGoSearchResults(max_results=3).invoke({"input": query})
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"web_results": formatted_search_docs}

@tool
def wikipedia_search(query: str) -> dict[str, str]:
    """
    Search Wikipedia for a query and returns a maximum of 2 results.
    
    Args:
        query: The search query.
    """
    search_docs = WikipediaLoader(query=query, load_max_docs=2).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ])
    return {"wikipedia_results": formatted_search_docs}

@tool
def arxiv_search(query: str) -> dict[str, str]:
    """Search Arxiv for a query and returns a maximum of 3 results.
    
    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ])
    return {"arxiv_results": formatted_search_docs}

# --- Load System Prompt ---
# This is the system prompt that will be used by the agent.
with open("system_prompt.txt", "r", encoding="utf-8") as file:
    system_prompt = file.read()

system_message = SystemMessage(content=system_prompt)

# Initialize HuggingFace embeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize Pinecone
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
if not pinecone_api_key:
    raise ValueError("PINECONE_API_KEY environment variable must be set.")

from pinecone import Pinecone, ServerlessSpec

# Create a Pinecone client instance
pc = Pinecone(api_key=pinecone_api_key)

# Now create or connect to your index
index_name = "documents"
index_dimension = 768  # Dimension for 'all-mpnet-base-v2' embeddings

try:
    # Check if index exists
    if index_name not in pc.list_indexes().names():
        # Create a new index
        pc.create_index(
            name=index_name,
            dimension=index_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",  # or your preferred cloud provider
                region="us-east-1"  # or your preferred region
            )
        )

    vector_store = PineconeVectorStore(
        index_name=index_name,
        embedding=hf_embeddings,
        pinecone_api_key=pinecone_api_key
    )
except Exception as e:
    raise RuntimeError(f"Failed to initialize Pinecone index: {e}")

create_retriever_tool = create_retriever_tool(
    retriever=vector_store.as_retriever(),
    name="Question Search",
    description="A tool to retrieve similar questions from a vector store.",
)

# Tools
tools = [
    web_search,
    wikipedia_search,
    arxiv_search
]

# Build graph function
def build_graph(provider: str = "huggingface"):
    """
    Build the graph
    """
    if provider == "huggingface":
        llm = ChatHuggingFace(
            llm=HuggingFaceEndpoint(
                model="Meta-DeepLearning/llama-2-7b-chat-hf",
                temperature=0,
            ),
        )
    else:
        raise ValueError("Invalid provider. Choose 'huggingface'.")
    
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Define the two nodes in the graph: retriever and assistant
    def assistant(state: MessagesState):
        """
        Assistant node: This node will use the LLM to generate a response based on the messages in the state.
        It will also use the tools if the condition is met.
        """
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    def retriever(state: MessagesState):
        """
        Retriever node: This node will use the vector store to find similar questions based on the first message in the state.
        It will return the similar question as an example message.
        """
        message_content = state["messages"][0].content
        if isinstance(message_content, str):
            query = message_content
        elif isinstance(message_content, list) and message_content and isinstance(message_content[0], str):
            query = message_content[0]
        else:
            raise ValueError("Message content must be a string for similarity_search.")
        similar_question = vector_store.similarity_search(query)
        example_msg = HumanMessage(
            content=f"Here I provide a similar question and answer for reference: \n\n{similar_question[0].page_content}",
        )
        return {"messages": [system_message] + state["messages"] + [example_msg]}


    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("retriever", retriever)
    graph_builder.add_node("assistant", assistant)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_edge(START, "retriever")
    graph_builder.add_edge("retriever", "assistant")
    graph_builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )
    graph_builder.add_edge("tools", "assistant")

    # Compile graph
    return graph_builder.compile()
