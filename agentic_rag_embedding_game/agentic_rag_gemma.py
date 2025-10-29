import streamlit as st
import ollama
from agno.agent import Agent
from agno.knowledge.knowledge import Knowledge
from agno.vectordb.lancedb import LanceDb, SearchType
from agno.models.ollama import Ollama

# ---------------------------------------------------------------------
# ✅ Custom Ollama embedder
# ---------------------------------------------------------------------
class OllamaEmbedder:
    def __init__(self, model="embeddinggemma:latest", dimensions=768):
        self.model = model
        self.dimensions = dimensions

    def embed(self, text: str):
        """Generate a vector embedding for a given text using Ollama."""
        response = ollama.embed(model=self.model, input=text)
        return response["embeddings"][0]

# ---------------------------------------------------------------------
# Streamlit app setup
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Agentic RAG with EmbeddingGemma",
    page_icon="🔥",
    layout="wide"
)

@st.cache_resource
def load_knowledge_base(urls):
    """Builds a KnowledgeBase from remote PDFs and a LanceDB vector database."""
    vectordb = LanceDb(
        table_name="recipes",
        uri="tmp/lancedb",
        search_type=SearchType.vector,
        embedder=OllamaEmbedder(model="mxbai-embed-large"),
    )
    kb = Knowledge(
        name="PDF Knowledge",
        description="Knowledge base for PDFs",
        vector_db=vectordb,
    )
    # Add remote PDFs from URLs
    for url in urls:
        kb.add_content_async(name="Remote PDF", url=url)
    return kb

# ---------------------------------------------------------------------
# Session state for URLs
# ---------------------------------------------------------------------
if "urls" not in st.session_state:
    st.session_state.urls = []

kb = load_knowledge_base(st.session_state.urls)

# ---------------------------------------------------------------------
# Create Agno Agent
# ---------------------------------------------------------------------
agent = Agent(
    model=Ollama(id="llama3.2:latest"),
    knowledge=kb,
    instructions=[
        "Search the knowledge base for relevant information and base your answers on it.",
        "Be clear, and generate well-structured answers.",
        "Use clear headings, bullet points, or numbered lists where appropriate.",
    ],
    search_knowledge=True,
    show_tool_calls=False,
    markdown=True,
)

# ---------------------------------------------------------------------
# Sidebar: Add PDF URLs
# ---------------------------------------------------------------------
with st.sidebar:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("google.png")
    with col2:
        st.image("ollama.png")
    with col3:
        st.image("agno.png")

    st.header("🌐 Add Knowledge Sources")
    new_url = st.text_input(
        "Add URL",
        placeholder="https://example.com/sample.pdf",
        help="Enter a PDF URL to add to the knowledge base",
    )
    if st.button("➕ Add URL", type="primary"):
        if new_url:
            st.session_state.urls.append(new_url)
            kb.add_content_async(name=f"PDF {len(st.session_state.urls)}", url=new_url)
            st.success(f"✅ Added: {new_url}")
        else:
            st.error("Please enter a URL")

    if st.session_state.urls:
        st.subheader("📚 Current Knowledge Sources")
        for i, url in enumerate(st.session_state.urls, 1):
            st.markdown(f"{i}. {url}")

# ---------------------------------------------------------------------
# Main app content
# ---------------------------------------------------------------------
st.title("🔥 Agentic RAG with EmbeddingGemma (100% local)")
st.markdown(
    """
This app demonstrates an **agentic RAG** system using local models via [Ollama](https://ollama.com/):

- **EmbeddingGemma / mxbai-embed-large** for vector embeddings  
- **LanceDB** as the local vector database  
- **Llama 3.2** for answer generation  

Add PDF URLs in the sidebar to start and ask questions about the content.
"""
)

query = st.text_input("Enter your question:")

if st.button("🚀 Get Answer", type="primary"):
    if not query:
        st.error("Please enter a question")
    else:
        st.markdown("### 💡 Answer")
        with st.spinner("🔍 Searching knowledge and generating answer..."):
            try:
                response = ""
                resp_container = st.empty()
                gen = agent.run(query, stream=True)
                for chunk in gen:
                    if chunk.content:
                        response += chunk.content
                        resp_container.markdown(response)
            except Exception as e:
                st.error(f"Error: {e}")

# ---------------------------------------------------------------------
# How it works expander
# ---------------------------------------------------------------------
with st.expander("📖 How This Works"):
    st.markdown(
        """
**This app uses the Agno framework to create an intelligent Q&A system:**

1. **Knowledge Loading** – PDF URLs are processed and stored in LanceDB  
2. **OllamaEmbedder (custom)** – Generates local embeddings for semantic search  
3. **Llama 3.2** – Generates final answers based on retrieved context  

**Key Components**
- `OllamaEmbedder`: Uses Ollama’s `mxbai-embed-large` model for embeddings  
- `LanceDB`: Local vector database for similarity search  
- `Knowledge`: Manages PDF document loading  
- `Agno Agent`: Coordinates retrieval and generation
        """
    )
