
from langchain_community.retrievers import BM25Retriever
from langchain_community.document_transformers import (
    LongContextReorder,
    EmbeddingsRedundantFilter,
)
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)


from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

# For the reranker
from sentence_transformers import CrossEncoder

# Replace the CrossEncoderReranker initialization with this:
def initialize_reranker():
    """Initialize the cross-encoder reranker"""
    try:
        model = CrossEncoder("BAAI/bge-reranker-base")
        
        def rerank_documents(documents: List[Document], query: str) -> List[Document]:
            if not documents:
                return []
            
            # Prepare document-query pairs
            pairs = [[query, doc.page_content] for doc in documents]
            
            # Get scores from the model
            scores = model.predict(pairs)
            
            # Sort documents by score
            scored_documents = list(zip(documents, scores))
            scored_documents.sort(key=lambda x: x[1], reverse=True)
            
            # Return top 3 documents
            return [doc for doc, score in scored_documents[:3]]
        
        return rerank_documents
    except Exception as e:
        st.error(f"Error initializing reranker: {str(e)}")
        return None

# Then modify the find_relevant_context function to use the new reranker:
def find_relevant_context(query: str, preprocessed_data: dict, k: int = 5) -> str:
    """Find most relevant texts using cosine similarity and reranking"""
    # Get query embedding
    embeddings = OpenAIEmbeddings()
    query_embedding = embeddings.embed_query(query)
    
    # Calculate initial similarities
    similarities = [
        np.dot(query_embedding, doc_embedding) / 
        (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding))
        for doc_embedding in preprocessed_data['embeddings']
    ]
    
    # Get top k most similar texts
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    candidate_texts = [preprocessed_data['texts'][i] for i in top_k_indices]
    
    # Rerank if reranker is available
    reranker = st.session_state.reranker
    if reranker:
        # Convert texts to Documents
        docs = [Document(page_content=text) for text in candidate_texts]
        
        # Rerank
        reranked_docs = reranker(docs, query)
        relevant_texts = [doc.page_content for doc in reranked_docs]
    else:
        # If reranker isn't available, just use top 3 from initial ranking
        relevant_texts = candidate_texts[:3]
    
    return "\n\n".join(relevant_texts)
# Updated prompt template
PROMPT_TEMPLATE = """Given the context below, please provide an accurate and detailed response to the user's inquiry about the MS in Applied Data Science program at the University of Chicago.
Respond with a JSON object (not in a code block) in the following format:
{{
    "answer": "detailed answer using only information from the context",
    "confidence": "high/medium/low, based on how thoroughly the context supports your answer",
    "reasoning": "concise explanation of how the context informs your answer"
}}
If the context does not contain enough information, respond with:
{{
    "answer": "Cannot be answered from the given context",
    "confidence": "low",
    "reasoning": "The provided context lacks sufficient details to answer this inquiry"
}}
Context: {context}
Question: {question}
Guidelines:
1. Use only information from the context provided.
2. Craft a detailed response that addresses the inquiry directly.
3. Use bullet points for distinct points if applicable.
4. Include a relevant URL if it directly supports the answer.
5. Assign confidence based on the relevance and completeness of the context.
6. Briefly explain your reasoning, focusing on how the context justifies your answer.
7. Return ONLY the JSON object without any code block markers or extraneous text.
"""

def format_response(response_text: str) -> dict:
    """Format and validate the response as JSON"""
    try:
        # Parse the response as JSON
        response_dict = json.loads(response_text)
        return response_dict
    except json.JSONDecodeError:
        # If parsing fails, return an error response
        return {
            "answer": "Error processing response",
            "confidence": "low",
            "reasoning": "Failed to parse response as JSON"
        }

def display_json_response(response_dict: dict, message_index: int):
    """Display the JSON response with expandable details"""
    # Always display the answer
    st.write(response_dict["answer"])
    
    # Create a unique key for this message's expander
    expander_key = f"details_{message_index}"
    
    # Add a button to show/hide details
    if st.button("Show Details", key=f"button_{message_index}"):
        st.session_state.expanded_messages.add(message_index)
    
    # If this message is expanded, show the confidence and reasoning
    if message_index in st.session_state.expanded_messages:
        st.markdown("---")
        # Display confidence with appropriate color
        confidence = response_dict["confidence"].lower()
        confidence_color = {
            "high": "green",
            "medium": "orange",
            "low": "red"
        }.get(confidence, "gray")
        
        st.markdown(f"**Confidence:** ::{confidence_color}[{confidence}]")
        
        # Display reasoning
        st.markdown("**Reasoning:**")
        st.write(response_dict["reasoning"])
        
        # Add button to hide details
        if st.button("Hide Details", key=f"hide_{message_index}"):
            st.session_state.expanded_messages.remove(message_index)
            st.rerun()

# Main app interface
st.title("🎓 MADS Program Chat Assistant")

# Check for preprocessed data
if not os.path.exists('processed_data.pkl'):
    st.error("""
    Preprocessed data not found! Please run preprocess.py first.
    Check the README.md for instructions.
    """)
    st.stop()

# Load preprocessed data and initialize reranker
if st.session_state.preprocessed_data is None:
    with st.spinner("Loading knowledge base and initializing models..."):
        st.session_state.preprocessed_data = load_preprocessed_data()
        st.session_state.reranker = initialize_reranker()
        if st.session_state.preprocessed_data is None:
            st.error("Failed to load knowledge base")
            st.stop()

# API Key input
if not st.session_state.api_key:
    with st.form("api_key_form"):
        api_key = st.text_input("OpenAI API Key:", type="password")
        submitted = st.form_submit_button("Submit")
        
        if submitted and api_key.startswith('sk-'):
            st.session_state.api_key = api_key
            os.environ['OPENAI_API_KEY'] = api_key
            st.rerun()
        elif submitted:
            st.error("Please enter a valid OpenAI API key")

else:
    # Display chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                display_json_response(message["content"], i)
            else:
                st.write(message["content"])

    # Chat input
    if question := st.chat_input("Ask about the MADS program"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        # Generate response
        with st.chat_message("assistant"):
            try:
                # Get relevant context
                relevant_context = find_relevant_context(
                    question, 
                    st.session_state.preprocessed_data
                )
                
                # Create message with context and question
                prompt = PROMPT_TEMPLATE.format(
                    context=relevant_context,
                    question=question
                )
                
                # Get response from ChatGPT
                chat = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0
                )
                
                response = chat.predict(prompt)
                
                # Parse and format the response
                response_dict = format_response(response)
                
                # Display the formatted response
                display_json_response(response_dict, len(st.session_state.messages))
                
                # Save assistant response
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_dict}
                )
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Sidebar controls
    with st.sidebar:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.session_state.expanded_messages = set()
            st.rerun()
        if st.button("Reset API Key"):
            st.session_state.api_key = None
            st.session_state.messages = []
            st.session_state.expanded_messages = set()
            st.rerun()

# Footer
st.markdown("---")
st.caption("Powered by LangChain and OpenAI")
