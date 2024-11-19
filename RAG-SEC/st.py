import streamlit as st
from model import ExpertSystemRAGModel
import os

# Function to load the model only if not already loaded
@st.cache_resource
def load_model():
    model = ExpertSystemRAGModel(vectorstore_path="stored_model/vectorstore.json")  # Use pre-saved vectorstore
    model.load_vectorstore()  # Load vectorstore from saved file
    return model

# Check if the vectorstore file exists
vectorstore_path = "stored_model/vectorstore.json"
if os.path.exists(vectorstore_path):
    # Use the pre-saved vectorstore
    st.session_state["model"] = load_model()
else:
    # Warn the user if the vectorstore file is missing
    st.error(
        f"Vectorstore file not found at {vectorstore_path}. Please ensure the file is created and placed in the correct location."
    )
    st.stop()  # Stop the app if vectorstore is not available

# Streamlit App
st.title("Expert System - SEC Reports")
st.write(
    """
    Welcome to the SEC Reports Expert System! This tool uses a Retrieval-Augmented Generation (RAG) 
    model to provide answers based on SEC reports. 
    Type your query below to get started.
    """
)

# Query Input
query = st.text_input("Enter your query:")

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        # Display a loading spinner
        with st.spinner("Processing your query..."):
            # Use the loaded model to answer the query
            response = st.session_state["model"].answer_query(query)

        # Display the answer and sources
        if "error" in response["answer"].lower():
            st.error(response["answer"])
        else:
            st.subheader("Answer:")
            st.write(response["answer"])

            st.subheader("Sources:")
            if response["sources"]:
                for source in response["sources"]:
                    st.write(f"- **{source['source']}**, Page: {source['page']}")
            else:
                st.write("No sources available.")

st.sidebar.title("About")
st.sidebar.info(
    """
    This is a Retrieval-Augmented Generation (RAG) system powered by LangChain, OpenAI GPT-4, 
    and FAISS. It processes SEC reports to provide insightful answers with references.
    """
)
