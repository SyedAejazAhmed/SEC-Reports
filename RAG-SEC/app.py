import os
import json
import requests
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage, SystemMessage
import streamlit as st

def download_pdf_from_github(pdf_url, save_path):
    """
    Download the PDF from GitHub and save it to the given path.
    """
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return save_path
    else:
        raise Exception(f"Failed to download PDF from GitHub. Status code: {response.status_code}")

class ExpertSystemRAGModel:
    def __init__(self, pdf_url, vectorstore_path, api_key):
        """
        Initialize the model with the GitHub URL for the PDF, vectorstore path, and the OpenAI API key.
        """
        self.api_key = api_key  # Access the OpenAI API key directly
        if not self.api_key:
            raise ValueError("OpenAI API key is not set. Ensure it is in the environment or secrets.toml.")

        # Download the PDF from GitHub
        self.pdf_path = download_pdf_from_github(pdf_url, "SEC-merged.pdf")
        self.vectorstore_path = vectorstore_path
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
        )

        # Initialize OpenAI's GPT-4 model
        self.llm = ChatOpenAI(model="gpt-4", openai_api_key=self.api_key)

        # PDF source mappings
        self.pdf_mappings = {
            "SEC-2019.pdf": (1, 53),
            "SEC-2020.pdf": (54, 90),
            "SEC-2021.pdf": (91, 136),
            "SEC-2022.pdf": (137, 257),
            "SEC-2023.pdf": (258, 350),
        }

    def get_source_and_page(self, merged_page_number):
        """
        Maps the merged PDF's page number to the corresponding source PDF and adjusted page number.
        """
        for pdf_name, (start_page, end_page) in self.pdf_mappings.items():
            if start_page <= merged_page_number <= end_page:
                relative_page_number = merged_page_number - start_page + 1
                return pdf_name, relative_page_number
        return "Unknown Source", merged_page_number

    def clean_content(self, text):
        """
        Clean unnecessary content like URLs and empty lines.
        """
        if not text:
            return ""
        lines = text.splitlines()
        cleaned_lines = [line.strip() for line in lines if line.strip() and "www." not in line]
        return " ".join(cleaned_lines)

    def process_pdf(self):
        """
        Processes the PDF, splits text into chunks, and assigns metadata.
        """
        print(f"Reading and processing PDF: {self.pdf_path}")
        pdf_reader = PdfReader(self.pdf_path)

        documents = []
        for page_num, page in enumerate(pdf_reader.pages):
            merged_page_number = page_num + 1  # Pages start from 1
            source, relative_page = self.get_source_and_page(merged_page_number)

            text = page.extract_text()
            if text:
                cleaned_text = self.clean_content(text)
                chunks = self.text_splitter.split_text(cleaned_text)

                for chunk_num, chunk in enumerate(chunks):
                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                "source": source,
                                "page": relative_page,
                                "chunk": chunk_num + 1,
                            },
                        )
                    )

        print(f"Processed {len(documents)} text chunks from the PDF.")
        return documents

    def create_vectorstore(self):
        """
        Processes the PDF, creates a FAISS vectorstore, and saves it.
        """
        documents = self.process_pdf()

        print("Creating vectorstore...")
        docsearch = FAISS.from_documents(documents, self.embeddings)

        print("Saving vectorstore in JSON format...")
        os.makedirs(os.path.dirname(self.vectorstore_path), exist_ok=True)
        with open(self.vectorstore_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"content": doc.page_content, "metadata": doc.metadata} for doc in documents],
                f,
                indent=4,
            )

        self.docsearch = docsearch  # Store vectorstore in memory
        print(f"Vectorstore created and saved at: {self.vectorstore_path}")

    def load_vectorstore(self):
        """
        Loads the vectorstore from the JSON file.
        """
        print("Loading vectorstore...")
        with open(self.vectorstore_path, "r", encoding="utf-8") as f:
            documents = json.load(f)

        self.docsearch = FAISS.from_documents(
            [Document(page_content=d["content"], metadata=d["metadata"]) for d in documents],
            self.embeddings,
        )
        print("Vectorstore loaded successfully!")

    def answer_query(self, query):
        """
        Answers a user query using the FAISS vectorstore and GPT-4.
        """
        if not hasattr(self, "docsearch"):
            print("Vectorstore not loaded. Loading...")
            self.load_vectorstore()

        print("Searching for relevant documents...")
        docs = self.docsearch.similarity_search(query, k=5)

        if not docs:
            return {
                "answer": "No relevant information found.",
                "sources": [],
            }

        # Combine document content for the context
        combined_content = "\n\n".join([f"Source: {doc.metadata.get('source')}, Page: {doc.metadata.get('page')}\n\n{doc.page_content}" for doc in docs])

        # Prepare messages for GPT-4 (Chat Completions API)
        messages = [
            SystemMessage(content="You are a helpful assistant. Answer questions based on the provided context."),
            HumanMessage(content=f"Context: {combined_content}\n\nQuestion: {query}")
        ]

        print("Generating answer with GPT-4...")
        try:
            response = self.llm.invoke(messages)  # Use invoke for chat models
            return {
                "answer": response.content,
                "sources": [{"source": doc.metadata.get("source"), "page": doc.metadata.get("page")} for doc in docs],
            }
        except Exception as e:
            return {
                "answer": f"Error during response generation: {str(e)}",
                "sources": [],
            }

# Streamlit App
def main():
    st.title("SEC-REPORTS")
    st.subheader("Based on PDFs of SEC-2019, 2020, 2021, 2022, 2023")

    # About Section in Sidebar
    with st.sidebar:
        st.header("About")
        st.write("""
            This app allows you to ask questions about SEC reports from 2019 to 2023.
            It uses a **Retrieval-Augmented Generation (RAG)** model powered by GPT-4.
            The model answers your queries by processing the content of these SEC reports.
            """)
        
    # Load secrets from Streamlit
    api_key = st.secrets["api_keys"]["openai"]
    vectorstore_path = st.secrets["paths"]["vectorstore_path"]

    # Initialize the RAG model
    @st.cache_resource
    def initialize_model():
        pdf_url = "https://raw.githubusercontent.com/SyedAejazAhmed/SEC-Reports/main/RAG-SEC/SEC-merged.pdf"
        model = ExpertSystemRAGModel(pdf_url, vectorstore_path, api_key)
        model.create_vectorstore()  # Ensure vectorstore is created
        return model

    model = initialize_model()

    # Form for input and submitting via Enter Key
    with st.form(key="query_form"):
        query = st.text_input("Enter your query:", placeholder="Ask me something about the SEC reports...")
        submit_button = st.form_submit_button("Submit")

    if submit_button:
        if query.strip():
            with st.spinner("Searching for relevant information..."):
                response = model.answer_query(query)
            
            # Display the answer
            st.subheader("Answer:")
            st.write(response["answer"])
            
            # Display the sources
            if response["sources"]:
                st.subheader("Sources:")
                for source in response["sources"]:
                    st.write(f"- **Source**: {source['source']}, **Page**: {source['page']}")
            else:
                st.write("No sources found.")
        else:
            st.warning("Please enter a query to proceed.")

if __name__ == "__main__":
    main()
