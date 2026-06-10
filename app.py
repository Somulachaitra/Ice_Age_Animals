import streamlit as Streamlit
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI  # New cloud-based Google model import
from langchain_classic.chains import RetrievalQA  
from langchain_core.prompts import PromptTemplate 

# Page setup for web interface UI layout
Streamlit.set_page_config(page_title="Ice Age Explorer AI", page_icon="🦣")
Streamlit.title("🦣 Ice Age Megafauna Expert Assistant")
Streamlit.write("Ask specialized questions regarding prehistoric beasts, ecosystems, and extinction history.")

# SECURE SECURITY PATTERN: Input box inside the sidebar keeps keys off GitHub
Streamlit.sidebar.header("Google API Configuration")
api_key = Streamlit.sidebar.text_input("Enter Google API Key:", type="password")

# Cache the database loading step so it stays fast across user questions
@Streamlit.cache_resource
def initialize_rag_system(google_api_key):
    DB_PATH = "./chroma_db"
    # Load same embedding engine used during ingestion
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect back to the existing disk database
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # We turn the vector database into a "Retriever" that grabs top 3 context documents
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Connect to Google's cloud-based LLM engine
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.2, 
        google_api_key=google_api_key
    )
    
    # Customize the prompt constraints to prevent hallucinations
    custom_prompt_template = """You are an expert paleontologist specializing in the Pleistocene Ice Age. 
Use the following pieces of context to answer the question at the end accurately. 
If you do not know the answer based on the context, say "I don't have that specific record in my Ice Age knowledge base." Do not make up facts.

Context:
{context}

Question: {question}
Helpful Scientific Answer:"""

    PROMPT = PromptTemplate(
        template=custom_prompt_template, 
        input_variables=["context", "question"]
    )
    
    # Connect LLM, Retriever, and Prompts into an executable QA Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

# Only initialize the AI when the user enters an API key
if api_key:
    try:
        qa_pipeline = initialize_rag_system(api_key)

        # Create UI Chat Input Box
        user_query = Streamlit.text_input("Enter your paleontology inquiry:")

        if user_query:
            with Streamlit.spinner("Searching prehistoric records via Google Cloud..."):
                # Execute the RAG lookup and generation loop
                response = qa_pipeline.invoke({"query": user_query})
                
                # Display results
                Streamlit.subheader("Expert Response:")
                Streamlit.write(response["result"])
    except Exception as e:
        Streamlit.error(f"System Setup Message: Google connection issue. Check your key spelling. (Error: {e})")
else:
    Streamlit.info("💡 Complete Setup: Paste your Google Gemini API Key into the sidebar on the left to activate your assistant.")