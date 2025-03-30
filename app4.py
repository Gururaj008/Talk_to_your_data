import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from docx import Document

GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]

def read_pdf(file):
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def read_docx(file):
    # Read DOCX content
    doc = Document(file)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def read_txt(file):
    text = file.getvalue().decode("utf-8")
    return text

def read_file(file):
    _, file_extension = os.path.splitext(file.name.lower())
    if file_extension == ".pdf":
        return read_pdf(file)
    elif file_extension == ".docx":
        return read_docx(file)
    elif file_extension == ".txt":
        return read_txt(file)
    else:
        st.warning("Unsupported file format. Please upload a PDF, DOCX, or TXT file.")
        return ""

def create_embeddings(uploaded_files):
    text = ""
    for file in uploaded_files:
        text += read_file(file)
    if text:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")

def generate_answer(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    
    # Updated prompt with instructions for bullet list formatting and clear sections
    prompt_template = """
You are an expert in evaluating profiles and answering questions based on the context provided.
Follow these instructions carefully:
1. Thoroughly analyze the provided context and answer the user question.
2. Provide your answer in the form of bullet points.
3. Try to provide as much information as possible.
4. If the context does not contain sufficient information to answer the user question, then respond: 
   "I can't answer the question based on the context provided, try rephrasing the question or ask a new question."
5. Do not fabricate any details.

Context:
{context}

Question:
{question}

Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def format_response(answer_text):
    """
    This function formats the raw answer text into a custom-styled HTML unordered list.
    Custom CSS is injected to improve the appearance (spacing, font-size, and margins).
    """
    # Custom CSS for the bullet list styling
    custom_css = """
    <style>
    .custom-ul {
      list-style: disc;
      margin-left: 30px;
      font-size: 18px;
      line-height: 1.6;
      color: #333;
    }
    .custom-ul li {
      margin-bottom: 10px;
    }
    </style>
    """
    # Split answer text into lines and create bullet items
    lines = answer_text.split('\n')
    list_items = ""
    for line in lines:
        clean_line = line.strip()
        if clean_line:
            # Remove any bullet markers
            if clean_line.startswith("*") or clean_line.startswith("-"):
                clean_line = clean_line.lstrip("*- ").strip()
            list_items += f"<li>{clean_line}</li>"
    # Combine CSS and formatted list
    formatted_html = custom_css + f"<ul class='custom-ul'>{list_items}</ul>"
    return formatted_html

if __name__ == "__main__":
    st.set_page_config(layout="wide")
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Agdasima');
        .custom-text { font-family: 'Agdasima', sans-serif; font-size: 70px; color: cyan; }
        </style>
        <p class="custom-text">SmartText Insight</p>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Agdasima');
        .custom-text-2 { font-family: 'Agdasima', sans-serif; font-size: 30px; color: cyan; }
        </style>
        <p class="custom-text-2">
            Unleashing the power of AI to transform your documents into enlightening conversations.
        </p>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # About the project
    st.subheader('About the project')
    st.markdown(
        '<div style="text-align: justify">SmartText Insight is an AI-driven application that simplifies document interaction and question-answering processes. Users can effortlessly upload PDF, DOCX, or TXT files and receive contextually relevant responses to their inquiries. The application employs advanced natural language processing techniques and document embeddings to enhance user understanding of textual content. The technology behind SmartText Insight includes the integration of Google DeepMind\'s "Gemini-Pro" model, GoogleGenerativeAIEmbeddings for document embeddings, and FAISS vector store for efficient indexing. With a straightforward and user-friendly interface, SmartText Insight streamlines document exploration and facilitates intelligent interactions with textual data.</div>',
        unsafe_allow_html=True
    )
    
    st.write('')
    st.write('')
    st.subheader('How to use the app')
    st.markdown(
        '<div style="text-align: justify">1. Upload your documents (PDF/DOCX/TXT) using the file uploader. Multiple files can be processed at once.</div>',
        unsafe_allow_html=True
    )
    st.write('')
    st.markdown(
        '<div style="text-align: justify">2. Click the "Click here to proceed" button to process and index the uploaded documents.</div>',
        unsafe_allow_html=True
    )
    st.write('')
    st.markdown(
        '<div style="text-align: justify">3. Enter your question in the text box below. The app will generate a response based on the content of your documents.</div>',
        unsafe_allow_html=True
    )
    st.write('')
    st.markdown(
        '<div style="text-align: justify">4. Click the "Fetch the answer" button to view the detailed, bullet-pointed response.</div>',
        unsafe_allow_html=True
    )
    
    st.write('')
    st.write('')
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader("Upload your Files", accept_multiple_files=True)

    if not uploaded_files:
        st.warning("Please upload files first.")

    # "Click here to proceed" button for processing files
    if st.button('Click here to proceed', use_container_width=True) and uploaded_files:
        with st.spinner("Training..."):
            create_embeddings(uploaded_files)
        st.success('Now I am ready to respond to your questions..')

    st.write('')
    # User question input
    user_question = st.text_input("Ask a Question from the File")

    # Process button
    if st.button("Fetch the answer", use_container_width=True) and uploaded_files and user_question:
        with st.spinner("Processing..."):
            response = generate_answer(user_question)
            res = response["output_text"]
            formatted_answer = format_response(res)
            
            st.markdown(formatted_answer, unsafe_allow_html=True)
            st.write('')
            st.write('')
            st.info("Feel free to ask more questions or upload additional documents.")

    st.divider()
    col1001, col1002, col1003, col1004, col1005 = st.columns([10,10,10,10,15])
    with col1005:
        st.markdown("""
            <style>
            @import url('https://fonts.googleapis.com/css2?family=Agdasima');
            .custom-text-10 { font-family: 'Agdasima', sans-serif; font-size: 28px; color: Gold; }
            </style>
            <p class="custom-text-10">An Effort by : MAVERICK_GR</p>
        """, unsafe_allow_html=True)
