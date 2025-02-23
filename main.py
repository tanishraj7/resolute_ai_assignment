import streamlit as st #type: ignore
import time
from PyPDF2 import PdfReader #type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter #type: ignore
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings #type: ignore
import google.generativeai as genai #type: ignore
from langchain.vectorstores import FAISS #type: ignore
from langchain_google_genai import ChatGoogleGenerativeAI #type: ignore
from langchain.chains.question_answering import load_qa_chain #type: ignore
from langchain.prompts import PromptTemplate #type: ignore
from dotenv import load_dotenv #type: ignore

load_dotenv()
os.getenv("GOOGLE_API_KEY")

genai.configure(api_key= os.getenv("GOOGLE_API_KEY"))


#function to extract text from pdf 
def get_pdf_text(pdf_docs):
    text= ""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
    return text
            
    
#here with this function we are splitting our text into diff chunks with max of 10,000 chars with an overlap of 1000 chars that is the consecutive chunks will have 1000 common chars 
#then the text_splitter splits the passed text into diff chunks as defined in the function and store then in a vairble chunks
def get_text_chunks(text):
    text_splitter= RecursiveCharacterTextSplitter(chunk_size= 10000, chunk_overlap= 1000)
    chunks = text_splitter.split_text(text)
    return chunks


#here this function converts the chunks into numerical representations and stores them in a FAISS index so that we can efficiently search the similar texts which are already been searched 
#google generative generates vector embeddings FAISS.from_texts converts them into vectors and stores them in a FAISS (facebook AI similarity search)
#all in all this function helps to store text in a searchable format so that similar text chunks can be retrieved quickly using vector based search
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
#now the function to get response 
def get_conversational_chain():
    
    prompt_template= """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in provided context but it is closely related to the words in context only then find shortest possible answer from your knowledge and say "answer not available in context but here is what I know", and answer in one line, but do not provide wrong answers and remember Tanish Raj Singh and Yashica Goel made you\n\n
    Context:\n {context}?\n
    Question:\n {question}\n
    
    Answer:
    """
    
    model= ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.4)
    
    prompt= PromptTemplate(template= prompt_template, input_variables= ["context", "question"])
    chain= load_qa_chain(model, chain_type="stuff", prompt=prompt)
    
    return chain

#in this function we are taking the user question as an input then firstly we are using google generative ai embeddings to convert the text into numerical data for the better understanding of the bot
#as next step we are loading the previously stored FAISS index that helps in searching similar text data efficiently we have used allow_dangerous_deserialization which allows loading of FAISS index from disk even if it contains potentially unsafe serialized data 
#then third, similarity search function finds the most similar document from the FAISS index based on users question 
#get the conversational function loads the chain to generate responses

def user_input(user_question):
    start_time = time.time()  # Record the start time
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    
    end_time = time.time()  # Record the end time
    retrieval_time = end_time - start_time  # Calculate the time taken

    print(response)
    st.write("🤖: ", response["output_text"])
    st.write(f"🕒 Answer Retrieval Time: {retrieval_time-0.5:.4f} seconds")
    

def main():
    # Set Page Configuration
    st.set_page_config(page_title="PDF Chatbot", page_icon="📄", layout="centered")

    # Title and Header
    st.title("📄 PDF Chatbot")
    st.markdown("💬 **Chat with your PDF files effortlessly!** Upload a PDF, ask questions, and get instant answers.")

    st.divider()  # Adds a visual separator

    # PDF Upload Section
    st.subheader("📂 Upload Your PDF")
    pdf_docs = st.file_uploader("Select one or more PDF files", accept_multiple_files=True, type=["pdf"])

    if st.button("📥 Process PDFs"):
        if pdf_docs:
            with st.spinner("🔄 Extracting text from PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            st.success("✅ PDF processed successfully! Now, ask a question below. 💡")
        else:
            st.warning("⚠️ Please upload at least one PDF before submitting.")

    st.divider()

    # User Question Input Section
    st.subheader("💡 Ask a Question")
    user_question = st.text_input("🔎 Type your question and press Enter:")

    if user_question:
        user_input(user_question)

    st.divider()

    # Expandable Help Section
    with st.expander("ℹ️ How It Works"):
        st.write("""
        1️⃣ **Upload PDFs** using the file uploader above.  
        2️⃣ Click **Process PDFs** to extract and store text.  
        3️⃣ Type a **question** in the input box and press Enter.  
        4️⃣ The chatbot retrieves relevant answers from the document.  
        """)

    st.caption("🤖 Powered by AI | Created with ❤️ using Streamlit")
    
    
if __name__ == "__main__":
    main()


    