import time
from dotenv import load_dotenv
import os
from langchain_community.llms import GooglePalm
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env
load_dotenv()

# Use ChatGoogleGenerativeAI instead of GooglePalm
llm = ChatGoogleGenerativeAI(
     model="gemini-1.5-pro",
     temperature=0.8,
)

# Initialize instructor embeddings using the Hugging Face model
embeddings = HuggingFaceInstructEmbeddings()
Vector_Path = "/Users/rafaelzieganpalg/Projects/Cognizant AI Bot/New_Faiss_Index"

def create_vector():
    try:
        # Load data from Dataset
        loader = CSVLoader(file_path='/Users/rafaelzieganpalg/Projects/Cognizant AI Bot/train.csv', source_column='question')
        data = loader.load()

        # Number of documents to simulate progress
        total_documents = len(data)
        processed_documents = 0

        # Start timing the vector creation and saving process
        start_time = time.time()

        # Create FAISS instance for vector database from 'data'
        Vector_db = FAISS.from_documents(documents=data, embedding=embeddings)

        # Simulate saving with progress updates
        print("Saving FAISS index...")
        for i in range(total_documents):
            processed_documents += 1
            time.sleep(0.05)  # Simulating time taken per document

            # Calculate elapsed time and estimated time remaining
            elapsed_time = time.time() - start_time
            progress = processed_documents / total_documents
            eta = (elapsed_time / progress) - elapsed_time

            # Display progress and ETA
            print(f"Progress: {processed_documents}/{total_documents} documents. ETA: {eta:.2f} seconds.", end="\r")

        # Save vector database locally
        Vector_db.save_local(Vector_Path)

        # Calculate total time taken
        total_time = time.time() - start_time
        print(f"\nVector database saved successfully in {total_time:.2f} seconds.")

    except Exception as e:
        print(f"An error occurred while creating the vector database: {e}")

def get_qa_chain():
    try:
        # Load the vector database from the local folder
        Vector_db = FAISS.load_local(Vector_Path, embeddings, allow_dangerous_deserialization=True)

        # Create a retriever for querying the vector database
        retriever = Vector_db.as_retriever()

        # Define the prompt template
        prompt_template = """Given the following context and a question, generate an answer based on this context or your answer.
                            In the answer, try to provide some text as possible from the "text" section in the source 
                            document context without making much changes. If the answer is not found in the context, 
                            kindly try to make up an answer but don't mention it as this document does not contain the answer.
                            CONTEXT: {context}
                            QUESTION: {question}"""
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain_type_kwargs = {"prompt": PROMPT}

        # Build the QA chain
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            input_key="query",
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        return chain
    except Exception as e:
        print(f"An error occurred while getting the QA chain: {e}")

if __name__ == "__main__":
    create_vector()
    chain = get_qa_chain()
    if chain:
        response = chain({"query": "what is the cause of common flu..?"})
        print(response)
