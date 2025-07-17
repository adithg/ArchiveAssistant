from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
import glob

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["PINECONE_API_KEY"] = os.getenv('PINECONE_API_KEY')

def main():
    # Load documents from local directory
    loader = DirectoryLoader('HenryTranscripts', glob="*.txt")
    docs = loader.load()
    
    print(f"Loaded {len(docs)} documents")
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )
    
    # Set up text splitter
    index_name = "archiveassistantlarge"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    
    print(f"Split into {len(split_docs)} chunks")
    
    # Create vector store
    vectorstore = PineconeVectorStore.from_documents(
        split_docs, 
        embeddings, 
        index_name=index_name
    )
    
    # Set up the QA chain
    llm = ChatOpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo"
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    # Interactive query loop
    print("\nVector store created successfully!")
    print("You can now ask questions about the documents.")
    print("Type 'quit' to exit.\n")
    
    while True:
        query = input("Enter your question: ")
        
        if query.lower() == 'quit':
            break
            
        try:
            result = qa_chain.invoke(query)
            print(f"\nAnswer: {result['result']}\n")
        except Exception as e:
            print(f"Error: {e}\n")

if __name__ == "__main__":
    main()