from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from dotenv import load_dotenv

load_dotenv()

loader = CSVLoader(file_path='data.csv')
docs = loader.load()

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore = Chroma.from_documents(docs, embedding_model, persist_directory="./api/chroma_db")

print("Database is created and embeddings are stored!")