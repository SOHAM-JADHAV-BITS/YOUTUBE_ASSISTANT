from dotenv import load_dotenv
import os
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain



load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

# Function to create vector database from YouTube URL
def create_vector_db_from_youtube_url(video_url):
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs, embeddings)
    return db

# Function to get response from query
def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0.6)
    prompt = PromptTemplate(input_variables=["question", "docs1"], template=f'''
    Answer the following question "{query}" by searching through the given document transcript "{docs_page_content}". If you don't have enough information,
    just return "I DON'T KNOW" as the final answer.
    ''')
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(question=query, docs1=docs_page_content)
    response = response.replace("\n", " ")
    return response, docs
