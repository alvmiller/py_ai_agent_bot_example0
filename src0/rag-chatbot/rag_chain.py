from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
#from langchain.llms import Gemini
from langchain.llms import Fireworks
from dotenv import load_dotenv
import os

from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor


load_dotenv()

class RAGChain:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = self.get_llm()
        print('OPENAI_API_KEY is    : ', OPENAI_API_KEY)
        print('GEMINI_API_KEY is    : ', GEMINI_API_KEY)
        print('FIREWORKS_API_KEY is : ', FIREWORKS_API_KEY)
        print('GOOGLE_API_KEY is    : ', GOOGLE_API_KEY)

    def get_llm(self):
        print('Searching API Key... ')
        if os.getenv("OPENAI_API_KEY"):
            print('Searching OPENAI_API_KEY... ')
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
        elif os.getenv("GEMINI_API_KEY"):
            print('Searching GEMINI_API_KEY... ')
            #return Gemini(api_key=os.getenv("GEMINI_API_KEY"), temperature=0)
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        elif os.getenv("GOOGLE_API_KEY"):
            print('Searching GOOGLE_API_KEY... ')
            return ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        elif os.getenv("FIREWORKS_API_KEY"):
            print('Searching FIREWORKS_API_KEY... ')
            return Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"), temperature=0)
        else:
            raise ValueError("No valid API key found! Please set one in .env file.")

    def create_chain(self):
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        return qa_chain

if __name__ == "__main__":
    from document_processor import DocumentProcessor
    from embedding_indexer import EmbeddingIndexer

    processor = DocumentProcessor("data/sample_text.txt")
    texts = processor.load_and_split()

    indexer = EmbeddingIndexer()
    vectorstore = indexer.create_vectorstore(texts)

    rag_chain = RAGChain(vectorstore)
    qa_chain = rag_chain.create_chain()

    query = "What is the capital of France?"
    result = qa_chain({"query": query})
    print(f"Answer: {result['result']}")
