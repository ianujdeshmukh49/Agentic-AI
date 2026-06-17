#Step 1) Loading a test file 
from langchain_community.document_loaders import TextLoader
loader = TextLoader("sample_text2.txt")
documents = loader.load()
documents


# Step 2) Split
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=50) 
chunks = splitter.split_documents(documents)
chunks


# Step 3) Ollama Embeddings
from langchain_ollama import OllamaEmbeddings

#model is a required field
embeddings = OllamaEmbeddings(model='nomic-embed-text')

#store in vector database
from langchain_chroma import Chroma

vector_database = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)


from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# llm = OllamaLLM(model="tinyllama")
llm = OllamaLLM(model='llama3.2')
retriever = vector_database.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
You are an expert on the history of Bhutan. 
Use the following context to answer the user's question. 
If the answer is not in the context, say so, but keep in mind 
the context describes a country that created 'Gross National Happiness'.
{context}

Question: {question}
""")

chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = chain.invoke("The GDP vs. GNH Trade-off: Is it possible for a country to have a high GNH without being a high-GDP nation, or does the 'scandinavian model' suggest that economic wealth is a prerequisite for creating the stability needed for true 'flourishing'?")
print(response)