from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embedding_model = MistralAIEmbeddings(
    model="mistral-embed"
)

vectorstore = Chroma(
    embedding_function=embedding_model,
    persist_directory= "chroma_db"
)

retriever = vectorstore.as_retriever(
    search_type = "mmr",
    search_kwargs = {
        "k": 4,
        "fetch_k": 10,
        "lambda_mult": 0.5
    }
)

llm = ChatMistralAI(model = "mistral-small-2506")

prompt = ChatPromptTemplate.from_messages(
    [("system", """you are a AI assistant .
      use only the provided context to answer the question.
      if the answer is not present in context,
      say : "i could not find the answer in the document"""),
     ("human", 
      """
      context  : {context}
      Question : {question}
      """)]
)

print("rag system createed")

print("press 0 to exit")

while True:
    query = input("you :")
    if query == 0:
        break
    
    docs= retriever.invoke(query)
    
    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )
    
    final_prompt = prompt.invoke({
        "context" : context,
        "question" : query
    })

    response = llm.invoke(final_prompt)
    
    print(f"\n AI : {response.content} ")