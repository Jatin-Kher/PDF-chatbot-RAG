from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()


data = TextLoader("docuemnt loaders/notes.txt")

data = TextLoader("document loaders/notes.txt")

docs = data.load()

template = ChatPromptTemplate.from_messages(
    [("system", "you are a AI that summarizes thetext"),
     ("human", "{data}")]
)

model = ChatMistralAI(model = "mistral-small-2506")

prompt = template.format_messages(data = docs[0].page_content)

result = model.invoke(prompt)

print(result.content)