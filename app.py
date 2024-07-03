import bs4
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_cohere import ChatCohere
from langchain_community.document_loaders import UnstructuredURLLoader
import streamlit as st
from arabic_support import support_arabic_text



load_dotenv()


llm = ChatCohere(model="command-r")

urls = [

    "https://www.hrdf.org.sa/programs/individuals/training/professional-certificates/"
]
#loader = UnstructuredURLLoader(urls=urls)
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader(r"C:\Users\Dell\QA Agent\الشهادات المدعومة.pdf")

data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(data)
embedding=OpenAIEmbeddings()


vectorstore = Chroma.from_documents(documents=splits, embedding=embedding)
# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}) # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

template = """استخدم الامثلة التالية للاجابة عن السؤال في النهاية.
سيتم سؤالك هل يتم توفير دعم للشهادة, اذا لم تتوفر الشهادة يرجى الاجابةبـ لا, ومن ثم أجب ببديل مشابه لمجال الشهادة المطلوبة.
{context}

السؤال: {question}

الاجابة:"""

from langchain_core.prompts import PromptTemplate

custom_rag_prompt = PromptTemplate.from_template(template)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

def generate_response(message):
    return rag_chain.invoke(message)
support_arabic_text(components=["alert", "input", "markdown","textarea"])

# 5. Build an app with streamlit
def main():
    # Support Arabic text alignment in all components

    st.header("اسأل هدف عن الشهادات المهنية")
    message = st.text_area("اذكر اسم الشهادة:")

    if message:
        st.write("الاجابة ....")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
    
    
    
