import streamlit as st
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import bs4
import fitz
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import time
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

def load_pdfs_from_folder(folder):
    all_docs = []
    if not os.path.exists(folder):
        return all_docs
    pdf_files = [f for f in os.listdir(folder) if f.endswith(".pdf")]
    for pdf_file in pdf_files:
        path = os.path.join(folder, pdf_file)
        doc = fitz.open(path)
        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                all_docs.append(Document(
                    page_content=text,
                    metadata={"source": pdf_file, "page": page_num + 1}
                ))
    return all_docs

def load_website(url):
    loader = WebBaseLoader(
        web_paths=[url],
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                ["p", "table", "tr", "td", "th", "h1", "h2", "h3", "li","code", "pre"]
            )
        }
    )
    docs = loader.load()
    return [doc for doc in docs if doc.page_content.strip()]

if "vector_store" not in st.session_state:
    with st.spinner("Loading and indexing website..."):

        urls = [
            "https://www.django-rest-framework.org/api-guide/requests/",
            "https://www.django-rest-framework.org/api-guide/responses/",
            "https://www.django-rest-framework.org/api-guide/views/",
            "https://www.django-rest-framework.org/api-guide/serializers/",
            "https://www.django-rest-framework.org/api-guide/authentication/",
            "https://www.django-rest-framework.org/api-guide/permissions/",
            "https://www.django-rest-framework.org/api-guide/throttling/",
            "https://www.django-rest-framework.org/api-guide/filtering/",
            "https://www.django-rest-framework.org/api-guide/pagination/",
            "https://www.django-rest-framework.org/api-guide/routers/",
            "https://www.django-rest-framework.org/api-guide/parsers/",
            "https://www.django-rest-framework.org/api-guide/generic-views/",
            "https://www.django-rest-framework.org/api-guide/renderers/",
            "https://www.django-rest-framework.org/api-guide/fields/",
            "https://www.django-rest-framework.org/api-guide/relations/",
            "https://www.django-rest-framework.org/api-guide/validators/",
            "https://www.django-rest-framework.org/api-guide/caching/",
            "https://www.django-rest-framework.org/api-guide/versioning/",
            "https://www.django-rest-framework.org/api-guide/content-negotiation/",
            "https://www.django-rest-framework.org/api-guide/metadata/",
            "https://www.django-rest-framework.org/api-guide/schemas/",
            "https://www.django-rest-framework.org/api-guide/format-suffixes/",
            "https://www.django-rest-framework.org/api-guide/reverse/",
            "https://www.django-rest-framework.org/api-guide/exceptions/",
            "https://www.django-rest-framework.org/api-guide/status-codes/",
            "https://www.django-rest-framework.org/api-guide/testing/",
            "https://www.django-rest-framework.org/api-guide/settings/",
            "https://www.django-rest-framework.org/topics/documenting-your-api/",
            "https://www.django-rest-framework.org/topics/internationalization/",
            "https://www.django-rest-framework.org/topics/ajax-csrf-cors/",
            "https://www.django-rest-framework.org/topics/html-and-forms/",
            "https://www.django-rest-framework.org/topics/browser-enhancements/",
            "https://www.django-rest-framework.org/topics/browsable-api/",
            "https://www.django-rest-framework.org/topics/rest-hypermedia-hateoas/"
        ]

        all_docs = []
        for url in urls:
            all_docs.extend(load_website(url))

        pdf_docs = load_pdfs_from_folder("data")    
        all_docs.extend(pdf_docs)
            
        if not all_docs:
            st.error("❌ No content extracted — site may be behind a login or JavaScript-rendered")
            st.stop()

        st.success(f"✅ Indexed {len(all_docs)} document(s)")

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = splitter.split_documents(all_docs)

        if not final_documents:
            st.error("❌ Text splitting produced no chunks")
            st.stop()

        st.session_state.vector_store = FAISS.from_documents(final_documents, embeddings)
        st.session_state.embeddings = embeddings

st.title("Django REST Framework RAG Demo")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """Answer only based on the context below. If the answer is not contained within the context, say you don't know. Always use all available context to provide the best answer possible. Answer only questions related to django rest framework. Do not attempt to answer questions about other topics. If the question is not about django rest framework, say you don't know.
<context>
{context}
</context>"""),
    ("human", "{input}")
])

retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})

chain = (
    {"context": retriever, "input": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

user_prompt = st.text_input("Ask a question about Django REST Framework:")

if user_prompt:
    start = time.process_time()
    response = chain.invoke(user_prompt)
    st.write(response)
    st.caption(f"Response time: {time.process_time() - start:.2f}s")