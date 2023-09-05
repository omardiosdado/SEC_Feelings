from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, OnlinePDFLoader
from langchain.document_loaders.base import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from sentence_transformers import SentenceTransformer
from streamlit_lottie import st_lottie_spinner
import os
import pinecone
import requests
import streamlit as st

favicon = 'https://polimata.ai/wp-content/uploads/2023/07/favicon-32x32-1.png'
st.set_page_config(
    page_title="SEC FeeLINGS",
    page_icon=favicon,
    initial_sidebar_state="expanded"
)

#hide_default_format = """
#       <style>
#       #MainMenu {visibility: hidden; }
#       footer {visibility: hidden;}
#       </style>
#       """
#st.markdown(hide_default_format, unsafe_allow_html=True)

st.sidebar.image("https://fr8technologies.com/wp-content/uploads/2023/02/Recurso-2@2x-1024x186-1.png", use_column_width=True)
st.title("SEC FeeLINGS 	:heart::chart_with_upwards_trend: ")
st.caption(':turtle: V1.01')
st.subheader('Audit Fr8Tech SEC Filing: 20-F')
st.write('	Annual and transition report of foreign private issuers [Sections 13 or 15(d)]')
st.divider()

st.sidebar.title(f'Sample Questions')
st.sidebar.write('what is Fr8Tech strategy?')
st.sidebar.write('What is the Fr8Tech capital structure')
st.sidebar.write('How can Fr8Tech become finantially healthy')
st.sidebar.write('What are the financial challenges for Fr8Tech')
st.sidebar.caption('Powered by Polímata.AI')




embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
pinecone.init(
    api_key=st.secrets['PINECONE_API_KEY'],  
    environment=st.secrets['PINECONE_ENV']  
)

index_name = "sec-feelings" # put in the name of your pinecone index here
docsearch = Pinecone.from_existing_index(index_name, embeddings)

llm=ChatOpenAI(model_name="gpt-4", temperature=0.0,openai_api_key=st.secrets["OPENAI_API_KEY"])
chain = load_qa_chain(llm, chain_type='stuff')




if 'click' not in st.session_state:
    st.session_state.click = False

def onClickFunction():
    st.session_state.click = True
    st.session_state.out1 = query

runButton = st.button('Enter',on_click=onClickFunction)


def load_lottieurl(url2: str):
    r = requests.get(url2)
    if r.status_code != 200:
        return None
    return r.json()

lottie_url_hello = "https://lottie.host/57b82a4f-04ed-47c1-9be6-d9bdf4a4edf0/whycX7qYPw.json"
lottie_url_download = "https://lottie.host/57b82a4f-04ed-47c1-9be6-d9bdf4a4edf0/whycX7qYPw.json"
lottie_hello = load_lottieurl(lottie_url_hello)
lottie_download = load_lottieurl(lottie_url_download)


if st.session_state.click:
    with st_lottie_spinner(lottie_download, key="download", height=200, width=300):
        response = (chain.run(input_documents=docs, question=query))
    st.subheader('Response:')
    st.info(response)
    
query=st.text_input('Ask question and press Enter:', key='pregunta')
docs=docsearch.similarity_search(query)
