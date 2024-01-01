import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CustomHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, documents):
        texts = [doc.text.replace("\n", " ") if hasattr(doc, 'text') else str(doc) for doc in documents]
        return super().embed_documents(texts)

def generate_embeddings(texts, llm):
    return [llm.get_embedding(text) for text in texts]

def generate_response(uploaded_file, query_text):
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]
        text_splitter = CharacterTextSplitter()
        texts = text_splitter.create_documents(documents)
        embeddings = CustomHuggingFaceEmbeddings(model_name="oshizo/sbert-jsnli-luke-japanese-base-lite")
        llm = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 1})
        qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
        responses = []
        for text in texts:
            responses.append(qa.run(text))
        return qa.run(query_text)

st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

uploaded_file = st.file_uploader('Upload an article', type='txt')
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.', disabled=not uploaded_file)

result = []
with st.form('myform', clear_on_submit=True):
    submitted = st.form_submit_button('Submit', disabled=not (uploaded_file and query_text))
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, query_text)
            result.append(response)

if len(result):
    st.info(result[0])
