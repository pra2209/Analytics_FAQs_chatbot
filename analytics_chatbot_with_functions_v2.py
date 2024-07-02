import json
import tiktoken

import pandas as pd
import openai
from openai import AzureOpenAI

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.document_loaders import PyPDFDirectoryLoader

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings)
import streamlit as st
from streamlit_chat import message

st.set_page_config(page_title = "Chatbot for Analytics FAQs", layout="wide")

client = AzureOpenAI(
    azure_endpoint="https://pranav-kumar-singh.openai.azure.com/",
    api_key="619e4ee0a33646baba842542652f29ef",
    api_version="2024-02-01"
)
deployment_name = "gpt-35-turbo"

def get_pdf_text_and_create_vectordb():
    pdf_folder_location = "."
    pdf_loader = PyPDFDirectoryLoader(pdf_folder_location)
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name='cl100k_base',
    chunk_size=512,
    chunk_overlap=16)
    Analytics_FAQs_chunks_ada = pdf_loader.load_and_split(text_splitter)
    
    Analytics_FAQs_collection = 'Analytics_FAQs'
    embedding_model = SentenceTransformerEmbeddings(model_name='thenlper/gte-large')
    vectorstore = Chroma.from_documents(Analytics_FAQs_chunks_ada,
                                        embedding_model,
                                        collection_name=Analytics_FAQs_collection,
                                        persist_directory='.')
    vectorstore.persist()
    vectorstore_persisted = Chroma(collection_name=Analytics_FAQs_collection,
                                   persist_directory='.',
                                   embedding_function=embedding_model)
    retriever = vectorstore_persisted.as_retriever(search_type='similarity',
                                                   search_kwargs={'k': 5})
    return retriever

def get_input_and_call_api(user_input, retriever):
    qna_system_message = """
    You are an assistant to a potential client of Freshworks Analytics who answers user queries on Analytics product features.
    User input will have the context required by you to answer user questions.
    This context will begin with the token: ###Context.
    The context contains references to specific portions of a document relevant to the user query.
    
    User questions will begin with the token: ###Question.
    Please answer user questions only using the context provided in the input.
    Do not mention anything about the context in your final answer. Your response should only contain the answer to the question.
    
    If the answer is not found in the context, respond "I don't know".
    """
    qna_user_message_template = """
    ###Context
    Here are some documents that are relevant to the question mentioned below.
    {context}
    
    ###Question
    {question}
    """
    relevant_document_chunks = retriever.get_relevant_documents(user_input)
    for document in relevant_document_chunks:
        print(document.page_content.replace("\t", " "))
        break
    
    context_list = [d.page_content for d in relevant_document_chunks]
    context_for_query = ". ".join(context_list)
    prompt = [
      {'role':'system', 'content': qna_system_message},
      {'role': 'user', 'content': qna_user_message_template.format(
         context=context_for_query,
         question=user_input
        )
      }
    ]

    try:
      response = client.chat.completions.create(
        model=deployment_name,
        messages=prompt,
        temperature=0
      )

      prediction = response.choices[0].message.content.strip()
    except Exception as e:
      prediction = f'Sorry, I encountered the following error: \n {e}'

    return prediction


def main():
    retriever = get_pdf_text_and_create_vectordb()
    st.title("ChatBot for Analytics FAQs")
#    if 'user_input' not in st.session_state:
#        st.session_state['user_input'] = []

#    if 'openai_response' not in st.session_state:
#        st.session_state['openai_response'] = []
        
    def get_text():
        input_text = st.text_input("write here", key="input")
        return input_text
        
    user_input = get_text()
    
    if user_input:
        output = get_input_and_call_api(user_input,retriever)
        output = output.lstrip("\n")
        st.write("Reply: ", output)
#        st.session_state.openai_response.append(user_input)
#        st.session_state.user_input.append(output)
    
#    message_history = st.empty()
    
#    if st.session_state['user_input']:
#       for i in range(len(st.session_state['user_input']) - 1, -1, -1):
#            message(st.session_state["user_input"][i],key=str(i),avatar_style="icons")
#            # This function displays OpenAI response
#            message(st.session_state['openai_response'][i],
#                    vatar_style="miniavs",is_user=True,
#                    key=str(i) + 'data_by_user')

if __name__ == "__main__":
    main()
