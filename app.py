import streamlit as st


from langchain_mistralai import ChatMistralAI


from langchain_ollama import ChatOllama
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate

import os
import getpass
#-----------------------------------------------------------------------------------------------------------


# Титульная страница
def title_page():
    st.title('Приложение по проектной деятельности: рекомендательная система с использованием RAG подхода')
    st.subheader('Выполнили: Мягков Е. В, Перепелкин М. А, Миронов Д. Г.')


# Страница со стеком технологий
def stack_page():
    st.title('Стек технологий')
    col1, col2 = st.columns(2, gap='large')

    with col1:
        st.image('logo/langchain.jpg')

    with col2:
        st.image('logo/ollama.jpg')





def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Интерактив с моделью
def query_page():
    st.title("🦜🔗 Ask model about games to play!")

    mistral_api_key = st.sidebar.text_input("MistralAI API Key", type="password")

    os.environ["MISTRAL_API_KEY"] = mistral_api_key



    
    model = ChatOllama(model="gemma2")

    vectorstore = FAISS.load_local('faiss_storage_1500_300',
                                embeddings=MistralAIEmbeddings(model="mistral-embed"),
                                allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever( search_type = 'similarity_score_threshold',
                                            search_kwargs={'k': 20,
                                                        'score_threshold': .48})

    prompt = change_prompt()



    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
        
    )


    with st.form("form"):
        text = st.text_area(
        "Enter text:",
        "Write your question here",
        )
        submitted = st.form_submit_button("Submit")
        resp = ""
        if len(mistral_api_key) == 0:
            st.warning("Please enter your MistralAI API key!", icon="⚠")
        if submitted and len(mistral_api_key) != 0:
            while(resp==""):
                try:
                    resp = rag_chain.invoke(text)
                except:
                    pass
            st.write(resp)





def change_prompt():
    template = st.text_input('Paste your own template (optional)')
    prompt = PromptTemplate(input_variables=['context', 'question'], input_types={}, partial_variables={}, template='', template_format='mustache')

    if (len(template) != 0):
        prompt.template = template
    else:
        template = open("default template.txt", "r").read()
        prompt.template = template

    return prompt





pg = st.navigation([st.Page(title_page), st.Page(stack_page), st.Page(query_page)])
pg.run()