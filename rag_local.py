from langchain_ollama import ChatOllama
import os
import getpass
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


os.environ['HF_TOKEN'] = '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'
os.environ["MISTRAL_API_KEY"] = 'O4uISBokqds8Bt6xXMrYIvEMntNHA68K'
os.environ['LANGCHAIN_API_KEY'] = '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'

def get_llm_resp(q:str):
    model = ChatOllama(model="gemma2")
    vectorstore = FAISS.load_local('../faiss_storage_1500_300/',
                                embeddings=MistralAIEmbeddings(model="mistral-embed"),
                                allow_dangerous_deserialization=True)


    retriever = vectorstore.as_retriever( search_type = 'similarity_score_threshold',
                                            search_kwargs={'k': 20,
                                                        'score_threshold': .48})
    prompt = hub.pull('gemma2_prompt',api_key=os.environ['LANGCHAIN_API_KEY'])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
        
    )

    return rag_chain.invoke(q)