# librerías
import os
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import VectorStoreRetrieverMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import configuracion

#OpenAI Key
openai_api_key = configuracion.OPENAI_API_KEY 

#BD Vectores y Retriever
ruta_vectorstore = os.path.abspath('NLP_vectores')

vector_db = Chroma(
    persist_directory=ruta_vectorstore,
    embedding_function=OpenAIEmbeddings(api_key=openai_api_key,
                                        model='text-embedding-3-large'
    )
)

retriever = vector_db.as_retriever(search_type='mmr')
memory = VectorStoreRetrieverMemory(retriever=retriever)

#Configurar Streamlit

st.title('Asistente Estudiantil UCM')
st.write('Este asistente responde preguntas relacionadas con el curso de NLP dictada\
          para la maestría de Ciencia de Datos de la Universidad Complutense de Madrid')

prompt_template = PromptTemplate(
    input_variables=['input', 'contexto'],
    template="""
Eres un asistente estudiantil de la Universidad Complutense de Madrid para la maestría de Ciencia de Datos y devuelves información sobre NLP.
Respondes preguntas {pregunta} de los usuarios  relacionados a NLP basandote en el contexto proporcionado y en
la información que puedas tener en tu modelo de entrenamiento OpenAi para exclusivamente complementar siempre y cuando te bases en el contexto proporcionado.
No hagas suposiciones.
Por favor devuelveme una respuestas informativas y detalladas y procura no equivocarte.
Evita que tus respuestas comiencen con En el contexto de... dame directamente la respuesta""")


pregunta = st.text_area('Haz tu pregunta')

if st.button('Enviar'):
    if pregunta:
        resultados_similares = vector_db.similarity_search(pregunta, k=3)
        
        contexto=''.join([doc.page_content for doc in resultados_similares])

        preguntaContexto = f'{pregunta}\nContexto: {contexto}.'
        
        llm = ChatOpenAI(model='gpt-4o', api_key=openai_api_key, temperature=0.3, max_tokens=1024)
        qa_chain = LLMChain(llm=llm, prompt=prompt_template, memory=memory)
        respuesta = qa_chain.invoke({'pregunta':preguntaContexto})
        resultado = respuesta['text']
        st.write(resultado)
    
    else: 
        st.write('Lo siento no encontré información a tu pregunta')
    

    