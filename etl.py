#Librerías
import os
import re
import emoji
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import configuracion

openai_api_key = configuracion.OPENAI_API_KEY
persist_directory = os.path.abspath("./NLP_Vectores")

#Funcion Limpieza
def preprocess(text:str) -> str:
    text = re.sub(r"<[^>]*>", "", text)         #Eliminar etiquetas HTML
    text = re.sub(r"http\S+|www.\S+", "", text) #Elimina URLs
    text = re.sub(r"Copyright.*", "", text)     #Elimina copyrights
    text = text.replace("\n", " ")              #Reemplaza Saltos de linea
    text = emoji.demojize(text)                 #Convierte emojis a texto
    text = re.sub(r":[a-z_&+-]+:", "", text)    #Elimina representaciones de emojis en texto
    text = re.sub(r"[^A-Za-z0-9\s.,!?;:'\"()]+áéíóúÁÉÍÓÚ", "", text)  # Mantiene letras, números, espacios y ciertos signos de puntuación

    return text

#carga, limpieza y embedding
def crear_embeddings():
    #Carga de data
    ruta_descarga = "D:\Jupyter\Proyecto NLP\PDF´s"
    loader = DirectoryLoader(path=ruta_descarga,loader_cls=PyMuPDFLoader)
    data = loader.load()

    #Agregar informacíon extra al content
    for doc in data:
        title = doc.metadata['title']
        page = doc.metadata['page']
        total_pages = doc.metadata['total_pages']
        resumen = f'Titulo PDF: {title}, pagina: {page}, total paginas: {total_pages}'
        doc.page_content += resumen
    
    #Limpiamos la data
    for doc in data:
        doc.page_content = preprocess(doc.page_content)
    
    #Cortamos la data
    splitter   = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80, length_function=len)
    data_split = []

    for doc in data:
        if len(doc.page_content) > 1000:
            doc_split =splitter.split_text(doc.page_content)
            for split in doc_split:
                new_doc = Document(
                    page_content=split,
                    metadata = doc.metadata
                )
                data_split.append(new_doc)
        else:
            data_split.append(Document(
                page_content=doc.page_content,
                metadata = doc.metadata
            ))
    #Creamos los vectores

    vectorstore = Chroma.from_documents(
        documents=data_split,
        embedding=OpenAIEmbeddings(api_key=openai_api_key,
                                   model='text-embedding-3-large'),
        persist_directory=persist_directory
    )
    return vectorstore
