#Practica 1: Normalizacion de texto
#Nombre: Tinoco Videgaray Sergio Ernesto
#Grupo: 5BV1
#Carrera: Ingenieria en Inteligencia Artificial
#Fecha de última modificación 29/09/23

from textblob import TextBlob
import pandas as pd
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import os

def get_unique_tokens(tokens):  #Obtener la lista de tokens unicos dentro del diccionario
 
    unique_tokens = []
 
    for x in tokens:
        if x not in unique_tokens:
            unique_tokens.append(x) #Agregar token unico a la lista
    return unique_tokens   

def histograma(dataframe):  #Visualizar el histograma del dataframe de tokens
    plt.figure(figsize=(15,5))
    plt.bar(dataframe["tokens"],dataframe["freq"])
    plt.xlabel("Tokens")
    plt.ylabel("Frecuencia")
    plt.title("Histograma")
    plt.show()    

def concatenar_lista(tokens):   #Concatenar lista de tokens para convertir en una cadena de texto
    #Esto con el fin de poder convertir la lista en un objeto de tipo TextBlob
    string_tokens=" ".join(tokens)
    return string_tokens

def remover_apostrofes(tokens): #Eliminar los apostrofes de la lista de tokens
    tokens_sin_apostorfes=[]
    apostrofes=['‘','’','“','”']

    for t in tokens:
        #Se agregan tokens que no esten en la lista de apostrofes
        if t not in apostrofes:
            tokens_sin_apostorfes.append(t) 
    return tokens_sin_apostorfes

def remover_acentos(tokens):    #Eliminar acentos de cada token
    tokens_sin_acentos=[]   #Lista vacia
    #Se genera una tabla de conversiones para remplazar cada caracter con tilde
    diccionario_conversion=str.maketrans({'á':'a','é':'e','í':'i','ó':'o'})

    for t in tokens:    #Se remplaza cada caracter con tilde en cada token y se agrega a la lista
        tokens_sin_acentos.append(str(t).translate(diccionario_conversion))
    return tokens_sin_acentos

def analizar_texto(texto):  #Realizar analisis exploratorio del texto

    tokens=texto.words  #Obtener tokens
    unique_tokens=get_unique_tokens(tokens) #Invocar funcion para obtener tokens unicos


    print("Total de tokens:",len(tokens))  #Numero de tokens totales


    print("Total de tokens unicos:",len(unique_tokens))  #Numero de tokens unicos

    #Obtener una lista con cada dupla que contiene al token y la cantidad de veces que aparece
    tokens_freq=list(zip(texto.word_counts.keys(),texto.word_counts.values()))  

    #Crear un dataframe con la lista de frecuencias
    df=pd.DataFrame(tokens_freq)
    df.columns=["tokens","freq"]    #Renombrar columnas
    #Cambiar los tipos de dato de cada columna
    df["tokens"]=df["tokens"].astype("string")
    df["freq"]=df["freq"].astype("int")

    print("10 tokens mas frecuentes")
    df_desc=df.sort_values(by=['freq'],ascending=False) #Ordenar dataframe descendentemente
    print(df_desc.head(10))

    print("10 tokens menos frecuentes")
    print(df_desc.tail(10))

   
    #Obtener muestra aleatoria del dataframe
    df_sample=df.sample(15,random_state=4)

    #Ordenar muestra de forma descendente
    df_sample=df_sample.sort_values(by="freq",ascending=False)

    #Generar histograma a partir de la muestra aleatoria
    histograma(df_sample)
    return tokens

def normalizar_texto(all_tokens,documento): #Funcion para la normalizacion del texto

    #Remover stopwords
    #Se crea la lista o conjunto de stopwords en el idioma correspondiente
    if(documento=="doc1"):
        #Se crea la lista o conjunto de stopwords en español
        stop_words=stopwords.words("spanish")
    else:
        #Se crea la lista o conjunto de stopwords en ingles
        stop_words=stopwords.words("english")


    #Remover stopwords del diccionario
    print("Tamaño del diciconario antes de remover stop words:",len(all_tokens))
    tokens = []
    for t in all_tokens:
        if t not in stop_words: #Agregar token que no este en la lista de stopwords
            tokens.append(t)
    print("Tamaño del diciconario despues de remover stop words:",len(tokens))

    #Remover apostrofes del diccionario
    print("Tamaño del diciconario antes de remover apostrofes:",len(tokens))
    tokens=remover_apostrofes(tokens)
    print("Tamaño del diciconario despues de remover apostrofes:",len(tokens))

    #Remover acentos del dicionario
    tokens=remover_acentos(tokens)

    #Converitr lista de tokens en objeto blob
    tokens=TextBlob(concatenar_lista(list(tokens)))

    #POS tagging Etiquetado
    tokens_tagged=tokens[:]
    tokens_tagged=tokens_tagged.sentences[0].tags
    print("Etiquetas asignadas por el POS tagging\n",tokens_tagged)

    tokens=tokens.words #Obtener tokens

    #Lematizar
    tokens_lematizados=tokens.lemmatize()

    #Stemming
    tokens_stemming=tokens.stem()

    print("\nComparacion token|lematizacion|stemming:\n")
    for t,l,s in zip(list(tokens),list(tokens_lematizados),list(tokens_stemming)):
        print(f"{t}   |   {l}  |   {s}")
    
    return TextBlob(concatenar_lista(list(tokens)))

#Cargar el corpus usando la ruta relativa a la carpeta documentos
corpus = os.listdir("./Documentos")
corpus = [os.path.join("./Documentos", f) for f in corpus] #Concatenar directorio con el nombre de cada documento

#Recorrer corpus
for doc in corpus:
    print("\nProcesando archivo",doc)
    texto=open(doc,"r", encoding="utf8").read() #Abrir documento

    #Instancia de la clase TextBlob
    doc_blob=TextBlob(texto)
    
    #Se invoca la funcion para el analisis exploratorio
    all_tokens=analizar_texto(doc_blob)
    #Se invoca la funcion para la normalizacion del texto pasando como parametro los tokens y el nombre del documento
    tokens=normalizar_texto(all_tokens,doc[-8:-4])
    #Se vuelve a realizar un analisis exploratorio despues de la normalizacion
    tokens=analizar_texto(tokens)