#Practica 2: Expresiones regulares y Vectorizacion de texto
#Nombre: Tinoco Videgaray Sergio Ernesto
#Grupo: 5BV1
#Carrera: Ingenieria en Inteligencia Artificial
#Fecha de última modificación 09/10/23
#Programa 2: Vectorizacion de texto

#Importar paquetes
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords,wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag,PerceptronTagger   #Etiquetado inteligente
from math import log10

def pos_tagger(nltk_tag):   #Funcion para ajustar etiquetas del POS-tagging al formato wordnet
    if nltk_tag.startswith('J'):    #Adjetivo
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):  #Verbo
        return wordnet.VERB
    elif nltk_tag.startswith('N'):  #Sustantivo
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):  #Adverbio
        return wordnet.ADV
    else:          
        return None

def normalizar(doc,n):  #Funcion para normalizar cada documento

    print("\nNormalizacion del documento",n)

    print("\nDocumento inicial:\n",doc)

    #Convertir a minuscula
    doc=doc.lower()

    print("\nDocumento en minusculas:\n",doc)

    #Tokenizar
    tokens=word_tokenize(doc)
    print("\nTokens iniciales:\n",tokens)

    #Remover stop-words
    tokens=[token for token in tokens if token not in stopwords.words("english")]
    print("\nTokens despues de remover stop-words:\n",tokens)

    #Remover puntos y comas
    tokens=[token for token in tokens if (token != "," and token!=".") ]
    print("\nTokens despues de remover puntos y comas:\n",tokens)

    #Aplicar stemming
    ps=PorterStemmer()
    print("\nStemming\nOriginal       |        Stemming")
       
    for token in tokens:
       print(f"{token}     |       {ps.stem(token)}")
    
    tokens=[ps.stem(token) for token in tokens]
    print("\nTokens despues de aplciar el stemming:\n",tokens)

    #Aplicar POS-tagging
    tagger=PerceptronTagger() #Usar un modelo pre-entrenado
    #tokens_tagged=pos_tag(tokens)  #Usando metodo pos tag 
    tagger.train([[("assess","VBP"),("tube","NN")]])    #Reentrar modelo
    tokens_tagged=tagger.tag(tokens)    #Aplicar etiquetado usando el modelo
    #Reasignar etiquetas con formato wordnet
    tokens_tagged= list(map(lambda x: (x[0], pos_tagger(x[1])), tokens_tagged))

    print("\nTokens con el POS-tagging:\n",tokens_tagged)
    
    #Lematizar
    lematizer=WordNetLemmatizer()   #Instancia del WordNetLemmatizer
    #Aplicar lematizacion utilizando las etiquetas asignadas en el POS-tagging
    tokens=[lematizer.lemmatize(token,tag) for token,tag in tokens_tagged]
    print("\nTokens despues de lematizar:\n",tokens)
    return tokens

def term_presence(dic,corpus): #Funcion que genera los vectores de hot coding
    vector_doc=[]
    for doc in corpus:  #Por cada documento en el corpus
        #Agregar al vector la lista de valores binarios en funcion de la presencia del termino en cada documento
        vector_doc.append(list(map(lambda x:1 if x in doc else 0,dic)))
    return vector_doc

def term_count(dic,corpus): #Funcion que genera los vectores de term count
    vector_doc=[]
    for doc in corpus: #Por cada documento en el corpus
        #Agregar al vector la lista con el numero de veces que aparece cada termino en el documento
        vector_doc.append(list(map(lambda x: doc.count(x),dic)))
    return vector_doc

def probabilidad(vectores_cantidades,corpus): #Funcion que genera los vectores de probabilidades
    total_terminos=len(corpus)
    #Se obtiene un vector del total de veces que aparece cada termino en todo el corpus
    vector_suma_cantidades=[sum(columna) for columna in zip(*vectores_cantidades)]
    vector_doc=[]
    #Se agrega una lista con la cantidad de veces que aparece el termino 
    #en todo el corpus dividido entre el total de terminos en el corpus
    vector_doc.append(list(map(lambda x: x/total_terminos,vector_suma_cantidades)))
    return vector_doc

def TF(vectores_cantidades): #Funcion para generar los vectores de term frecuency
    vector_doc=[]
    for vector in vectores_cantidades: #Por cada vector en la lista de vectores de cantidades
        #Agregar la lista con las cantidades que aparece el termino en el documento 
        #dividido entre el total de terminos en el documento
        vector_doc.append(list(map(lambda x: x/len(vector),vector)))
    return vector_doc

def IDF(num_docs,vectores_term_presence): #Funcion para generar el vector de IDF
    vector_doc=[]
    #Se obtiene un vector de cantidades sumando el total de documentos en los que
    #aparece cada termino en todo el corpus utilizando las columnas del vector del hot encoding
    vector_ocurrencias=[sum(columna) for columna in zip(*vectores_term_presence)]
    for ocurrencias in vector_ocurrencias:  #Por cada valor del vector de ocurrencias
        #Agregar el valor del IDF correspondiente a cada termino
        vector_doc.append(log10(num_docs/ocurrencias))
    return vector_doc

def TF_IDF(vectores_tf,vector_idf): #Funcion para generar los vectores TF-IDF
    vector_doc=[]
    for tf in vectores_tf: #Por cada vector en los vectores de TF
        #Agregar el producto de cada TF con el valor de su IDF
        vector_doc.append([tf[i]*vector_idf[i] for i in range(len(tf))])
    return vector_doc

#Cargar documentos de texto
doc1=open("documentos_vectorizacion/doc1.txt").read()
doc2=open("documentos_vectorizacion/doc2.txt").read()
doc3=open("documentos_vectorizacion/doc3.txt").read()

#Normalizar documentos
doc1_n=normalizar(doc1,1)
doc2_n=normalizar(doc2,2)
doc3_n=normalizar(doc3,3)

#Crear diccionario
vector_corpus=doc1_n+doc2_n+doc3_n
#print(vector_corpus)

#Diccionario usando valores unicos
diccionario=[]
for token in vector_corpus:
    if token not in diccionario:
        diccionario.append(token)

print("Diciconario\n",diccionario)
corpus=[doc1_n,doc2_n,doc3_n]
#print(corpus)

#Vectorizar texto
vectores_term_presence=term_presence(diccionario,corpus)
vectores_term_count=term_count(diccionario,corpus)
vector_probabilidad=probabilidad(vectores_term_count,vector_corpus)
vectores_term_freq=TF(vectores_term_count)
vectores_inv_doc_freq=IDF(len(corpus),vectores_term_presence)
vectores_tf_idf=TF_IDF(vectores_term_freq,vectores_inv_doc_freq)


print("\nVectores de Term Presence")
for i in range(len(vectores_term_presence)):
    print(f"Documento {i+1}\n {vectores_term_presence[i]}")

print("\nVectores de Term Count")
for i in range(len(vectores_term_count)):
    print(f"Documento {i+1}\n {vectores_term_count[i]}")    

print("\nVector de Probabilidad\n",vector_probabilidad)      

print("\nVectores de TF")
for i in range(len(vectores_term_freq)):
    print(f"Documento {i+1}\n {vectores_term_freq[i]}")    

print("\nVector de IDF\n",vectores_inv_doc_freq)    

print("\nVectores de TF-IDF")
for i in range(len(vectores_tf_idf)):
    print(f"Documento {i+1}\n {vectores_tf_idf[i]}")  
