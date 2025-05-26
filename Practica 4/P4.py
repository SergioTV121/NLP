#Practica 4: 
#Nombre: Tinoco Videgaray Sergio Ernesto
#Grupo: 5BV1
#Carrera: Ingenieria en Inteligencia Artificial
#Fecha de última modificación 13/12/23

#Importacion de paquetes
import nltk
import io
import os
import re
from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.corpus import wordnet as wn
import spacy
from keybert import KeyBERT
import numpy as np
from scipy import spatial
from torch.nn.functional import cosine_similarity
import torch
from transformers import BertModel, BertTokenizer
#Diccionario de embeddings
embeddings_dict = {}

def segmentar_texto(doc,category):
    #Tokenizar
    sent_text = nltk.sent_tokenize(doc)
    word_freq={}
    for sentence in sent_text:
        tokenized_text = nltk.word_tokenize(sentence)           
        tagged = nltk.pos_tag(tokenized_text)   #Etiquetar de acuerdo a la categoria dada
        for word,tag in tagged:
            if(tag[0]==category):
                if (word not in list(STOP_WORDS) and word not in punctuation and len(word)>1):
                    if(word not in word_freq.keys()):   #Incrementar frecuencias
                        word_freq[word]=1
                    else:
                        word_freq[word]+=1
    #Devolver el valor mas alto            
    common_word=max(word_freq,key=word_freq.get)
    return common_word,list(word_freq.keys())

def similar_words_synsets(common_word,words):
    #Obtener synsets de la palabra mas comun
    synset_cw=wn.synsets(common_word)[0]

    #Terminos similares dentro del documento
    #Wup similarity
    wup_scores={}

    for word in words:
        synsets_w=wn.synsets(word)
        try:    #Calcular la similitud usando WuPalmer usando el segundo synset
            wup_scores[word]=synset_cw.wup_similarity(synsets_w[1])
        except:
            try:#Utilizar el primero en caso de no tener mas synsets
                wup_scores[word]=synset_cw.wup_similarity(synsets_w[0])
            except:#Continuar al siguiente termino en caso de no haber ninguno
                continue
    #Ordenar diccionario de acuerdo a los puntajes calculados de mayor a menor
    wup_similar = sorted(wup_scores.items(), key=lambda x:x[1],reverse=True)[-5:]
    
    #Path similarity
    path_scores={}
    for word in words:
        synsets_w=wn.synsets(word)  #Obtener lista de synsets
        try:#Seleccionar el segundo ya que suele ser el mas parecido
            path_scores[word]=synset_cw.path_similarity(synsets_w[1])
        except:
            try:#Seleccionar el primer synset en caso de no haber segundo
                path_scores[word]=synset_cw.path_similarity(synsets_w[0])
            except:#Continuar al siguiente termino en caso de no haber ninguno
                continue
    #Ordenar diccionario de acuerdo a los puntajes calculados de mayor a menor
    path_similar = sorted(path_scores.items(), key=lambda x:x[1],reverse=True)[-5:]
    return wup_similar,path_similar

def kw_bert(doc):  #Extraer keywords del documento usando bert
    kw_model = KeyBERT()    #Instancia de la clase KeyBert
    #Obtener la mejor frase usando bi-gramas
    keywords=kw_model.extract_keywords(docs=doc, keyphrase_ngram_range=(2,2),top_n=1)
    return keywords[0][0]

def similarity_score(s1, s2):   #Calcular similitud usando path
    s =[]
    for i1 in s1:
        r = []
        scores = [x for x in [i1.path_similarity(i2) for i2 in s2] if x is not None]
        if scores:
            s.append(max(scores))   #Agregar el mejor puntaje
    return sum(s)/len(s)    #Normalizar puntajes

def convert_tag(tag):
    #Convertir etiqueta en el formato de WordNet
     
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None

def doc_to_synsets(doc):    #Convertir documento a synsets
    tokens = nltk.word_tokenize(doc)
    pos = nltk.pos_tag(tokens)
    tags = [tag[1] for tag in pos]
    wntag = [convert_tag(tag) for tag in tags]  #Convertir tag a WordNet
    ans = list(zip(tokens,wntag))
    sets = [wn.synsets(x,y) for x,y in ans]
    final = [val[0] for val in sets if len(val) > 0]
     
    return final

def similar_docs_synset(doc1,doc2):#Obtener la similitud enter documentos usando synsets
    #Convertir documentos a synsets
    synsets1 = doc_to_synsets(doc1) 
    synsets2 = doc_to_synsets(doc2)
    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2

def crear_embeddings_glov():
    #Cargar glov con 200 dimensiones desde el archivo
    with open("glov embeddings/glove.6B.200d.txt", 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()   #Separar por espacios
            word = values[0]    #Obtener palabras
            vector = np.asarray(values[1:], "float32") #Obtener vector del embedding
            embeddings_dict[word] = vector  #Agregar embedding al diccionario
    f.close()   #Cerrar archivo

def similar_words_embeddings(common_word,doc_words):
    #Obtener la similitud usando embeddings
    common_word_embedding=embeddings_dict[common_word]
    doc_embedding_dic={}
    for word in doc_words:  #Obtener las palabras del documento
        if word in embeddings_dict.keys():
            doc_embedding_dic[word]=spatial.distance.euclidean(embeddings_dict[word], common_word_embedding)
        else:
            continue    
    #Obtener los 5 mejores embeddings de acuerdo a la menor distancia euclideana
    return sorted(doc_embedding_dic.items(), key=lambda x:x[1])[1:6]

def cargar_corpus():
    #Cargar archivos de los libros dentro de la carpeta corpus
    lista_libros = os.listdir("./corpus/")
    corpus={}
    for libro in lista_libros:
        #Extraer nombre del archivo sin la extension
        corpus[libro[:-4]]=io.open("./corpus/"+libro,mode="r",encoding="utf-8").read()
    return corpus

def similar_docs_embeddings(phrase1,phrase2):   #Similitud de documentos con BERT
    #Cargar modelo BERT y tokenizador
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    #Tokenizar frases
    encoded_input1=tokenizer(phrase1,return_tensors="pt")
    encoded_input2=tokenizer(phrase2,return_tensors="pt")
    #Obtener embeddings por medio de los tensores
    embed1 = model(**encoded_input1)
    embed2 = model(**encoded_input2)

    #Caluclar similitud coseno de los embeddings
    similarity_tensor=cosine_similarity(embed1[0],embed2[0])
    return torch.mean(similarity_tensor,1).item() #Deveulve el promedio de la similitud para un tensor n dimensional

#Cargar documentos del corpus
corpus=cargar_corpus()

#Crear embeddings de glov
crear_embeddings_glov()

#Variables principales
frase_clave_libro1=""
docs_similarity={}
primer_libro=True

i=1
for titulo,documento in corpus.items():
    #Extraer primer capitulo de cada libro usando RegEx
    doc=documento[re.search("(Chapter|CHAPTER) (One|one|1|i|I)",documento).end()+1:
                  re.search("(Chapter|CHAPTER) (Two|two|2|ii|II)",documento).start()-9]

    print("\nLibro",i)
    #Similitud de palabras con synsets
    #Verbos similares
    common_verb,words=segmentar_texto(doc,'V')
    print(f"Verbo mas comun:{common_verb}")
    
    wup_sim,path_sim=similar_words_synsets(common_verb,words)
    print(f"Verbos similares a {common_verb} dentro del documento :\
          \nWup similarity:{wup_sim}\nPath similarity:{path_sim}\
          \nEmbeddings:{similar_words_embeddings(common_verb,words)}\n")

    #Sustantivos similares
    common_noun,words=segmentar_texto(doc,'N')
    print(f"Sustantivo mas comun:{common_noun}")    
    wup_sim,path_sim=similar_words_synsets(common_noun,words)
    print(f"Sustantivos similares a {common_noun} dentro del documento:\
          \nWup similarity:{wup_sim}\nPath similarity:{path_sim}\n")
    
    #Similitud de documentos
    frase_clave=kw_bert(doc)
    print(f"Frase clave del libro {i}:{frase_clave}")
    
    if primer_libro:
        primer_libro=False
        frase_clave_libro1=frase_clave
    else:
        print(f"Similitud con {frase_clave_libro1}\nSynsets:{similar_docs_synset(frase_clave_libro1,frase_clave)}\
        \nEmbeddings:{similar_docs_embeddings(frase_clave_libro1,frase_clave)}")

    

    i+=1
