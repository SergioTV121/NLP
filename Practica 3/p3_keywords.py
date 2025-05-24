#Practica 3: 
#Nombre: Tinoco Videgaray Sergio Ernesto
#Grupo: 5BV1
#Carrera: Ingenieria en Inteligencia Artificial
#Fecha de última modificación 01/12/23
#Programa 1: Extraccion de KeyWords

import io
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keybert import KeyBERT
from keyphrase_vectorizers import KeyphraseCountVectorizer
from rake_nltk import Rake
import spacy
import pytextrank
import re

def tf_idf(doc): 

    #Normalizacion del documento

    #Eliminacion de signos de puntuacion
    doc = doc.lower().replace(".", "").replace(",","").replace(")","").replace("(","").replace("]","").replace("[","")
    
    #Tokenizar
    tokens = word_tokenize(doc) 

    #Remover stopwords
    tokens=[token for token in tokens if token not in stopwords.words("english")]
       
    #Utilizar valores de TF_IDF para asignar un puntaje a los tokens
    from sklearn.feature_extraction.text import TfidfVectorizer 
    vectorizer = TfidfVectorizer() 
    tfidf = vectorizer.fit_transform(tokens) 

    #Obtener las 5 palabras clave con mejor puntaje
    keywords = sorted(vectorizer.vocabulary_, key=lambda x: tfidf[0, vectorizer.vocabulary_[x]], reverse=True)[:5] 

    return keywords

def bert(doc):
    kw_model = KeyBERT()    #Instancia de la clase KeyBert

    #Obtener las 5 mejores keywords usando Vectorizer
    keywords=kw_model.extract_keywords(docs=doc, vectorizer=KeyphraseCountVectorizer(),top_n=5)
    return keywords

def rake(doc):
    rake_nltk_var = Rake(stopwords.words("english"))  #Instancia de la clase Rake

    #Metodo para extraer las oraciones clave usando el objeto rake
    rake_nltk_var.extract_keywords_from_text(doc) 
    #Se obtienen las 5 oraciones con mejor puntaje
    keyword_extracted = rake_nltk_var.get_ranked_phrases()[:5]
    return keyword_extracted
    
def textRank(doc):

    #Se carga el pipeline de spacy
    nlp = spacy.load("en_core_web_sm")
    #Se añade el modelo de textrank al pipeline
    nlp.add_pipe("textrank")

    doc = nlp(doc)  #Se carga el nuevo objeto de tipo doc
    #Se copian las primeras 5 frases con su puntaje
    frases=[]
    for phrase in doc._.phrases[:5]:
        frases.append((phrase.text,phrase.rank))
    return frases

#Cargar documentos del corpus
corpus={}
corpus["Tecnologia1"]=io.open("./corpus/tecno1.txt",mode="r",encoding="utf-8").read()
corpus["Tecnologia2"]=io.open("./corpus/tecno2.txt",mode="r",encoding="utf-8").read()
corpus["Tecnologia3"]=io.open("./corpus/tecno3.txt",mode="r",encoding="utf-8").read()
corpus["Psicologia1"]=io.open("./corpus/psyco1.txt",mode="r",encoding="utf-8").read()
corpus["Psicologia2"]=io.open("./corpus/psyco2.txt",mode="r",encoding="utf-8").read()
corpus["Psicologia3"]=io.open("./corpus/psyco3.txt",mode="r",encoding="utf-8").read()

#Imprimir las 5 palabras clave de cada libro
for titulo,documento in corpus.items():

    #Extraer texto del libro sin incluir bibliografia o apendices
    doc=documento[:re.search("END|APPENDIX|Bibliography|INDEX",documento).start()-3]

    #TF-IDF – NLTK
    kw_tf_idf=tf_idf(doc)

    #BERT – Transformers
    kw_bert=bert(doc)

    #Rake - NLTK
    kw_rake=rake(doc)

    #TextRank – Spacy.
    kw_textRank=textRank(doc)


    print(f"\n\nLibro {titulo}\nTF-IDF\n{kw_tf_idf}\nBERT\n{kw_bert}\nRAKE\n{kw_rake}\nTextRank\n{kw_textRank}")