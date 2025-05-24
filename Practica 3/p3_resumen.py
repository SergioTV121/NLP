#Practica 3: 
#Nombre: Tinoco Videgaray Sergio Ernesto
#Grupo: 5BV1
#Carrera: Ingenieria en Inteligencia Artificial
#Fecha de última modificación 01/12/23
#Programa 2: Generador de resumenes

#Paquetes a utilizar
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import io
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from gensim.summarization import summarize
import re

def word_freq_summarize(text):
    #Se carga el pipeline de spacy
    nlp = spacy.load('en_core_web_sm')
    #Se agrega el texto al pipeline
    doc= nlp(text)
    tokens=[token.text for token in doc]
    #Se crea un diccionario para las frecuencias de cada palabra
    word_frequencies={}
    for word in doc:
        #Se normaliza cada palabra
        if word.text.lower() not in list(STOP_WORDS):
            if word.text.lower() not in punctuation:
                if word.text not in word_frequencies.keys():
                    word_frequencies[word.text] = 1
                else:
                    word_frequencies[word.text] += 1

    max_frequency=max(word_frequencies.values())    #Maxima frecuencia
    for word in word_frequencies.keys():
        word_frequencies[word]=word_frequencies[word]/max_frequency #Se normaliza la frecuencia

    sentence_tokens= [sent for sent in doc.sents]   #Se obtienen las oraciones
    sentence_scores = {}    #Diccionario para el puntaje de las oraciones
    for sent in sentence_tokens:    #Por cada oracion
        for word in sent:   #Se recorre cada palabra en la oracion
            if word.text.lower() in word_frequencies.keys():    
                #Se suman las frecuencias de cada palabra para el puntaje
                if sent not in sentence_scores.keys():                            
                    sentence_scores[sent]=word_frequencies[word.text.lower()]
                else:
                    sentence_scores[sent]+=word_frequencies[word.text.lower()]
     
    #Se obtienen las 10 oraciones con mayor puntaje del diccionario
    summary=nlargest(10, sentence_scores,key=sentence_scores.get)
    #Se concatenan las oraciones para formar el resumen final
    final_summary=[word.text for word in summary]
    summary=''.join(final_summary)
    return summary

def gensim_summarize(doc):
    #Se aplica el metodo summariza del paquete gensim.summarization
    #Se define un porcentaje o ratio para el resumen
    text_summary = summarize(doc,ratio=0.1)
    return text_summary
    
def text_rank_summarize(doc):

    #Se crea un ojbeto de tipo PlaintextParser junto con un Tokenizador
    #Esto con el fin de crear un objeto de tipo document
    parser = PlaintextParser.from_string(doc,Tokenizer("english"))
    #Objeto de tipo TextRankSummarizer
    summarizer = TextRankSummarizer()

    #Se carga el objeto del plainParser al objeto summarizer y se define 
    #el numero de oraciones (5)
    summary =summarizer(parser.document,10)
    text_summary=""

    for sentence in summary:
        text_summary+=str(sentence)

    return text_summary

def lsa_summarize(doc):

    #Se crea un ojbeto de tipo PlaintextParser junto con un Tokenizador
    parser = PlaintextParser.from_string(doc, Tokenizer("english"))

    #Objeto de tipo LsaSummarizer
    summarizer_lsa = LsaSummarizer()

    #Se carga el objeto del plainParser al objeto summarizer y se define el
    #numero de oraciones (10)
    summary =summarizer_lsa(parser.document,10)
    return summary


#Cargar documentos del corpus en un objeto de tipo diccionario
corpus={}
corpus["Psicologia"]=io.open("./corpus/psyco1.txt",mode="r",encoding="utf-8").read()
corpus["Tecnologia"]=io.open("./corpus/tecno3.txt",mode="r",encoding="utf-8").read()

#Genera el resumen del primer capitulo de cada libro
for titulo,documento in corpus.items():

    #Extrae el primer capitulo del documento utilizando una RegEx
    doc=documento[:re.search("(i|I){2}|Chapter Two",documento).start()-7]

    #Frecuencia de palabras normalizada
    summary_wfn=word_freq_summarize(doc)

    #Gensim
    summary_gensim=gensim_summarize(doc)

    #TextRank
    summary_tr=text_rank_summarize(doc)

    #LSA
    summary_lsa=lsa_summarize(doc)

    print(f"\n\nLibro {titulo}\nFreq Normalizada de Palabras\n{summary_wfn}\n\nGensim\n{summary_gensim}\n\nTextRank\n{summary_tr}\n\nLSA\n{summary_lsa}")
