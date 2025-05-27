#Practica 5: 
#Nombre: Tinoco Videgaray Sergio Ernesto
#Grupo: 5BV1
#Carrera: Ingenieria en Inteligencia Artificial
#Fecha de última modificación 13/01/24

#Paquetes
import pandas as pd
import pysentiment2 as ps
from nltk.corpus import opinion_lexicon
from sklearn.metrics import confusion_matrix, accuracy_score
import re
from sklearn import metrics
import numpy  as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag
from random import shuffle
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import pickle as pickle
from math import log10


import time


#PREPROCESAMIENTO
#Analisis exploratorio de datos
def eda(dataset):
    df=dataset
    print(df.shape) #Dimensiones del dataset
    print(df.info())    #Metadatos del dataset
    
    for column in df.columns:
        print(df[column])

#Etiquetar cada sentimiento segun el puntaje de cada compra
def calif_to_sentiment(df):
    df = df[['Review', 'Score']]
    df['Sentiment'] = df['Score'].apply(lambda x: 'positive' if x > 3 
                                        else 'negative' if x < 3
                                        else 'neutral')
    df = df[['Review', 'Sentiment']]
    return df

def balancear_clases(dataframe,samples):    #Equilibrar numero de muestras para cada sentimiento
    df_b = pd.DataFrame({'Review': [],
                        'Sentiment': []})
    p,n,neu=0,0,0   #Contadores
    s=samples #Muestras por clase
    for row in range(len(dataframe)):   #Por cada reseña
        sentiment=dataframe.iloc[row,1]
        if sentiment == "positive" and p<s:
            df_b=df_b._append([dataframe.loc[row,["Review","Sentiment"]]], ignore_index=True)
            p+=1
        elif sentiment == "negative" and n<s:
            df_b=df_b._append([dataframe.loc[row,["Review","Sentiment"]]], ignore_index=True)
            n+=1
        elif sentiment == "neutral" and neu<s:
            df_b=df_b._append([dataframe.loc[row,["Review","Sentiment"]]], ignore_index=True)
            neu+=1
        else:
            if p<s or n<s or neu<s: #Si todavia faltan muestras
                continue    #Seguir iterando
            else:
                break   #En caso contrario salir del bucle
    df_b = df_b.sample(frac=1).reset_index(drop=True)      #Mezclar muestras            
    return df_b

def limpiar_dataframe(dataframe):
    text_clean=[]
    stop_words = set(stopwords.words('english'))    #Lista de stopwords
    for reseña in dataframe["Review"]:  #Por cada reseña
        #Se define la expresion regular para limpiar el corpus de las reseñas
        reg_ex=':-\)|:\)|: \)|<.{1,2} />|<a.+">|</a>|\$?\d+\.?\d*\S |( street|avenue| north| south| east| west|corner)\S+|\(?Pack of \d+\)|,|\.|\?|!|-|:|;|\)|\(|"'+"|\s'"+"|'\s"+"|`"
        text=re.sub(reg_ex, '', reseña) #Remplazar cualquier coicnidencia por caracter nulo
        #Eliminar stopwords del corpus
        tokens = word_tokenize(text)
        tokens= [w for w in tokens if not w.lower() in stop_words]
        text=' '.join(tokens)
        text_clean.append(text)
    dataframe["Review"]=text_clean
    return dataframe

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
        return wordnet.ADV

def normalizar_dataframe(dataframe):
    text_normalized=[]
    ps=PorterStemmer()  #Se define el objeto para aplicar el stemming
    lematizer=WordNetLemmatizer()   #Instancia del WordNetLemmatizer
    for reseña in dataframe["Review"]:  #Por cada reseña
        #Tokenizar
        tokens=word_tokenize(str(reseña).lower())
        #Aplicar stemming
        tokens=[ps.stem(token) for token in tokens]
        #Aplicar POS-tagging
        tokens=pos_tag(tokens)  #Usando metodo pos tag 
        #Reasignar etiquetas con formato wordnet
        tokens= list(map(lambda x: (x[0], pos_tagger(x[1])), tokens))
        #Aplicar lematizacion utilizando las etiquetas asignadas en el POS-tagging
        tokens=[lematizer.lemmatize(token,tag) for token,tag in tokens]
        text_normalized.append(' '.join(tokens))
    #Definir tf-idf vectorizer
    vectorizer = TfidfVectorizer()
    #Ajustar y transformar vectorizador
    X = vectorizer.fit_transform(text_normalized)    
    #Obtener tokens del diccionario
    tokens = vectorizer.get_feature_names_out()

    vectores_tf_idf=[]
    for x in X.toarray():
        vectores_tf_idf.append([i for i in x])  #Convertir en matriz

    dataframe["Review"]=text_normalized
    dataframe["TF_IDF"]=vectores_tf_idf

    return dataframe

#PROCESAMIENTO
def harvard_polarity(dataframe):    #Polaridad utilizando diccionario de harvard 4
    hiv4 = ps.HIV4()
    scores=[]   #Lista de puntajes
    reseñas=dataframe["Review"] 
    for reseña in reseñas:  #Por cada reseña
        tokens = hiv4.tokenize(reseña)  #Tokenizar reseña
        scores.append(hiv4.get_score(tokens)["Polarity"])   #Obtener puntaje con el diccionario
    dataframe["Harvard_Polarity"]=scores    #Guardar puntajes

    print(dataframe.loc[:,["Sentiment","Harvard_Polarity"]])    #Mostrar puntajes
    #Convertir puntajes en etiquetas de sentimiento 
    predict = dataframe["Harvard_Polarity"].apply(lambda x: 2 if x > 0.4
                                        else 0 if x < -0.4
                                        else 1)
    #Convertir variables categoricas a dumys (numericas)
    codificadorDatos=LabelEncoder()
    label=codificadorDatos.fit_transform(dataframe["Sentiment"])

    #Matriz de Confusión
    print(f'Matriz de Confusión:\n{confusion_matrix(label, predict)}')
    #Precision del modelo
    print(f'Exactitud del modelo:\n{accuracy_score(label, predict)}')
    #Covertir dumys a categoricas
    predict=codificadorDatos.inverse_transform(predict)
    dataframe["Harvard_Polarity"]=predict

def opinionLexicon(dataframe):  #polaridad utilizando diccionario de opinion lexicon
    sentiment=[]    #Lista de sentimientos
    hiv4 = ps.HIV4()
    reseñas=dataframe["Review"]
    lst_positive=opinion_lexicon.positive()     #Lista de sentimientos positivos
    lst_negative=opinion_lexicon.negative() #Lista de sentimientos negativos
    for reseña in reseñas:   #Por cada reseña
        tokens = hiv4.tokenize(reseña) #Tokenizar reseña
        #Sumar numero de tokens que se encuentran en la lista de terminos positivos
        p = sum(el in tokens for el in lst_positive)
         #Sumar numero de tokens que se encuentran en la lista de terminos negativos
        n=sum(el in tokens for el in lst_negative)
        #Agrega etiqueta de sentimiento segun el contador que haya sido mayor
        sentiment.append("positive" if p>n else "negative" if p<n else "neutral")
    dataframe["OP_Lexicon_Polarity"]=sentiment

    #Variables ficticias a dumys (numericas)
    codificadorDatos=LabelEncoder()
    label=codificadorDatos.fit_transform(dataframe["Sentiment"])
    predict=codificadorDatos.fit_transform(dataframe["OP_Lexicon_Polarity"])

    #Matriz de Confusión
    print(f'Matriz de Confusión:\n{confusion_matrix(label, predict)}')
    #Precision del modelo
    print(f'Exactitud del modelo:\n{accuracy_score(label, predict)}')
    return dataframe

def stringTolist(ini_list): #Conertir cadena a lista
    import ast
    res = ast.literal_eval(ini_list)
    return res

def logistic_regression(dataframe):
    # REGRESION LOGISTICA
    label=dataframe.iloc[:,2].values #Etiquetas de sentimientos
    data=[]
    for string in dataframe.iloc[:,3].values: #Por cada vector TF-IDF
        data.append(stringTolist(string))   #Convertir cadena TF-IDF a lista
    
    data=np.array(data) #Converir lista en objeto numpy
      
    #Variables ficticias a dumys
    codificadorDatos=LabelEncoder()
    label=codificadorDatos.fit_transform(label)
    
    #Dividir conjunto de datos en pruebas y entrenamiento
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label,test_size=0.2,random_state=3)

 
    # Creación del modelo
    lr = LogisticRegression(multi_class='multinomial') 
    # Ajustamos el modelo a los datos de entrenamiento
    lr = lr.fit(X_train, y_train)
    # Realizamos prediciones
    y_pred = lr.predict(X_test)

    #Matriz de Confusión
    print(f'Matriz de Confusión:\n{confusion_matrix(y_test, y_pred)}')
    #Precision del modelo
    print(f'Exactitud del modelo:\n{accuracy_score(y_test, y_pred)}')

    #Predecir todos los valores segun el modelo ajustado
    lr_pred=lr.predict(data)
    #Convertir valores predecidos a su correspondiente etiqueta
    lr_pred=codificadorDatos.inverse_transform(lr_pred)
    dataframe["LR_Polarity"]=lr_pred

def decision_tree(dataframe):
    #Arboles de Decision para clasificacion
    label=dataframe.iloc[:,2].values    #Etiquetas de sentimientos
    data=[]
    for string in dataframe.iloc[:,3].values:
        data.append(stringTolist(string))
    
    data=np.array(data)

    #Variables ficticias a dumys
    codificadorDatos=LabelEncoder()
    label=codificadorDatos.fit_transform(label)

    #Dividir conjunto de datos en pruebas y entrenamiento
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.8,random_state=23)

    #Crear instancia de la clase decision tree clasifier
    dtC = DecisionTreeClassifier()
    #Ajustar modelo al conjunto de entrenamiento 
    dtC.fit(X_train, y_train)   
    #Realizar la inferencia de datos con el modelo ajustado previamente
    y_pred=dtC.predict(X_test)

    # Matriz de Confusión
    print(f'Matriz de Confusión:\n{confusion_matrix(y_test, y_pred)}')

    #Precision del modelo
    print(f'Exactitud del modelo:\n{accuracy_score(y_test, y_pred)}')

    #Predecir todos los valores segun el modelo ajustado
    data_pred=dtC.predict(data)
    #Convertir valores predecidos a su correspondiente etiqueta
    data_pred=codificadorDatos.inverse_transform(data_pred)
    dataframe["DT_Polarity"]=data_pred

def svm(dataframe):
    #Maquinas de soporte vectorial para clasificacion
    label=dataframe.iloc[:,2].values    #Etiquetas de sentimiento
    data=[]
    for string in dataframe.iloc[:,3].values:   #Obtener vectores TF-IDF del dataframe
        data.append(stringTolist(string))
    
    data=np.array(data) #Convertir en objeto numpy

    #Variables ficticias a dumys
    codificadorDatos=LabelEncoder()
    label=codificadorDatos.fit_transform(label)

    #Dividir conjunto de datos en pruebas y entrenamiento
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2,random_state=33)


    #Escalar variables
    sc_x=StandardScaler()
    sc_y=StandardScaler()
    X_train=sc_x.fit_transform(X_train)
    X_test=sc_x.transform(X_test)


    #Entrenar el modelo SVM
    svC=SVC()
    svC.fit(X_train,y_train)
    #Prediccion del test de prueba
    y_pred=svC.predict(X_test)


    # Matriz de Confusión
    print(f'Matriz de Confusión:\n{confusion_matrix(y_test, y_pred)}')

    #Precision del modelo
    print(f'Precisión del modelo:\n{metrics.accuracy_score(y_test, y_pred)}')

    #Predecir todos los vectores TF-IDF
    data_pred=svC.predict(data)
    #Convertir valores predecidos a su etiqueta de sentimiento
    data_pred=codificadorDatos.inverse_transform(data_pred)
    dataframe["SVM_Polarity"]=data_pred

def nn(dataframe):
    #Red neuronal
    #Definir tokenizador
    tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
    tokenizer.fit_on_texts(dataframe['Review']) #Tokenizar reseñas del dataframe
    word_index = tokenizer.word_index   
    #Obtener representaciones de embeddings en one hot encoding
    sequences = tokenizer.texts_to_sequences(dataframe['Review'])   
    data= pad_sequences(sequences, maxlen=100, truncating='post')   


    #Variables ficticias a dumys en one hot encoding 
    label = pd.get_dummies(dataframe['Sentiment']).values

    #Dividir conjunto en pruebas y entrenamiento
    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.2,random_state=13)

    #Definir arquitectura de la red neuronal
    model = Sequential()
    model.add(Embedding(5000, 100, input_length=100))   #Capa de embedding
    model.add(Conv1D(64, 5, activation='relu')) #Capa convolucional
    model.add(GlobalMaxPooling1D()) #Capa de max pooling
    model.add(Dense(32, activation='relu')) #Capa completamente conectada con activacion relu
    model.add(Dropout(0.5)) #Probabilidad de "apagar" neurona
    model.add(Dense(3, activation='softmax'))   #Capa de salida con funcion softmax
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    #Entrenar modelo con los hiperparametros
    model.fit(x_train, y_train, epochs=13, batch_size=32, validation_data=(x_test, y_test))

    #Hacer la inferencia de los datos de prueba
    y_pred = np.argmax(model.predict(x_test), axis=-1)
    print("Precision:", accuracy_score(np.argmax(y_test, axis=-1), y_pred))

    #Predecir todos los datos con el modelo entrenado
    data_pred=np.argmax(model.predict(data), axis=-1)

    #Convertir valores estimados en su etiqueta correspondiente al sentimiento
    polarity=[]
    for pred in data_pred:
        if pred == 0:
            polarity.append("negative")
        elif pred == 1:
            polarity.append("neutral")
        else:
            polarity.append("positive")

    dataframe["NN_Polarity"]=polarity
    


def preprocesarDatos(s):
    df=pd.read_csv("reviews.csv")
    #Invocar a la funcion para realizar el analisis exploratorio de datos
    eda(df)
    df=calif_to_sentiment(df)
    df=balancear_clases(df,s)
    df=limpiar_dataframe(df)
    df=normalizar_dataframe(df)
    eda(df) #Analizar nuevo dataframe
    df.to_csv('dataframe_vectorized.csv')

def procesarDatos():
    #Cargar dataframe preprocesado
    df=pd.read_csv("dataframe_vectorized.csv")
    #eda(df) #Analizar dataframe

    #Harvard polarity
    harvard_polarity(df)
    print(df.loc[:,["Sentiment","Harvard_Polarity"]])
    
    #Opinion Lexicon Polarity
    opinionLexicon(df)
    print(df.loc[:,["Sentiment","OP_Lexicon_Polarity"]])
    
    #Regresion logistica
    logistic_regression(df)
    print(df.loc[:,["Sentiment","LR_Polarity"]])

    #Arboles de decision
    decision_tree(df)   
    print(df.loc[:,["Sentiment","DT_Polarity"]])

    #Maquinas de soporte vectorial
    svm(df)  
    print(df.loc[:,["Sentiment","SVM_Polarity"]])

    #Redes neuronales
    nn(df)  #Entrenar modelo
    print(df.loc[:,["Sentiment","NN_Polarity"]])

    #Exportar resultados
    df_p=df.loc[:,["Sentiment","Harvard_Polarity","OP_Lexicon_Polarity",\
                 "LR_Polarity","DT_Polarity","SVM_Polarity","NN_Polarity"]]
    df_p.to_csv('dataframe_predicted.csv')

def main(): #Funcion main
    start_time = time.time()
    preprocesarDatos(2000)
    procesarDatos()
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()


