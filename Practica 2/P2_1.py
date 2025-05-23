#Practica 2: Expresiones regulares y Vectorizacion de texto
#Nombre: Tinoco Videgaray Sergio Ernesto
#Grupo: 5BV1
#Carrera: Ingenieria en Inteligencia Artificial
#Fecha de última modificación 09/10/23
#Programa 1: Expresiones regulares

#Importar paquetes
import re

def regex_gastos(texto):    #Regex del documento de gastos
    regex=[]
    regex.append(re.findall("[r|R].*[g|G]",texto))  #Contengan una 'r' y una 'g'
    regex.append(re.findall("[1-9]\d{2,10}",texto)) #Valores mayores a 100
    regex.append(re.findall("[a|A][^a|A|b|B].*[b|B][^b|B|c|C].*[c|C]",texto)) #Contengan a,b y c con las restricciones descritas
    regex.append(re.findall("[a|A].*\d|\d.*[a|A]",texto))   #Contengan una 'a' y un digito en cualquier orden
    regex.append(re.findall("[d|D].*[i|I]",texto))  #Contengan una 'd' y una 'i'
    i=0
    print("\nDocumento de gastos")
    for reg in regex:
        i+=1
        print("\nMatchs en RegEx",i)
        for match in reg: print(match)

def regex_peliculas(texto): #Regex de peliculas estrenadas antes del 2002
    texto=str(texto).splitlines()
    peliculas=[]
    for linea in texto:
        #Se divien en 2 rangos: antes del 2000 y a partir del 2000
        if (re.findall("\(1\d{3}|2\d{2}[0-1]\)",linea)):   
            peliculas.append(linea[:-7])
    print("\nPeliculas estrenadas antes del 2002")            
    for pelicula in peliculas: print(pelicula)

def regex_receteas(texto):  #Regex de recetas que NO contengan 'chocolate'
    texto=str(texto).splitlines()
    recetas=[]
    for linea in texto:
        #Usando grupos de no captura y un operador negative lookahead
        #Se agregan las lineas que no contencan la palabra 'cholocate' considerando mayusculas y minsuculas
        if (re.findall("^(?:(?![C|c][H|h][O|o][C|c][O|o][L|l][A|a][T|t][E|e]).)*$",linea)):
            recetas.append(linea)
    print("\nRecetas que no contenan 'chocolate'")            
    for receta in recetas: print(receta)
    
def insertar_comas(cadena): #Regex para agrupar 3 digitos y agregar comas
    cadena_invertida=cadena[::-1]   #Se invierte la cadena de digitos
    #Se aplica la expresion regular para agrupar los digitos en 3 o 1 y 2
    lista=re.findall("\d{3}|\d{1,2}",cadena_invertida)
    grupo_comas=[]
    primer_grupo=True #Bandera para identificar al primer grupo de digitos
    for grupo in lista: #Por cada grupo de digitos
        #Agregar a la lista el grupo concatenandolo con una coma en caso de no ser el primer grupo
        grupo_comas.append(grupo.replace(grupo,grupo if primer_grupo else ","+grupo))
        primer_grupo=False
    #Convertir lista de grupos en caden de texto
    grupo_comas="".join(grupo_comas)
    return grupo_comas[::-1] #Retornar lista invertida

def regex_poblaciones(texto):   #Funcion para separar numero de poblacion en comas
    texto=str(texto).splitlines()
    paises=[]
    poblaciones=[]
    for linea in texto:
        #Agregar nombres de paises separando por espacios a excepcion
        paises.append(linea.split(" ")[1] if linea.split(" ")[1]!="United" else "United States")
        #Agregar numero de habitantes separando por comas usando la funcion definida previamente
        poblaciones.append(insertar_comas(linea))
    print("\nPoblaciones con formato de comas")           
    for pais in zip(paises,poblaciones): print(pais)    

def reducir_IPV6(ipv6): #Aplicar regex para abreviar IPV6
    ip=ipv6[:]  #Copia de la direccion IPV6
    #Aplicar regex en la que se evaluan 2 casos:
    #1 se tienen hasta 3 ceros a la izquerda de cada bloque seguido de algun valor hexadecimal
    #2 se tienen bloques de ceros adyacentes
    matches=re.findall("(:0{1,3}[1-9a-f]|(:0000)+)",ip)   #:0{1,3}[1-9a-f]|(:0000)+
    for match in matches:
       if len(match[0]) < 6: #Si es el caso 1 con ceros a la izquierda
           #Se remplazan por el primer y ultimo elemento que corresponen con los 2 puntos y el valor hexadecimal
           ip=ip.replace(match[0],match[0][0]+match[0][-1]) 
       else: #Si es el caso 2 con bloques de ceros
           ip=ip.replace(match[0],":",1)   #Remplazar bloques por dos puntos ':'
    return ip

def regex_simplificar_IPV6(texto): #Funcion para abreviar IPV6
    texto=str(texto).splitlines()
    direcciones_simplificadas=[]
    for linea in texto: #Por cada IPV6
        #Añadir la IPV6 abreviada usando la funcion definida previamente
        direcciones_simplificadas.append(reducir_IPV6(linea))
    print("\nIPV6 abreviadas")           
    for ip,ip_ab in zip(texto,direcciones_simplificadas):
        print(f"{ip}      |       {ip_ab}")

#Cargar documento de texto
doc_gastos=open("./documentos_regex/expenses.txt").read()
doc_peliculas=open("./documentos_regex/peliculas.txt").read()
doc_recetas=open("./documentos_regex/recetas.txt").read()
doc_poblaciones=open("./documentos_regex/poblaciones.txt").read()
doc_ipv6=open("./documentos_regex/ipv6.txt").read()

#Aplicar las regex a cada documento
regex_gastos(doc_gastos)
regex_peliculas(doc_peliculas)
regex_receteas(doc_recetas)
regex_poblaciones(doc_poblaciones)
regex_simplificar_IPV6(doc_ipv6)
