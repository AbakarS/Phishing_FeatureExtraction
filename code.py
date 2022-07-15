# -*- coding: utf-8 -*-

----------------------## **Importer et nettoyer les données les données**---------------------------------


import pandas as pd
df=pd.read_csv("/content/urldata.csv")


# Afficher les 5 premieres lignes 
df.head(5)


"""### **Sppression de tous les espaces dans les URLs**"""
def RemoveSpace(url):
    return url.replace(" ", "")
df['url']= df['url'].apply(lambda x: RemoveSpace(x))


"""### **Afficher la forme de notre data set**"""
df.shape
"""Il y a quatre colonnes et 450176 observations dans notre data set


### **Verification des valeurs manquantes**
"""
print(df.isnull().sum())
"""On constate qu'il existe aucune valeur manquante dans notre data set.

### **Verification des doublons**
"""


# Les doublons
df.duplicated(keep=False).sum()

"""Aucun  doublon non plus dans notre data set


### **Supprimer la variable "Unnamed: 0"**
"""
df=df.drop("Unnamed: 0", axis=1)


"""### **Jeter un coup d'oeil sur quelques URLs de notre data set**"""
df["url"][10:15]


"""### **Description de notre data set**"""
df.info()
"""On constate, dans cette description, qu'on a une variable numérique (result) et deux autres catégorielles (url et label) avec 450176 observations et aucune variable manquante.


### **Verifier les catégories de classes de notre data set**
"""
print(df["label"].value_counts())
#Afficher un espace
print()
print(df["result"].value_counts())
"""On constate qu'il y a 345738 URLs sont classées benignes et 104438 sont classées malveillantes.


### **Visualiser les catégories de classes de notre variable target (label ou result)**
[Countplot](https://seaborn.pydata.org/generated/seaborn.countplot.html)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

total = float(len(df["label"])) # Une URL par ligne 
plt.figure(figsize=(12, 7))
ax = sns.countplot(x="label", saturation =0.70, data=df, palette="Set1") 
plt.ylabel("Nombre d'URLs",fontsize=15)
plt.xlabel("label", fontsize=15)
for p in ax.patches:
     height = p.get_height()
     ax.text(p.get_x()+p.get_width()/2.,
             height + 3,
             '{:1.2f}'.format(height/total),
             ha="center")


#camembert
"""[camembert](https://stackoverflow.com/questions/63687789/how-do-i-create-a-pie-chart-using-categorical-data-in-matplotlib)
"""
plt.rcParams["figure.figsize"] = [15,10] 
df.label.value_counts().plot(kind='pie', autopct='%1.0f%%')
"""On constate que nos données sont mal equilibrées. La catégorie des URLs malignes domine largement la catégorie des URLs malveillantes. Nous allons appliquer la technique de sous-échantillonnage (undersampling) pour équlibrer nos données.
"""

----------------------------- ### **Technique d'Undersampling**------------------------------------------------
# Nombre de classes
count_0, count_1 = df['result'].value_counts()
print("Nombre de frequences de 0 :", count_0)
print("Nombre de frenquences de 1 :",count_1)


# Séparer les classes 
classe_0 = df[df['result'] == 0]
classe_1 = df[df['result'] == 1] 
print('classe 0 :', classe_0.shape)
print('classe 1 :', classe_1.shape)


#Sous-échantillonnage aléatoire de count_0
classe_0_min = classe_0.sample(count_1)


#Nouvelle base de données sous-échantillonnée
new_data = pd.concat([classe_0_min, classe_1], axis=0)
print("Le total de classes 1 et 0 après avoir effectué la métbhode undersampling :",new_data['label'].value_counts())
print()
print("On constate que les classes de données sont équilibrées de maniére égale ")


"""### **Données sous-échantillonnées**"""
new_data


"""**Visualiser les données sous-échantillonnées**"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#Countplot des classes
total = float(len(new_data["label"])) # Nombre total d'observations 
plt.figure(figsize=(10, 7))
ax = sns.countplot(x="label", saturation =0.75, data=new_data, palette="Set1") 
plt.ylabel("Nombre d'URLs", fontsize=15)
plt.xlabel("label", fontsize=15)
for p in ax.patches:
     height = p.get_height()
     ax.text(p.get_x()+p.get_width()/2.,
             height + 3,
             '{:1.2f}'.format(height/total),
             ha="center")

"""On constate que nos catégories de classes sont maintenant équilibrées à hauteur de 50% chacune. 
NB : il faut garder à l'esprit que ce type de sous-échantillonnge supprimera des données dans la catégorie de classe dominante et ce qui pourra en même temps supprimer des informations qui pourraient être utiles.
"""

-----------------------------------## **Features engineering**------------------------------
Les caracteristiques suivantes seront extraites de l'URL pour la classification.


### **1) Les caractéristiques de ponctuation**
- Compter le nombre de '@'
- Compter le nombre de '?'
- Compter le nombre de '%'
- Compter le nombre de '#'
- Compter le nombre de '.'
- Compter le nombre de 'www'
- Compter le nombre de '-'
- Compter le nombre de '/'


### **2) Le nombre de http et https dans l'URL**
- Compter le hattp
- Compter le https


### **3) Les caractéristqiues de longueur**
- Longueur de l'URL
- Longueur du nom d'hôte
- Longueur du chemin


#### **4) Les caractéristiques binaires**
- "http/https" dans le nom de domaine 
- Utilisation de IP ou non
- Redierection


#### **1.1. Les caractéristiques de ponctuation**


# Compter le nombre de @
def get_count_at(url):
    return url.count('@')
df['count@'] = df['url'].apply(lambda x: get_count_at(x))
df.head(5)


# Compter le nombre ?
def get_count_interrogation_point(url):
    return url.count('?')
df['count?'] = df['url'].apply(lambda x: get_count_interrogation_point(x))
df.head(5)


# Compter le nombre de %
def get_count_symbol(url):
    return url.count('%')
df['count%'] = df['url'].apply(lambda x: get_count_symbol(x))
df.head(5)


# Compter le nombre de #
def get_count_symbol(url):
    return url.count('#')
df['count#'] = df['url'].apply(lambda x: get_count_symbol(x))
df.head(5)


# Compter le nombre de points (.)
def get_count_dot(url):
    return url.count('.')
df['count.'] = df['url'].apply(lambda x: get_count_dot(x))
df.head(5)


# Compter le nombre www
def get_count_subdomain(url):
    return url.count('www')
df['count-www'] = df['url'].apply(lambda x: get_count_subdomain(x))
df.head(5)


# Compter le tiret d'union (-)
def get_count_hyphen(url):
    return url.count('-')
df['count-'] = df['url'].apply(lambda x: get_count_hyphen(x))
df.head()


# Compter le nombre du slash
def get_single_slash(url):
    return url.count('/')
df['count/'] = df['url'].apply(lambda i: get_single_slash(i))
df.head()


"""#### **2.1. Le nombre de http et https dans l'URL**"""


# Compter le nombre de hattp
def get_protocol_count(url):
  return url.count('http')
df['count_http'] = df['url'].apply(lambda i: get_protocol_count(i))
df.head()


# Compter le nombre https
def get_protocol_count(url):
  return url.count('https')
df['count_https'] = df['url'].apply(lambda i: get_protocol_count(i))
df.head()


"""#### **3.1. Les caractéristiques de longueur**"""

# Bibliothéques 
from urllib.parse import urlparse
import os.path


# Longueur de l'URL
def get_url_length(url):
  return len(url)
df['url_length'] = df['url'].apply(lambda i: get_url_length(i))
df.head(5)


#Longueur du nom de domaine
def get_hostnam_length(url):
  return len(urlparse(url).netloc)
df['hostnam_length'] = df['url'].apply(lambda i: get_hostnam_length(i))
df.head(5)


#Longueur du path
def get_path_length(url):
  return len(urlparse(url).path)
df['path_length'] = df['url'].apply(lambda i: get_path_length(i))
df.head(5)



"""#### **4.1. les Caractéristiques binaires**"""

# L'existence du https dans la partie du nom domain
def httpsDomain(url):
  domain = urlparse(url).netloc
  if 'https' in domain:
    return -1
  else:
    return 1
df['https_Domain'] = df['url'].apply(lambda x: httpsDomain(x))
df.head()


# L'existence du http dans la partie du nom domain
def httpDomain(url):
  domain = urlparse(url).netloc
  if 'http' in domain:
    return -1
  else:
    return 1
df['http_Domain'] = df['url'].apply(lambda x: httpDomain(x))
df.head()


#Utilisation de l'adresse IP dans le nom du domaine
import re 
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return -1
    else:
        # print 'No matching pattern found'
        return 1
df['use_ip'] = df['url'].apply(lambda i: having_ip_address(i))
df.head()


# La position du double slash ('//') dans l'URL 
def redirection(url):
  pos = url.rfind('//')
  if pos > 6:
    if pos > 7:
      return -1
    else:
      return 1
  else:
    return 1
df['redirection'] = df['url'].apply(lambda i: redirection(i))
df.head()


# Afficher la forme de la base de données 
print(df.shape)


"""### **Suppression de la variable "URL"**"""
df=df.drop("url", axis=1)
"""Une fois toutes les caractéristiques que nous voudrions sont extraites, nous supprimons la variable url. Nous gardons le reste de variable pour faire une meilleure visualisation de données."""


#Afficher les données finales
df.head()
df.shape
"""Nous avons au final 19 variables"""

---------------------### **Statistiques descriptives**----------------------------------

df.describe().T


-----------------------------------"""### **Corrélation: Heatmap**------------------------------
[heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html)"""
#Heatmap
import matplotlib.pyplot as plt
import seaborn as sns 
corrmat = df.corr()
f, ax = plt.subplots(figsize=(25,20))
sns.heatmap(corrmat, square=True, annot = True, annot_kws={'size':14})

----------------------------## **Visualisation des données**------------------------------

"""### **Histogrammes**
[Histogramme](https://matplotlib.org/3.5.1/api/_as_gen/matplotlib.pyplot.hist.html)
"""

# Histogramme de la longueur de l'url
plt.figure(figsize=(12,7))
plt.hist(df['url_length'], bins=150, color = "lightblue", ec="red", lw=2)
plt.xlabel("longueur d'URL",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=15)
plt.xlim(0,200)


#Histogramme de la longueur du nom de l'hôte
plt.figure(figsize=(12,7))
plt.hist(df['hostnam_length'], bins=30, color = "lightblue", ec="red", lw=2)
plt.xlabel("Lontgueur du Host name",fontsize=15)
plt.ylabel("Nombre d'Urls",fontsize=15)
#plt.ylim(0,1000)
plt.xlim(0,70)


#countplot de l'adresse IP
total = float(len(df["use_ip"])) 
plt.figure(figsize=(12, 7))
ax = sns.countplot(x="use_ip", saturation =0.75, data=df, palette="Set1") 
plt.xlabel("Proportion d'URL contenant une adresse IP",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=15)
for p in ax.patches:
     height = p.get_height()
     ax.text(p.get_x()+p.get_width()/2.,
             height + 3,
             '{:1.2f}'.format(height/total),
             ha="center")


#Coutplot de l'adresse IP par rapport à la variable target (label)
total = float(len(df["use_ip"])) 
plt.figure(figsize=(12, 7))
ax = sns.countplot(x="use_ip", hue="label", saturation =0.75, data=df, palette="Set1") 
plt.xlabel("Proportion d'URL contenant une adresse IP",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=15)
for p in ax.patches:
     height = p.get_height()
     ax.text(p.get_x()+p.get_width()/2.,
             height + 3,
             '{:1.2f}'.format(height/total),
             ha="center")


#countplot de la varible count_http
plt.figure(figsize=(12,7))
plt.ylim((0, 1500))
sns.countplot(df['count_http'], saturation=0.80, palette="Set1")
plt.xlabel("Nombre de http dans une URL",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=15)


# countplot de la variable count_http en fonction de la variable label
plt.figure(figsize=(12,7))
sns.countplot(df['count_http'],hue='label', saturation=0.80, palette="Set1",data=df)
plt.xlabel("Nombre de https dans une URL",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=15)
plt.ylim((0,1500))


#countplot de la variable count_https
plt.figure(figsize=(12,7))
sns.countplot(df['count_https'],saturation=0.80, palette="Set1",data=df)
plt.ylim((0,1500))
plt.xlabel("Nombre de https",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=15)


#countplot de la variable count_http en fonction de la variable cible label
plt.figure(figsize=(12,7))
sns.countplot(df['count_https'],hue="label", saturation=0.80, palette="Set1",data=df)
plt.ylim((0,1500))
plt.xlabel("Nombre de https",fontsize=15)
plt.ylabel("Number of URLs",fontsize=15)


#countplot de la variable count-www
plt.figure(figsize=(12,7))
sns.countplot(df['count-www'], saturation=0.80, palette="Set1")
plt.ylim(0,1500)
plt.xlabel("Le nombre de WWW",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=18)


#countplot de la variable count-www en fonction de la variable cible label
plt.figure(figsize=(12,7))
sns.countplot(df['count-www'],hue='label', saturation=0.80, palette="Set1", data=df)
plt.ylim(0,1500)
plt.xlabel("Nombre de points",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=15)


# countplot de la varible count. en fonction de la variable cible label
plt.figure(figsize=(14,7))
sns.countplot(df['count.'],hue='label', saturation=0.80, palette="Set1", data=df)
plt.ylim(0,1500)
plt.xlabel("Nombre de points",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=15)


#countplot de la variable cout@ en fonction de la variable label
plt.figure(figsize=(14,7))
sns.countplot(df['count@'],hue='label', saturation=0.80, palette="Set1", data=df)
plt.ylim(0,1500)
plt.xlabel("Nombre de @",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=15)


#countplot de la variable http_Domain en fonction de la variable label
plt.figure(figsize=(14,7))
sns.countplot(df['http_Domain'],hue='label', saturation=0.80, palette="Set1", data=df)
plt.ylim(0,1500)
plt.xlabel("Nombre de http dans la partie 'nom du domaine'",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=15)


#countplot de la variable https_Domain en fonction de la variable label
plt.figure(figsize=(14,7))
sns.countplot(df['https_Domain'],hue='label', saturation=0.80, palette="Set1", data=df)
plt.ylim(0,1500)
plt.xlabel("Nombre de https dans la partie 'nom du doamine' ",fontsize=15)
plt.ylabel("Nombre d'URLs",fontsize=15)
