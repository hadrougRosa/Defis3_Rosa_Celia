#*****Celia Boulahouat & Rosa Hadroug*****

from string import printable
import matplotlib.pyplot as plt
import numpy as np

def calculMoy (f,list):
    line = f.readline()
    line = str(line)
    tmp = 0
    c = 0
    while line:
        c+= 1
        list.append(len(line) - 1)
        line = f.readline()
        tmp += len(line)
    print("Le nombre de carecteres dans le corpus: ",tmp)
    x = 0
    som = 0
    taille = int(len(list))
    while x < taille:
        som = som + list[x]
        x = x + 1
    f.close()
    resultat = som / taille

    return resultat


#Renvoie le nbre doccurence de chaque caractere dans le texte(u)    
def nbr_occur(u):
    h = {}
    for a in u:
        if a in h: h[a] += 1
        else: h[a] = 1
    return h


def repetiotionChar(f, list):
    # repitition
    count = 0
    RepetitionNbr = []
    RepetitionChar = []
    taille = int(len(list))
    for i in range(0, 47):
        for j in range(0, taille):
            if list[j] == i:
                count = count + 1
        RepetitionNbr.append(i)
        RepetitionChar.append(count)
        count = 0
    #print("**",RepetitionChar)

    
    plt.bar(RepetitionNbr, RepetitionChar, color='green', label="Taille des mots de passe")
    plt.title("Analyse 1")
    plt.xlabel('x Label')
    plt.xlabel('Taille')
    plt.legend(facecolor='gray')
    plt.show()
    f.close()
    return RepetitionNbr,RepetitionChar


#-----Lecture de mon corpus-----
f = open('Partie_Analyse_Statistiques/passwords.txt')
montexte= f.read()

#------Main------
chars = sorted(list(set(montexte)))
n_vocab = len(chars)
n_chars = len(montexte)
f.close()
h = nbr_occur(montexte)
print("Frequence dapparition de chaque caractere \n ",h) 

print("Total Vocab: ", n_vocab)

list = []
fichier = open("Partie_Analyse_Statistiques/passwords.txt", "r", encoding='utf8', errors='ignore')
moyenne = calculMoy(fichier,list)
print("Longueur moyenne des carecteres dans les mot de passe: ", moyenne)
fichier.close()


    