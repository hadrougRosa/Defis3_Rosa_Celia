from math import *
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import sys
import os

file = open('Partie_Analyse_Statistiques/passwords.txt')
totalLength = 0
n = 0
listemots = []
smallwords = []
longwords = []
listechar = {}
listechar2 = {}
nbmotslen = {}
nbcharlen = {}
lenlist = []
var = 0
ecart = 0
avgalpha = 0
avgnum = 0
avgnonalnum = 0

for mot in file:
	totalLength += len(mot.strip())
	n += 1
	listechar2 = {}
	onlychar = True
	onlynum = True
	onlyspe = True
	alpha = 0
	num = 0
	nonalnum = 0
	for char in mot.strip():
		if char not in listechar:
			listechar[char] = 1
		else:
			listechar[char] += 1
		if char not in listechar2:
			listechar2[char] = 1
		else:
			listechar2[char] += 1
		if char.isalpha() == False:
			onlychar = False
		else:
			alpha += 1
		if char.isnumeric() == False:
			onlynum = False
		else:
			num += 1
		if onlychar == False and onlynum == False:
			nonalnum += 1
		else:
			onlyspe = False
	listemots.append([mot.strip(), len(mot.strip()), onlychar, onlynum, listechar2, alpha, num, nonalnum, onlyspe])
	lenlist.append(len(mot.strip()))
	if len(mot.strip()) < 3 :
		smallwords.append(mot)
	if len(mot.strip()) > 20 :
		longwords.append(mot)
#avgLength = totalLength/n

lenlist.sort()
for l in lenlist:
	if l not in nbmotslen:
		nbmotslen[l] = 1
	else:
		nbmotslen[l] += 1

nbMots = n


nbOnlyChar = 0
nbOnlyNum = 0
nbMots = n
ecart = lenlist[-1] - lenlist[0]
for i in listemots:
	if i[2] == True:
		nbOnlyChar += 1
	if i[3] == True:
		nbOnlyNum += 1

	#x = i[1]-avgLength
	#var = (float(x)*x)/n
	avgalpha += i[5]
	avgnum += i[6]
	avgnonalnum += i[7]
ecarttype = sqrt(var)
avgalpha = avgalpha/nbMots
avgnum = avgnum/nbMots
avgnonalnum = avgnonalnum/nbMots

print("nombre de mots : " + str(nbMots))
#print("taille moyenne des mots : " + str(avgLength))
print("nombre de mots onlychar : " + str(nbOnlyChar))
print("nombre de mots onlynum : " + str(nbOnlyNum))
print("nombre de caracteres differents : " + str(len(listechar)))
print("nombre moyen de caracteres alphabetiques : " + str(avgalpha))
print("nombre moyen de caracteres numeriques : " + str(avgnum))
print("nombre moyen de caracteres non alphanumeriques : " + str(avgnonalnum))


# -----------

lists = sorted(listechar.items(), key=lambda x: x[1])
x, y = zip(*lists)
y = [float(i)/n for i in y]
y_pos = np.arange(len(x))

#fig= plt.figure(figsize=(20,10))
plt.bar(y_pos, y, align='center', alpha=1, color='green')
plt.title("Analyse 2")
plt.xticks(y_pos, x, rotation = 0)
plt.ylabel('frequence')
plt.title('Frequence dapparition de chaque caractere')
plt.margins(x=0)
plt.show()


	
	
