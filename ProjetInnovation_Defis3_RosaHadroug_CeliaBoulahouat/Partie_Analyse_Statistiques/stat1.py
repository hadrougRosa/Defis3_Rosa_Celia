#*****Celia Boulahouat & Rosa Hadroug*****

from typing import Sequence
from matplotlib import pyplot as plt


file = open("Partie_Analyse_Statistiques/passwords.txt", "r", encoding='utf8', errors='ignore')
print('**Performance des Mots de passe**')
court=0
moy=0
fort=0
for x in file :
    if len(x)<5 :
        court = court+1
    elif len(x) in range(5,10)  :
        moy = moy+1
    elif len(x) > 10:
        fort = fort + 1
print('Mots de passe très court',court)
print('Mots de passe moyen',moy)
print('Mots de passe fort',fort)

#---- Graphe de Performance mdp-----
robust=[court,moy,fort]
type=['Court','Moyen','Fort']
plt.pie(robust,labels = type, autopct = '%2.1f%%' )
plt.title("Analyse 3")
plt.show()

