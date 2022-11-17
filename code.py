#fonction de calcul des règles
from mlxtend.frequent_patterns import association_rules


#importation des données
import pandas
D = pandas.read_table("market_basket.txt",delimiter="\t",header=0)


#10 premières lignes
print(D.head(10))

#vérification des dimensions
print(D.shape)
TC= pandas.crosstab(D.ID,D.Product)
print(TC.iloc[:30,:3])


#importation de la fonction apriori
from mlxtend.frequent_patterns import apriori

#itemsets frequents
freq_itemsets = apriori(TC,min_support=0.025,max_len=4,use_colnames=True)

#affichage des 15 premiersitemsets 
print(freq_itemsets.head(15))

#fonction de test
def is_inclus(x,items):return items.issubset(x)

#recherche des index des itemsets comprenant le produit ‘Aspirin’.
import numpy
id = numpy.where(freq_itemsets.itemsets.apply(is_inclus,items={'Aspirin'}))
print(freq_itemsets.loc[id])
id = numpy.where(freq_itemsets.itemsets.apply(is_inclus,items={'Aspirin','Eggs'}))
print(freq_itemsets.loc[id])

#génération des règles à partir des itemsets fréquents
regles = association_rules(freq_itemsets,metric="confidence",min_threshold=0.75)
#5 "premières" règles
print(regles.iloc[:5,:])
#affichage des règles avec un LIFT supérieur ou égal à 7
myRegles = regles.loc[:,['antecedents','consequents','lift']]
print(myRegles[myRegles['lift'].ge(7.0)])
#filtrer les règles menant au conséquent {‘2pct_milk’}
print(myRegles[myRegles['consequents'].eq({'2pct_Milk'})])