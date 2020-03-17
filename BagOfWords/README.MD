# Bag of Words 
Am definit functia normalize_data pentru scalarea datelor in functie de norma (standard, minmax, l1, L2) <br>
Am definit o clasa in care am initializat cu ajutorul constructorului vocabularul(cu id unic pentru fiecare cuvant), un vector de cuvinte si numarul acestora.<br>
In metoda buildVocab i-am dat fiecarui cuvant in parte un id unic <br>
In metoda getFeatures am facut o matrice in care am stocat numarul de aparitii al unui cuvant in fiecare fraza <br>
Am obtinut apoi reprezentarile pentru multimea de antrenare si testare iar apoi le-am normalizat folosind norma l2 <br>
Am antrenat apoi SVM-ul pentru a determina daca un text este spam sau nu si apoi am afisat acuratetea si f1-score. <br>
Am afisat apoi top negative si positive

