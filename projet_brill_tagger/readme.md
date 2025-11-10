# Système d'étiquetage morphosyntaxique Brill Tagger

## INFORMATIONS DU PROJET
- **Noms des éditeurs** : Yiwei ZUO et Peng CHEN  
- **Sujet** : Brill's Tagger  
- **Langue de programmation** : Python 3.12  
- **Date** : 16/06/2025  

## APERÇU DU PROJET
Ce projet implémente un étiqueteur morphosyntaxique basé sur les règles (Brill's tagger) [1], comprenant des fonctionnalités complètes d'entraînement et d'étiquetage. Le système permet à l'utilisateur :

1. D'utiliser un modèle pré-entraîné pour l'étiquetage morphosyntaxique  
2. D'entraîner un nouveau modèle avec un corpus français au format Brown, ou en utilisant le fichier fourni `sequoia-9.2.fine.brown`  
3. D'étiqueter du texte de manière interactive  

## CONFIGURATION REQUISE
- **Python 3.12 ou version ultérieure**  
- **Bibliothèque obligatoire** :
```bash
pip install matplotlib
```
*(pour afficher les courbes de précision pendant l'entraînement)*

## STRUCTURE DES FICHIERS
```
projet_brill_tagger/
├── brill_tagger_trainer.py        
├── model_trained_results/   
│   ├── best_patches.json
│   └── most_common_pos_dict.json
├── pretrained_model/
│   ├── pretrained_patches.json
│   └── pretrained_pos_dict.json
├── corpus.brown                     
├── rapport.pdf                    
├── interface.py                   
└── README.md                      
```

## INSTALLATION ET LANCEMENT

#### Après avoir téléchargé le dossier `projet_brill_tagger` [1], suivez les étapes ci-dessous :

### Étape 1 : Ouvrir le terminal dans le dossier du projet

Dans votre terminal, placez-vous dans le dossier :

```bash
cd chemin/vers/le/dossier/projet_brill_tagger
````

Par exemple sur Windows :

```bash
cd "C:\Users\couple\Desktop\projet_brill'stagger"
```

---

### Étape 2 : Installer les dépendances (matplotlib)

```bash
pip install matplotlib
```

---

### Étape 3 : Préparer le corpus français

Vous pouvez :

* Préparer vous-même un fichier au format **Brown**
* Ou utiliser le fichier fourni : `sequoia-9.2.fine.brown`

**Format attendu :**

* Une phrase par ligne
* Chaque mot est suivi de son étiquette, séparés par une barre oblique `/`

**Exemple :**

```
Le/DET chat/NC mange/V la/DET souris/NC ./PONCT
```

---

### Étape 4 : Lancer le programme principal

```bash
python interface.py
```

## GUIDE D'UTILISATION

### Menu principal :
```
==============================
Bienvenue dans le système d'étiquetage Brill Tagger
==============================
Veuillez choisir :
1. Utiliser le modèle pré-entraîné pour l'étiquetage
2. Entraîner un nouveau modèle avec votre corpus
3. Afficher l'aide
4. Quitter le programme
```

### Option 1 : Étiquetage
- Entrez une phrase à étiqueter
- Résultat affiché sous la forme : `mot/étiquette`
- Tapez `back` pour revenir, `exit` pour quitter

### Option 2 : Entraînement
- Entrez le chemin vers un corpus compatible
- Le programme entraînera un modèle et affichera une courbe de précision
- Le modèle est sauvegardé dans `model_trained_results/`

### Option 3 : Aide
- Affiche les instructions détaillées

## TRAITEMENT AUTOMATIQUE DU TEXTE
Vous pouvez simplement entrer du texte librement, le système convertira automatiquement le format et retournera le texte avec les étiquettes.

## MODÈLE PRÉ-ENTRAÎNÉ
Le répertoire `pretrained_model` contient un modèle entraîné sur le corpus français Sequoia :
- `pretrained_pos_dict.json` : dictionnaire morphosyntaxique
- `pretrained_patches.json` : règles de transformation

## MODÈLES PERSONNALISÉS
Après entraînement, les modèles sont enregistrés dans :
- `model_trained_results/most_common_pos_dict.json`
- `model_trained_results/bestpatches.json`

## FOIRE AUX QUESTIONS
**Q : Pourquoi installer matplotlib ?**  
R : Pour afficher une courbe de précision pendant l'entraînement.

**Q : Quelles langues sont supportées ?**  
R : Le français.

## QUITTER LE PROGRAMME
Tapez `exit` à tout moment pour quitter.

## SUPPORT TECHNIQUE
1. Vérifiez que `matplotlib` est bien installé
2. Assurez-vous que le corpus est au bon format (format Brown)
3. Lisez les messages d'erreur dans le terminal
4. Consultez le fichier `rapport.pdf` pour plus de détails techniques

## BIBLIOGRAPHIE
[1] Brill, Eric. "A simple rule-based part of speech tagger." *Speech and Natural Language: Proceedings of a Workshop Held at Harriman*, New York, February 23-26, 1992. 1992.