
from brill_tagger_trainer import InitialTagger, PatchLearner
import json
import re
import sys
def load_model_from_file(dict_path, patch_path):
    """
    Load a trained POS tagging model from JSON files.
    Args:
        dict_path (str): Path to the most common POS dictionary, e.g., "model_trained_results/most_common_pos_dict.json"
        patch_path (str): Path to the learned patch rules, e.g., "model_trained_results/bestpatches.json"
    Returns:
        tuple:
            - most_common_pos_dict (dict)
            - patches_data (list[dict])
    """
    with open(dict_path, "r", encoding="utf-8") as f:
        most_common_pos_dict = json.load(f)
    with open(patch_path, "r", encoding="utf-8") as f:
        patches_data = json.load(f)["patches"]
    return most_common_pos_dict, patches_data


def add_func_to_patches(learner: PatchLearner, patches: list):
    """
    Restore the function pointer for each patch's template by assigning
    the correct template function from PatchLearner.templates.

    Why this is necessary:
        When patches are saved to a JSON file, Python function objects
        (like templates' functions) cannot be serialized. As a result, the
        'func' field inside each patch's 'template' is lost during saving.
        This function re-links each patch's 'template' to its corresponding
        function implementation using its 'name'.

    Args:
        learner (PatchLearner): The PatchLearner instance that holds all template functions.
        patches (list): A list of patch dictionaries loaded from a saved JSON file.
    Example:
        A patch loaded from disk may look like this:
            {
                "from_tag": "V",
                "to_tag": "VINF",
                "template": { "name": "prev_tag" },
                ...
            }
        This function will find the real 'prev_tag' function in learner.templates
        and assign it to patch["template"]["func"].
    """
    name_to_template = {t["name"]: t for t in learner.templates}
    for patch in patches:
        template_name = patch["template"]["name"]
        patch["template"]["func"] = name_to_template[template_name]["func"]
    learner.patches = patches


def tag_my_sentence(sentence: str, pos_dict: dict, patches: list) -> str:
    """
    Tag a sentence using the most common POS dictionary and a list of learned patches.
    Args:
        sentence (str): A raw sentence without POS tags. Example: "Le chat dort."
        pos_dict (dict): A dictionary mapping each word to its most common POS tag.
        patches (list[dict]): A list of learned patch rules (with template names but no functions).

    Returns:
        str: The tagged sentence, where each word is followed by its POS tag.
             Example: "Le/DET chat/NC dort/V ./PONCT"
    """
    # --- Text Preprocessing ---
    # Add space after contractions like l'homme → l' homme
    sentence = re.sub(r"(\w)'(\w)", r"\1' \2", sentence)

    # Add space around punctuation
    punct = r"""!\"#$%&()*+,-./:;<=>?@[\\\]^_`{|}~…«»"“”‘’—"""
    sentence = re.sub(f"([{re.escape(punct)}])", r" \1 ", sentence)

    # Normalize whitespace
    sentence = re.sub(r"\s+", " ", sentence).strip()

    #Use the initial tagger to assign the most common tag to each word
    tagger = InitialTagger(pos_dict)
    words, tags = tagger._tag_sentence(sentence)
    tagged = list(zip(words, tags))  # [("Le", "DET"), ("chat", "NC"), ...]

    #Initialize the learner and assign template functions to each patch
    learner = PatchLearner()
    add_func_to_patches(learner, patches)

    patched = learner._apply_patches([tagged])[0]

    return " ".join([f"{word}/{pos}" for word, pos in patched])



def show_help():
    """Afficher le guide d'utilisation complet   ONLINE HELP"""
    print("\n" + "=" * 50)
    print("AIDE - Brill Tagger")
    print("=" * 50)
    print(
        "\nCe système permet d’étiqueter du texte en français avec un étiqueteur morphosyntaxique basé sur les règles (Brill Tagger).")
    print("Vous pouvez soit utiliser un modèle pré-entraîné, soit entraîner le vôtre avec un corpus compatible.\n")

    print("Fonctionnalités principales :")
    print("1. Utiliser le modèle pré-entraîné pour l’étiquetage")
    print("2. Entraîner un nouveau modèle avec un corpus français au format Brown, ou en utilisant le fichier fourni 'sequoia-9.2.fine.brown'")
    print("3. Afficher ce guide d'aide")
    print("4. Quitter le programme\n")

    print("FORMAT DU CORPUS (format Brown) :")
    print("- Une phrase par ligne")
    print("- Chaque mot est annoté sous la forme mot/POS")
    print("  Exemple : Le/DET chat/NC mange/V la/DET souris/NC ./PONCT\n")

    print("ÉTAPES D’UTILISATION :")
    print("1. Téléchargez le dossier 'projet_brill_tagger' et lancez le programme avec : python interface.py")
    print("2. Choisissez une des options du menu")
    print("3. Pour l'entraînement :")
    print("   - Fournissez le chemin vers un corpus compatible")
    print("   - Le programme entraîne un modèle et affiche une courbe de précision (si matplotlib est installé)")
    print("   - Les fichiers du modèle sont enregistrés dans 'model_trained_results/'")
    print("4. Pour l’étiquetage :")
    print("   - Entrez un texte (ex : Le chat dort.)")
    print("   - Le système affiche la phrase étiquetée (ex : Le/DET chat/NC dort/V ./PONCT)")
    print("   - Tapez 'back' pour revenir ou 'exit' pour quitter\n")

    print("TRAITEMENT AUTOMATIQUE :")
    print("Vous pouvez entrer du texte librement, le système s’occupe du traitement et du formatage automatique.\n")

    print("MODÈLE PRÉ-ENTRAÎNÉ :")
    print("Le répertoire 'pretrained_model/' contient un modèle basé sur le corpus Sequoia :")
    print("  - pretrained_pos_dict.json : dictionnaire morphosyntaxique")
    print("  - pretrained_patches.json : règles de transformation\n")

    print("MODÈLES ENREGISTRÉS APRÈS ENTRAÎNEMENT :")
    print("  - model_trained_results/most_common_pos_dict.json")
    print("  - model_trained_results/best_patches.json\n")

    print("DÉPENDANCES :")
    print("  - Python 3.12 ou version ultérieure")
    print("  - Bibliothèque 'matplotlib' ")
    print("    À installer avec : pip install matplotlib\n")

    print("DOCUMENTATION :")
    print("  - Consultez 'rapport.pdf' pour des détails techniques sur l’algorithme et l’implémentation.\n")

    print("=" * 50)
    print("Tapez 'back' pour revenir au menu principal")
    print("=" * 50 + "\n")


def main():

    while True:
        print("\n" + "=" * 50)
        print("Bienvenue dans le Brill Tagger !")
        print("=" * 50)
        print("Veuillez choisir une option :")
        print("1. Utiliser le modèle pré-entraîné pour l’étiquetage")
        print("2. Entraîner un nouveau modèle avec un corpus français au format Brown, "
              "ou en utilisant le fichier fourni 'sequoia-9.2.fine.brown' ")
        print("3. Afficher l’aide")
        print("4. Quitter le programme")

        choice = input("Votre choix (tapez: 1-4) : ").strip()

        if choice == "1":  # Utiliser modèle pré-entraîné
            try:
                pos_dict, patches = load_model_from_file(
                    "pretrained_model/pretrained_pos_dict.json",
                    "pretrained_model/pretrained_patches.json"
                )
                print("✓ Modèle pré-entraîné chargé avec succès !")
            except Exception as e:
                print(f"Erreur lors du chargement du modèle : {e}")
                continue

            while True:
                sent = input("\nEntrez votre texte à étiqueter (tapez 'back' pour revenir au menu) : ").strip()
                if sent.lower() == "back":
                    break
                if sent.lower() == "exit":
                    sys.exit(0)

                try:
                    tagged = tag_my_sentence(sent, pos_dict, patches)
                    print("\nRésultat de l’étiquetage :", tagged)
                except Exception as e:
                    print(f"Erreur pendant l’étiquetage : {e}")

        elif choice == "2":  # Entraîner un modèle
            from brill_tagger_trainer import train_from_corpus

            path = input("\nEntrez le chemin vers votre corpus (ex. C:\\Users\\username\\corpus.brown) : ").strip()
            if path.lower() == "exit":
                sys.exit(0)
            if path.lower() == "back":
                continue

            import os
            if not os.path.exists(path):
                print(f"Erreur : chemin introuvable - {path}")
                continue

            print("Comment souhaitez-vous entraîner votre tagger's model ?")
            print("1. Une division aléatoire du corpus (5 fois) pour choisir le meilleur modèle ")
            print("2. Une division aléatoire (1 fois, seed = 42) pour l'entraîner directement ")
            print()
            print("Attention : le processus d'entraînement est visualisé. ")
            print("Une fois l'apprentissage des règles terminé, un graphique en courbes s'affichera automatiquement. ")
            print("Vous devrez le fermer pour poursuivre l'exécution du programme.")
            print()
            split_choice = input("Votre choix (Tapez 1 ou 2):  ").strip().lower()
            if split_choice == "exit":
                sys.exit(0)
            if split_choice == "back":
                continue

            use_random = (split_choice == "1")

            try:
                print("\nEntraînement en cours, veuillez patienter...")
                result = train_from_corpus(path, use_random_split_5times=use_random, show_plot=True)
                pos_dict, patches, _ = result.values()

                pos_dict, patches = load_model_from_file(
                    "model_trained_results/most_common_pos_dict.json",
                    "model_trained_results/bestpatches.json"
                )
                print("✓ Modèle entraîné et chargé avec succès !")
            except Exception as e:
                print(f"Erreur pendant l’entraînement : {e}")
                continue

            while True:
                sent = input("\nEntrez une phrase à étiqueter (tapez 'back' pour revenir au menu) : ").strip()
                if sent.lower() == "back":
                    break
                if sent.lower() == "exit":
                    sys.exit(0)



                try:
                    tagged = tag_my_sentence(sent, pos_dict, patches)
                    print("\nRésultat de l’étiquetage :", tagged)
                except Exception as e:
                    print(f"Erreur pendant l’étiquetage : {e}")

        elif choice == "3":  # Aide
            show_help()
            while True:
                cmd = input("Tapez 'back' pour revenir au menu principal : ").strip().lower()
                if cmd == "back":
                    break
                if cmd == "exit":
                    sys.exit(0)

        elif choice == "4" or choice.lower() == "exit":  # Quitter
            print("Merci d’avoir utilisé le système. Au revoir !")
            sys.exit(0)

        else:
            print("Option invalide. Veuillez entrer un chiffre entre 1 et 4.")
            print("Astuce : entrez '3' pour afficher l’aide.")


if __name__ == "__main__":
    main()
