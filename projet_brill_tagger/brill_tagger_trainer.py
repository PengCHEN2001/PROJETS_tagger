import math
import os
import re
from collections import defaultdict
from collections import Counter
import matplotlib.pyplot as plt
import random
import copy
import json


class CorpusProcessor:
    """
    Process a tagged corpus, including loading, splitting, and cleaning.
    """

    def __init__(self, input_path):
        self.path = input_path

    def _load_corpus_lines(self):
        """
        Load all non-empty lines from the corpus file.
        Returns:
            list[str]: A list of non-empty, stripped lines from the file.
            ["Le/DET chat/NC dort/V.", s2, s3....]
        """
        with open(self.path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f.readlines() if line.strip()]

    def _get_corpus_parts(self, random_split_5times=False, seed = 42):
        """
        Split the corpus into training (90%), development (5%), and test (5%) sets.
        Args:
            random_split_5times (bool): Whether to set the random seed for shuffle before splitting.
        Returns:
            tuple[list[str], list[str], list[str]]: (train_corpus, dev_corpus, test_corpus)
        """
        sentences = self._load_corpus_lines()

        # shuffle the sentence order randomly
        if random_split_5times:
            random.shuffle(sentences)
        # use a fixed random seed to shuffle the sentences
        # This ensures the split is reproducible (same result every run)
        else:
            random.seed(seed)
            random.shuffle(sentences)
            
            

        cutoff_train = math.floor(len(sentences) * 0.9)
        cutoff_dev = math.floor(len(sentences) * 0.95)

        train_corpus = sentences[:cutoff_train]
        dev_corpus = sentences[cutoff_train:cutoff_dev]
        test_corpus = sentences[cutoff_dev:]

        return train_corpus, dev_corpus, test_corpus

    def _corpus_without_pos(self, corpus):
        """
        Remove POS tags from a tagged corpus.
        Args:
            corpus (list[str]): A list of POS-tagged sentences,
            where each sentence is a string in the format "word1/POS1 word2/POS2 ...".
        Returns:
            list[str]: A list of cleaned sentences containing only the words without POS tags. [Le chat dort, s2, s3...]
        """
        corpus_without_pos = [' '.join(re.findall(r"(\S+)/", sentence)) for sentence in corpus]
        return corpus_without_pos


class MostCommonPosBuilder:
    """
    Build a dictionary mapping each word to its most frequent POS tag in the training corpus.
    """

    def __init__(self):
        """
        Initialize the internal dictionary for storing word-to-POS counts.
        default dict(<class 'collections.Counter'>, {'dog': Counter({'VB': 2, 'NN': 1}),'cat': Counter({'NN': 3})})
        """
        self.word_pos_counts = defaultdict(Counter)

    def _most_common_pos(self, train_corpus):
        """
        Extract most common POS tags for each word from the training corpus.
        Args:
            train_corpus (list[str]): A list of POS-tagged sentences, where each sentence is
            formatted as "word1/POS1 word2/POS2 ...".
        Returns:
            dict[str, str]: A dictionary mapping each word to its most frequent POS tag.
            Example: {'chat': 'NC', 'mange': 'V', ...}
        """
        for sentence in train_corpus:
            for token in sentence.split():
                if '/' in token:
                    word, pos = token.rsplit('/', 1)
                    self.word_pos_counts[word][pos] += 1
                # word_pos_counts: defaultdict(<class 'collections.Counter'>, {'dog': Counter({'VB': 2, 'NN': 1}),'cat': Counter({'NN': 3})})
                # .word_pos_counts[word] obtained the dict 1 's value 'counter', which is the second 'dict', so word_pos_counts[word][pos] is equivalent to accessing the value of the second dictionary

        most_common_pos_dic = {}
        for word, counts in self.word_pos_counts.items():
            # Select the most common POS
            most_common_pos_dic[word] = counts.most_common(1)[0][0]
            # .most_common(1) gives the highest frequency [('NN', 3)]， first [0] gives ('NN', 3)，second [0] gives 'NN'

        return most_common_pos_dic


class InitialTagger:
    """
    Tags a cleaned corpus using most_common_pos_dic.
    Unknown words are tagged using fallback rules (Morphosyntactic rules).
    """

    def __init__(self, most_common_pos_dic):
        """
        Initialize with a dictionary of most common POS tags.
        """
        self.most_common_pos_dic = most_common_pos_dic

    def _tag_sentence(self, sentence):
        """
        Tag a single sentence using the most_common_pos_dic or fallback rules.
        Args:
            sentence (str): One sentence without POS tags. Example: "Le chat dort."
        Returns:
            tuple: (words_list, pos_list)
            Example: (["Le", "chat", "dort"], ["DET", "NC", "V"])

        """
        words_in_sentence = sentence.split()
        pos_list = []

        for word in words_in_sentence:
            if word in self.most_common_pos_dic:
                predicted_pos = self.most_common_pos_dic[word]
            else:
                predicted_pos = self.basic_morphosyntactic_rules(word)

            pos_list.append(predicted_pos)

        return words_in_sentence, pos_list

    def _tag_corpus(self, corpus_without_pos):
        """
        Tag an entire cleaned corpus.
        Args:
            corpus_without_pos (list[str]): List of sentences without POS tags.
        Returns:
            list[list[tuple[str, str]]]: List of tuples (words, pos) for each sentence.
            Example: [[("Le", "DET"), ("chat", "NC"), ("dort", "V")], [s2]]
        """
        initial_tagged_corpus = []

        for sentence in corpus_without_pos:
            words_list, pos_list = self._tag_sentence(sentence)
            tagged_sentence = list(zip(words_list, pos_list))
            initial_tagged_corpus.append(tagged_sentence)

        return initial_tagged_corpus

    def basic_morphosyntactic_rules(self, word):
        """
        Fallback rules (Morphosyntactic rules) to guess POS for unknown words.
        Args:
            word (str): The unknown word.
        Returns:
            str: The predicted POS tag (e.g., 'NC', 'V', 'ADJ', 'VPP', 'UNK', etc.).
        """

        if re.fullmatch(r"\d+([.,]\d+)?", word):  # number
            return "NC"
        if word.endswith(("age", "tion", "ment", "isme", "eur", "ité")):
            return "NC"
        # limit the length to avoid words like "En" are known by NPP
        elif len(word) > 3 and word[0].isupper():
            return "NPP"
        elif word.endswith(("tif", "tive", "eux", "able", "ible", "ique")):
            return "ADJ"
        elif word.endswith(("is", "it")):
            return "V"
        elif word.endswith(("er", "ir", "re", "oir", "aire", "dre", "tre", "uire", "enir", "aître",)):
            return "VINF"
        elif word.endswith(("é", "ée")):
            return "VPP"
        else:
            # Use the most frequent tag ("NC" 15684) in the whole corpus instead of "UNK"
            return "NC"


class PatchLearner:
    """
    Learn transformation rules (patches) to correct POS tags
    in a tagged corpus based on a set of predefined templates.
    """

    def __init__(self):
        """
        Initialize the PatchLearner with the templates and an empty patch list.
        """
        self.templates = [
            {"name": "prev_tag", "desc": "PREVIOUS WORD TAG IS", "func": self._prev_tag},
            {"name": "next_tag", "desc": "NEXT WORD TAG IS", "func": self._next_tag},
            {"name": "prev_two_tag", "desc": "PREVIOUS TWO WORD TAGS ARE", "func": self._prev_two_tag},
            {"name": "next_two_tag", "desc": "NEXT TWO WORD TAGS ARE", "func": self._next_two_tag},
            {"name": "prev_one_or_two_has_tag", "desc": "EITHER OF PREVIOUS TWO WORDS HAS TAG",
             "func": self._prev_one_or_two_has_tag},
            {"name": "next_one_or_two_has_tag", "desc": "EITHER OF NEXT TWO WORDS HAS TAG",
             "func": self._next_one_or_two_has_tag},
            {"name": "prev_within_3_has_tag", "desc": "ANY OF PREVIOUS THREE WORDS HAS TAG",
             "func": self._prev_within_3_has_tag},
            {"name": "next_within_3_has_tag", "desc": "ANY OF NEXT THREE WORDS HAS TAG",
             "func": self._next_within_3_has_tag},
            {"name": "prev_z_and_next_w", "desc": "PREVIOUS WORD IS Z AND NEXT WORD IS W",
             "func": self._prev_z_and_next_w},
            {"name": "prev_z_and_prev_two_w", "desc": "PREVIOUS WORD IS Z AND PREVIOUS TWO WORDS ARE W",
             "func": self._prev_z_and_prev_two_w},
            {"name": "next_z_and_next_two_w", "desc": "NEXT WORD IS Z AND NEXT TWO WORDS ARE W",
             "func": self._next_z_and_next_two_w},
            {"name": "current_is_cap", "desc": "CURRENT WORD IS CAPITALIZED", "func": self._current_is_cap},
            {"name": "current_not_cap", "desc": "CURRENT WORD IS NOT CAPITALIZED", "func": self._current_not_cap},
            {"name": "prev_is_cap", "desc": "PREVIOUS WORD IS CAPITALIZED", "func": self._prev_is_cap},
            {"name": "prev_not_cap", "desc": "PREVIOUS WORD IS NOT CAPITALIZED", "func": self._prev_not_cap}
        ]
        self.patches = []

    # ========Define the functions that depend on each template======

    # Functions that return str or None : "DET",  'ADJ_NOUN'
    def _prev_tag(self, i, sent):
        return sent[i - 1][1] if i > 0 else None

    def _next_tag(self, i, sent):
        return sent[i + 1][1] if i < len(sent) - 1 else None

    def _prev_two_tag(self, i, sent):
        return f"{sent[i - 2][1]}_{sent[i - 1][1]}" if i >= 2 else None

    def _next_two_tag(self, i, sent):
        return f"{sent[i + 1][1]}_{sent[i + 2][1]}" if i < len(sent) - 2 else None

    # Functions that return List[str] or None : ['DET', 'ADJ']
    def _prev_one_or_two_has_tag(self, i, sent):
        return [sent[i - 1][1]] + ([sent[i - 2][1]] if i > 1 else []) if i > 0 else None

    def _next_one_or_two_has_tag(self, i, sent):
        return [sent[i + 1][1]] + ([sent[i + 2][1]] if i < len(sent) - 2 else []) if i < len(sent) - 1 else None

    def _prev_within_3_has_tag(self, i, sent):
        if i == 0:
            return None
        tags = [sent[j][1] for j in range(max(0, i - 3), i)]
        return tags if tags else None

    def _next_within_3_has_tag(self, i, sent):
        if i >= len(sent) - 1:
            return None
        tags = [sent[j][1] for j in range(i + 1, min(len(sent), i + 4))]
        return tags if tags else None

    # Functions that return Tuple[str, str] or None :  ('voyages', 'les_voyages') or None
    def _prev_z_and_next_w(self, i, sent):
        if i > 0 and i < len(sent) - 1:
            return (sent[i - 1][0], sent[i + 1][0])
        return None

    def _prev_z_and_prev_two_w(self, i, sent):
        return (sent[i - 1][0], f"{sent[i - 2][0]}_{sent[i - 1][0]}") if i >= 2 else None

    def _next_z_and_next_two_w(self, i, sent):
        return (sent[i + 1][0], f"{sent[i + 1][0]}_{sent[i + 2][0]}") if i < len(sent) - 2 else None

    # Functions that return str or None: 'CAP' or 'LOW' or None
    def _current_is_cap(self, i, sent):
        word = sent[i][0]
        return "CAP" if word and word[0].isupper() else None

    def _current_not_cap(self, i, sent):
        word = sent[i][0]
        return "LOW" if word and not word[0].isupper() else None

    def _prev_is_cap(self, i, sent):
        if i == 0:
            return None
        word = sent[i - 1][0]
        return "CAP" if word and word[0].isupper() else None

    def _prev_not_cap(self, i, sent):
        if i == 0:
            return None
        word = sent[i - 1][0]
        return "LOW" if word and not word[0].isupper() else None

    def _parse_corpus(self, corpus):
        """
        Convert the format of the corpus with pos
        Args:
            corpus (list[str]): List of POS-tagged sentences, where each token is in "word/tag" format.
            Example: ["Le/DET chat/NC dort/V.", "Elle/PRO court/VITE/ADV."]
        Return:
            list[list[tuple[str, str]]]: Parsed corpus.
            Example: [[("Le", "DET"), ("chat", "NC"), ("dort", "V")], [s2]]
        """
        parsed = []
        for sent in corpus:
            word_tags = [tuple(word_tag.rsplit('/', 1)) for word_tag in sent.split()]
            parsed.append(word_tags)
        return parsed

    def _learn_patches(self, parsed_tagged, parsed_gold, max_patches2save=30, verbose=True):
        """
        Learn a set of patch rules to iteratively improve POS tagging accuracy.

        This method compares the current tagged sentences (parsed_tagged) against the gold-standard
        tagged sentences (parsed_gold), generating candidate patches based on predefined templates.
        It scores patches by their ability to correct errors without introducing new ones,
        then applies the best patches iteratively until the maximum number is reached or no
        further improvement is possible.

        Args:
            parsed_tagged (list[list[tuple[str, str]]]): The current tagged sentences as a list of
                sentences, where each sentence is a list of (word, predicted_tag) tuples.
            parsed_gold (list[list[tuple[str, str]]]): The gold standard tagged sentences in the
                same format as parsed_tagged, used as the correct reference.
            max_patches2save (int, optional): The maximum number of patches to learn and save.
                Defaults to 30.
            verbose (bool, optional): Whether to print detailed learning progress and summary.
                Defaults to True.
        Returns:
            list[dict]: A list of learned patch dictionaries. Each patch contains:
                - 'from_tag': The original POS tag to be changed.
                - 'to_tag': The target POS tag after applying the patch.
                - 'template': The template object describing the patch condition.
                - 'context_tag': The context information where the patch applies.
                - 'score': The net positive score indicating patch effectiveness.
                - 'corrected': Number of errors corrected by this patch.
                - 'introduced': Number of new errors introduced by this patch.
        """

        #Collect all unique POS tags from the gold (correct) annotations
        tag_candidates = set()
        for sent in parsed_gold:
            for _, tag in sent:
                tag_candidates.add(tag)
        print(f"Candidates of to_tag: {tag_candidates}")

        #Initialize the list of learned patches and a working copy of predicted tags
        self.patches = []
        current_tagged = copy.deepcopy(parsed_tagged)

        #loop to iteratively learn patches
        while len(self.patches) < max_patches2save:
            patch2score = defaultdict(lambda: {"score": 0, "corrected": 0, "introduced": 0})
            patch_infos = {} # Save metadata for each patch

            #Compare prediction vs gold to evaluate all possible patches
            for sent_pred, sent_gold in zip(current_tagged, parsed_gold):
                if len(sent_pred) != len(sent_gold):
                    continue
                for i in range(len(sent_pred)):
                    word_pred, tag_pred = sent_pred[i]
                    _, tag_gold = sent_gold[i]

                    for template in self.templates:
                        context = template["func"](i, sent_pred) # Try to apply the template at position i
                        if context is None:
                            continue

                        for to_tag in tag_candidates:
                            if to_tag == tag_pred:  # Skip if the target to_tag is same as current
                                continue

                            # Create a unique patch name
                            patch_name = f"{template['name']}|{tag_pred}->{to_tag}|{str(context)}"

                            # Initialize patch info if new
                            if patch_name not in patch_infos:
                                patch_infos[patch_name] = {
                                    "from_tag": tag_pred,
                                    "to_tag": to_tag,
                                    "template": template,
                                    "context_tag": context
                                }

                            if tag_pred != tag_gold and to_tag == tag_gold:
                                patch2score[patch_name]["score"] += 1
                                patch2score[patch_name]["corrected"] += 1
                            elif tag_pred == tag_gold and to_tag != tag_gold:
                                patch2score[patch_name]["score"] -= 1
                                patch2score[patch_name]["introduced"] += 1

            #Build a list of patch candidates with positive scores
            patch_candidates = []
            for patch_name, stats in patch2score.items():
                if stats["score"] <= 0:
                    continue
                info = patch_infos[patch_name]
                patch_candidates.append({
                    "from_tag": info["from_tag"],
                    "to_tag": info["to_tag"],
                    "template": info["template"],
                    "context_tag": info["context_tag"],
                    "score": stats["score"],
                    "corrected": stats["corrected"],
                    "introduced": stats["introduced"]
                })

            if not patch_candidates:
                break

            # Sort patches by score (descending), then by corrected count
            patch_candidates.sort(key=lambda x: (-x["score"], -x["corrected"]))

            best_patch = {}
            # verify if the current patch is unique (= hasn’t been used)
            for patch_candidate in patch_candidates:
                if patch_candidate not in self.patches:
                    best_patch = patch_candidate
                    break

            #Save and apply the best patch to the predicted tags
            self._apply_patch(best_patch, current_tagged)
            self.patches.append(best_patch)

            #Print the applied patch's infos and its result(accuracy) of application in the dev_corpus
            acc = compute_accuracy_from_parsed(current_tagged, parsed_gold)
            print(f"[Iteration {len(self.patches)}] Patch applied: {best_patch['template']['name']} | "
                  f"{best_patch['from_tag']} -> {best_patch['to_tag']} | context: {best_patch['context_tag']} | "
                  f"Accuracy: {acc:.5f}")

        #After learning, show final stats
        if verbose:
            print(f"\n=== Total number of learned patches: {len(self.patches)} ===")
            self._print_patch_stats(current_tagged, parsed_gold)

        return self.patches

    def _print_patch_stats(self, parsed_tagged, parsed_gold):
        """
        Print detailed statistics about the applied learned patches.
        Args:
            parsed_tagged (list[list[tuple[str, str]]]): Corpus with predicted tags after patching.
            parsed_gold (list[list[tuple[str, str]]]): Gold-standard corpus for comparison.
        """

        print(
            f"\n=== Final accuracy of tagging dev_corpus after applying patches: {compute_accuracy_from_parsed(parsed_tagged, parsed_gold):.5%} ===")
        print(f"\nTop 20 patches:")
        for i, patch in enumerate(self.patches[:20]):
            print(f"{i + 1:02d}. {patch['from_tag']} → {patch['to_tag']} | "
                  f"{patch['template']['name']}({patch['context_tag']}) | "
                  f"Score: {patch['score']} (+{patch['corrected']}/-{patch['introduced']})")


    def _apply_patch(self, patch, corpus):
        """
        Apply a single patch to a parsed corpus in-place.

        This function iterates over a parsed corpus,
        and modifies in-place all words whose tag matches 'from_tag' and satisfy the
        rule's context condition, by replacing their tag with 'to_tag'
        Args：
            patch (dict): A patch dict
            corpus : [ [(word1, tag1), (word2, tag2), ...], [s2]... ]
        returns:
            None. The function modifies the corpus in place.
        """
        from_tag = patch["from_tag"]
        to_tag = patch["to_tag"]
        template = patch["template"]
        context_tag = patch["context_tag"]
        for sentence in corpus:
            for i, (word, tag) in enumerate(sentence):
                if tag == from_tag and template["func"](i, sentence) == context_tag:
                    sentence[i] = (word, to_tag)

    def _apply_patches(self, initial_tagged_corpus):
        """
        Apply all learned patches to the initial tagged corpus
        Args:
            tagged_corpus : [[("Le", "DET"), ("chat", "NC"), ("dort", "V")], [s2]]
        Return:
            ist[list[tuple[str, str]]]: Parsed corpus.
            Example: [[("Le", "DET"), ("chat", "NC"), ("dort", "V")], [s2]]
        """
        # _apply_patch() modifies the initial_tagged_corpus in-place,
        # so we use deepcopy to avoid changing the original initial tagged corpus.
        new_corpus = copy.deepcopy(initial_tagged_corpus)
        for patch in self.patches:
            self._apply_patch(patch, new_corpus)
        return new_corpus


def compute_accuracy_from_parsed(predicted_parsed, gold_parsed):
    """
    Compute part-of-speech tagging accuracy from parsed corpus format.

    Args:
        predicted_parsed [ [(word1, tag1), (word2, tag2), ...], [s2]... ]
        gold_parsed [ [(word1, tag1), (word2, tag2), ...], [s2]... ]
    Returns:
        float: Accuracy score between 0.0 and 1.0.
        This is the proportion of correctly predicted tags for words that match in both corporas.
    """
    correct = total = 0
    for sent_pred, sent_gold in zip(predicted_parsed, gold_parsed):
        for (w1, t1), (w2, t2) in zip(sent_pred, sent_gold):
            if w1 == w2:
                total += 1
                if t1 == t2:
                    correct += 1
    accuracy = correct / total
    return accuracy if total > 0 else 0.0


def count_pos_tags(corpus):
    """
    Count the frequency of each POS tag in a POS-tagged corpus.
    Args:
        corpus (list[str]): List of sentences with POS tags (e.g., ["Le/DET chat/NC dort/V", ...])
    Returns:
        Counter: A Counter object with POS tags and their frequencies.
        e.g. Counter({'NC': 15684, 'P': 9438,

    """
    pos_counter = Counter()
    for sentence in corpus:
        tokens = sentence.strip().split()  # "Le/DET chat/NC" → ["Le/DET", "chat/NC"]
        for token in tokens:
            _, pos = token.rsplit("/", 1)
            pos_counter[pos] += 1

    print("The frequency of the pos tag in the whole corpus:", pos_counter)
    return pos_counter

def find_errors(current, gold):
    errors = Counter()
    for sent_cur, sent_gold in zip(current, gold):
        for (word_cur, tag_cur), (word_gold, tag_gold) in zip(sent_cur, sent_gold):
            if word_cur == word_gold and tag_cur != tag_gold:
                errors[(tag_cur, tag_gold)] += 1
    return errors

def train_from_corpus(corpus_path: str, use_random_split_5times: bool = False, show_plot: bool = False):
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"Le chemin fourni est invalide : {corpus_path}")
    if not os.path.isfile(corpus_path):
        raise ValueError(f"Ce n'est pas un fichier valide : {corpus_path}")

    processor = CorpusProcessor(corpus_path)
    corpus_lines = processor._load_corpus_lines()
    count_pos_tags(corpus_lines)

    colors = ['r', 'b', 'g', 'm', 'c']
    if show_plot:
        plt.figure(figsize=(8, 6))

    num_runs = 5 if use_random_split_5times else 1
    best_accuracy = -1
    best_run_data = {}

    for run_id in range(num_runs):
        print(f"\n=== Run {run_id + 1} ===")

        # Get training, dev, and test corpora, with optional random shuffling
        train_raw, dev_tagged, test_tagged = processor._get_corpus_parts(random_split_5times=use_random_split_5times)

        # Remove POS tags from the tagged corpus.
        dev_clean = processor._corpus_without_pos(dev_tagged)
        test_clean = processor._corpus_without_pos(test_tagged)

        # Build the most common POS dictionary from the raw training corpus
        builder = MostCommonPosBuilder()
        most_common_pos_dict = builder._most_common_pos(train_raw)

        # Use the dictionary most_common_pos_dict to perform initial tagging
        tagger = InitialTagger(most_common_pos_dict)
        dev_predicted = tagger._tag_corpus(dev_clean)
        test_predicted = tagger._tag_corpus(test_clean)

        # Learn patch rules from dev_predicted vs gold tags(dev_tagged)
        learner = PatchLearner()
        dev_gold_parsed = learner._parse_corpus(dev_tagged)
        errors = find_errors(dev_predicted, dev_gold_parsed)
        print(errors)
        learner._learn_patches(dev_predicted, dev_gold_parsed, max_patches2save=30)


        # Apply all learned patches to the test_predicted
        test_after_patch = learner._apply_patches(test_predicted)
        test_gold_parsed = learner._parse_corpus(test_tagged)
        test_accuracy = compute_accuracy_from_parsed(test_after_patch, test_gold_parsed)
        print(f"=== Final test accuracy after applying all patches: {test_accuracy:.5%} ===")
        errors2 = find_errors(test_predicted, test_gold_parsed)
        print("Type d'erreur avant l'application des patches sur test corpus")
        print(errors2)
        errors3 = find_errors(test_after_patch, test_gold_parsed)
        print("Type d'erreur après l'application des patches sur test corpus")
        print(errors3)


        # Compute initial error rate (before applying any patches)
        initial_accuracy = compute_accuracy_from_parsed(tagger._tag_corpus(test_clean), test_gold_parsed)
        error_rates = [1 - initial_accuracy]
        patch_numbers = [0]
        print(f"=== Initial test error rate (no patches): {error_rates[0]:.5%} ===")

        # Apply patches one by one and record error rate after each
        temp_test = tagger._tag_corpus(test_clean)
        for idx, patch in enumerate(learner.patches):
            learner._apply_patch(patch, temp_test)
            acc = compute_accuracy_from_parsed(temp_test, test_gold_parsed)
            error_rate = 1 - acc
            patch_numbers.append(idx + 1)
            error_rates.append(error_rate)
            print(f"After applying patch {idx + 1} ({patch['template']['name']}): error rate = {error_rate:.5f}")


        # Plot error rate curve
        if show_plot:
            plt.plot(patch_numbers, error_rates, marker='o', linestyle='-', color=colors[run_id],
                     label=f'Run {run_id + 1}')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_run_data = {
                "accuracy": best_accuracy,
                "most_common_pos_dict": most_common_pos_dict,
                "patches": copy.deepcopy(learner.patches)
            }

    if show_plot:
        plt.title('Error Rate vs. Number of Patches Applied in the test corpus')
        plt.xlabel('Number of Patches Applied')
        plt.ylabel('Error Rate (1 - Accuracy)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Ensure the output directory exists
    os.makedirs("model_trained_results", exist_ok=True)

    # Save patches
    patches_for_json = []
    for patch in best_run_data["patches"]:
        patch_copy = copy.deepcopy(patch)
        patch_copy["template"] = {"name": patch["template"]["name"]}
        patches_for_json.append(patch_copy)

    with open("model_trained_results/bestpatches.json", "w", encoding="utf-8") as f_patch:
        json.dump({
            "accuracy": best_run_data["accuracy"],
            "patches": patches_for_json
        }, f_patch, ensure_ascii=False, indent=4)
    print("✔ Saved bestpatches.json")

    # Save dictionary
    with open("model_trained_results/most_common_pos_dict.json", "w", encoding="utf-8") as f_dict:
        json.dump(best_run_data["most_common_pos_dict"], f_dict, ensure_ascii=False, indent=4)
    print("✔ Saved most_common_pos_dict.json")

    print(f"\nBest run accuracy in test_corpus: {best_accuracy:.5%}")
    return best_run_data


# Save dictionary
def main():
    corpus_path = input("Veuillez saisir le chemin du fichier de corpus :\n").strip()
    if not os.path.exists(corpus_path):
        print(f"Le chemin fourni est invalide : {corpus_path}")
        return

    print("Souhaitez-vous une division aléatoire du corpus (5 fois) pour choisir le meilleur modèle (choix1)")
    print("ou une division aléatoire (1 fois, seed = 42) pour l'entraîner directement (choix2) ?")
    choice = input("Votre choix (Tapez 1 ou 2):  ").strip().lower()
    use_random = (choice == "1")

    train_from_corpus(corpus_path, use_random_split_5times=use_random, show_plot=True)


if __name__ == "__main__":
    main()
