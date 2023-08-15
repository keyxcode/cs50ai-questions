import nltk
from nltk.tokenize import wordpunct_tokenize
import sys
import os
import string
import math

FILE_MATCHES = 1
SENTENCE_MATCHES = 1

nltk.download("stopwords")
STOP_WORDS = nltk.corpus.stopwords.words("english")
PUNCTUATIONS = string.punctuation


def main():
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {filename: tokenize(files[filename]) for filename in files}
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    files = dict()
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename)) as f:
            files[os.path.splitext(filename)[0]] = f.read()

    return files


def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    words = wordpunct_tokenize(document)

    processed_words = [
        word.lower()
        for word in words
        if word not in STOP_WORDS and all(letter not in PUNCTUATIONS for letter in word)
    ]

    return processed_words


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    num_documents = len(documents)

    words_from_documents = list(documents.values())
    word_to_frequency = dict()
    for idx, words in enumerate(words_from_documents):
        unique_words = (word for word in words)
        words_from_remaining_documents = words_from_documents[idx:]
        for word in unique_words:
            if word in word_to_frequency:
                continue
            word_to_frequency[word] = 0
            for words_from_other_document in words_from_remaining_documents:
                if word in words_from_other_document:
                    word_to_frequency[word] += 1

    common_word_to_frequency = {k: v for k, v in word_to_frequency.items() if v != 0}
    common_word_to_idf = {
        k: math.log(num_documents / v) for k, v in common_word_to_frequency.items()
    }

    # with open("words.txt", "w") as f:
    #     for word in common_word_to_idf:
    #         f.write(f"{word}:{common_word_to_idf[word]}\n")

    return common_word_to_idf


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    raise NotImplementedError


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
