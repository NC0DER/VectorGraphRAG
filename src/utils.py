import re
import nltk
import string

english_stopwords = set(
"""
a about above across after afterwards again against all almost alone along
already also although always am among amongst amount an and another any anyhow
anyone anything anyway anywhere are around as at

back be became because become becomes becoming been before beforehand behind
being below beside besides between beyond both bottom but by

call can cannot ca could

did do does doing done down due during

each eight either eleven else elsewhere empty enough even ever every
everyone everything everywhere except

few fifteen fifty first five for former formerly forty four from front full
further

get give go

had has have he hence her here hereafter hereby herein hereupon hers herself
him himself his how however hundred

i if in indeed into is it its itself

keep

last latter latterly least less

just

made make many may me meanwhile might mine more moreover most mostly move much
must my myself

name namely neither never nevertheless next nine no nobody none noone nor not
nothing now nowhere

of off often on once one only onto or other others otherwise our ours ourselves
out over own

part per perhaps please put

quite

rather re really regarding

same say see seem seemed seeming seems serious several she should show side
since six sixty so some somehow someone something sometime sometimes somewhere
still such

take ten than that the their them themselves then thence there thereafter
thereby therefore therein thereupon these they third this those though three
through throughout thru thus to together too top toward towards twelve twenty
two

under until up unless upon us used using

various very very via was we well were what whatever when whence whenever where
whereafter whereas whereby wherein whereupon wherever whether which while
whither who whoever whole whom whose why will with within without would

yet you your yours yourself yourselves
""".split()
)

contractions = ["n't", "'d", "'ll", "'m", "'re", "'s", "'ve"]
english_stopwords.update(contractions)

for apostrophe in ["‘", "’"]:
    for stopword in contractions:
        english_stopwords.add(stopword.replace("'", apostrophe))


def tokenize_sentences(text: str) -> list[str]:
    """
    Function which tokenizes a text into a list of sentences.

    Arguments
    ---------
    text: text to be tokenized (str).

    Returns
    -------
    <object>: the list of sentences (list[str]).
    """
    sentences = nltk.sent_tokenize(text, language = 'english')
    return [sentence.strip() for sentence in sentences]


def remove_punctuation(term: str) -> str:
    """
    Function which removes punctuation characters from a term.

    Arguments
    ---------
    term: term to remove punctuation from (str).

    Returns
    -------
    <object>: the term without punctuation (str).
    """
    return term.translate(str.maketrans('', '', string.punctuation))


def remove_stopwords(text: str) -> str:
    """
    Function which removes stopwords from a text.

    Arguments
    ---------
    text: text to remove stopwords from (str).

    Returns
    -------
    <object>: the text without stopwords (str).
    """
    terms = [
        term 
        for term in text.split() 
        if remove_punctuation(term) not in english_stopwords
    ]
    return ' '.join(terms)


def extract_ngrams(text: str, n: int = 3) -> list[tuple[str]]:
    """
    Function which extracts n-grams from a text up to a certain length.
    
    Arguments
    ---------
    text: The text to extract n-grams from (str).
    n: The maximum n-gram length (int).

    Returns
    -------
    n_grams: the list of unique n-grams (list[tuple[str]]). 

    """

    # Extract n-grams of varying length up to n.
    n_grams = [
        n_gram
        for i in range(1, n+1)
        for n_gram in nltk.ngrams(
            map(remove_punctuation, text.split()), i
        )
    ]
    return n_grams


def extract_search_terms(text: str) -> list[str]:
    """
    Function that extracts search terms (cleaned n-grams) from text.

    Parameters
    -----------
    text: the input text to extract search terms from (str).

    Returns
    --------
    search_terms: the search terms (list[str]).
    """
    
    # Lowercase the text and remove its stopwords.
    cleaned_text = remove_stopwords(text.lower())

    # Tokenize the text into sentences.
    sentences = tokenize_sentences(cleaned_text)

    # Extract the list of unique n_grams from each sentence.
    n_grams = list({' '.join(n_gram) for sentence in sentences for n_gram in extract_ngrams(sentence, n = 3)})

    return n_grams


def extract_letter_from_generated_answer(answer: str, possible_answers: str, letter_range: str) -> str:
    """
    Extracts the answer letter from the generated answer using a letter range (e.g., A-D).

    Parameters
    -----------
    answer (str): The answer text from which the letter should be extracted.
    possible_answers (str): A list of all the possible answers.

    Returns:
    -----------
    <object>: The extracted letter if found, otherwise 'No match' (str).
    """

    # Ensure extra whitespaces are removed.
    answer = ' '.join(answer.split())

    # Separate options into a list
    options = possible_answers.split('\n\n')
    options_without_letter = [option[3:] for option in options]

    # Example Cases: "A", "B", "A.", "B:", "(A)", "**C"
    if len(answer) < 4:
        if (match := re.search(rf'\b[{letter_range}]\b', answer)):
            return match.group(0)

    # Example Cases: 'A. <answer_text>', 'B. "<answer_text>"'
    elif answer in options:
        return answer[0]

    # Example Cases: '... answer is: A\b', '... answer is C\b', '... Answer is:B\b', '...Answer **C\b', '...answer is (A\b' ...
    elif (match := re.search(rf'[aA]answer( is)?[:.]?\s?(\*+)?[\'\"]?[\(]?\s?[{letter_range}]\b', answer)):
        return match.group(0)[-1]

    # The models sometimes list all the options and repeat the best one without explicitly mentioning it being the best answer.
    # This works even if the answer is at the end of the model's generated output.
    elif (matches := re.findall(rf'\b[{letter_range}][.:\)\'\"(\*+)]', answer)):
        # Remove special character after the letter.
        matches = [match[0] for match in matches]
        letter = max(matches, key = matches.count)
        return letter

    # In this extreme case the model auto-completes the question with the answer without mentioning the letter.
    else:
        for option in options_without_letter:
            if option.lower() in answer.lower():
                return options[options_without_letter.index(option)][0]

    return 'No match'
