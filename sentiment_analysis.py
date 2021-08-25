from transformers import pipeline
from transformers import AutoTokenizer,TFAutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re

def ang_sentiment(text):
    lower_case= text.lower()
    cleaned_text = lower_case.translate(str.maketrans('','',string.punctuation))
    tokenized_words = word_tokenize(cleaned_text,"english")
    
    final_words=[]
    for word in tokenized_words:
        if word not in stopwords.words("english"):
            final_words.append(word)

    result=" ".join(final_words)
    nlp = pipeline('sentiment-analysis')
    return (nlp(result))
    


def fr_sentiment(text):
    lower_case= text.lower()
    cleaned_text = lower_case.translate(str.maketrans('','',string.punctuation))
    tokenized_words = word_tokenize(cleaned_text,"french")

    tokenizer = AutoTokenizer.from_pretrained("tblard/tf-allocine")
    model = TFAutoModelForSequenceClassification.from_pretrained("tblard/tf-allocine")


    final_words=[]
    for word in tokenized_words:
        if word not in stopwords.words("french"):
            final_words.append(word)

    result=" ".join(final_words)
    nlp = pipeline('sentiment-analysis',model=model,tokenizer=tokenizer)
    return (nlp(result))


def ar_sentiment(text):
    arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
    english_punctuations = string.punctuation
    punctuations_list = arabic_punctuations + english_punctuations
    arabic_diacritics = re.compile("""
                                 ّ    | # Tashdid
                                 َ    | # Fatha
                                 ً    | # Tanwin Fath
                                 ُ    | # Damma
                                 ٌ    | # Tanwin Damm
                                 ِ    | # Kasra
                                 ٍ    | # Tanwin Kasr
                                 ْ    | # Sukun
                                 ـ     # Tatwil/Kashida
                             """, re.VERBOSE)
    translator = str.maketrans('', '', punctuations_list)
    text=text.translate(translator)

    text = re.sub(arabic_diacritics, '', text)

    text=re.sub(r'(.)\1+', r'\1', text)
    
    tokenized_words = word_tokenize(text)
    
    sw=stopwords.words('arabic')
    tokens=[i for i in tokenized_words if not i in sw]

    result=" ".join(tokens)
    
    tokenizer = AutoTokenizer.from_pretrained("akhooli/xlm-r-large-arabic-sent")

    model = AutoModelForSequenceClassification.from_pretrained("akhooli/xlm-r-large-arabic-sent")

    nlp = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)

    return (nlp(result))
        
