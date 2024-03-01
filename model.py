import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import fitz
import nltk
import openpyxl
import streamlit as st
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from langdetect import detect



#FUTURES:
#   DISPARAR O APLICATIVO APENAS QUANDO CLICAR NO BOTÃO DE PROCESSAR OK
#   TRAZER A TABELA DE TERMOS FREQUENTES NO MESMO IDIOMA DO TEXTO    OK
#   TRAZER O RESUMO PARA O MESMO IDIOMA DO TEXTO                     OK
#   USAR A FUNÇÃO DO R PARA FAZER A ANÁLISE DE VARIOS SENTIMENTOS E POR NOS CARDS

class TextAnalysis:

    #ENTRADA
    def __init__(self, uploaded_file):
        self.uploaded_file = uploaded_file
        self.text_completo, self.text, self.pages = self.read_uploaded_file()
        self.linguagem = str(detect(self.text_completo))

    def read_uploaded_file(self):
        texto = ""
        pages = 0

        # Verifica se um arquivo foi carregado
        if self.uploaded_file is not None:
            # Lê o conteúdo do arquivo PDF
            with fitz.open(stream=self.uploaded_file.getvalue(), filetype="pdf") as doc:
                pages = doc.page_count
                for pagina_num in range(doc.page_count):
                    pagina = doc[pagina_num]
                    bloco_texto = pagina.get_text("text")
                    texto += bloco_texto

        return texto, texto, pages
    

    #TRATAMENTO
    def translate_text(self):
        if self.linguagem != 'en':
            corpus = pd.DataFrame()
            corpus['texto'] = [self.text]
            corpus['tradução'] = corpus['texto'].apply(lambda x: TextBlob(x).translate(from_lang=self.linguagem, to='en')).astype('str')
            self.text = corpus['tradução'][0]
        else:
            self.text = self.text

        return self.text

    def to_lower(self):
        self.text = self.text_completo.lower()
        return self.text

    def remove_expressoes(self):
        self.text = re.sub(r"\n", " ", self.text)
        self.text = re.sub(r'\r', " ", self.text)
        self.text = re.sub(r'-', ' ', self.text)
        self.text = re.sub(r'\t', ' ', self.text)
        self.text = re.sub(r'\s+', ' ', self.text)
        return self.text

    def remove_stopwords_br(self):
        stop_words_pt = set(stopwords.words('portuguese'))
        tokens = word_tokenize(self.text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words_pt]
        self.text = ' '.join(filtered_tokens)
        return self.text

    def remove_stopwords_en(self):
        stop_words_en = set(stopwords.words('english'))    
        tokens = word_tokenize(self.text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words_en]
        self.text = ' '.join(filtered_tokens)
        return self.text

    def remove_pontuaiton(self):
        punctuation = set(string.punctuation)
        for i in self.text:
            if i in punctuation:
                self.text = self.text.replace(i, "")
        return self.text

    def remove_numbers(self):
        self.text = ''.join(filter(lambda z: not z.isdigit(), self.text))
        return self.text

    def tokenize(self):
        self.token = pd.DataFrame()
        self.token['token'] = word_tokenize(self.text)
        return self.token

    def clear_txt(self):
        self.remove_expressoes()
        self.translate_text()
        self.to_lower()
        self.remove_stopwords_en()
        self.remove_stopwords_br()
        self.remove_pontuaiton()
        self.remove_numbers()
        self.tokenize()
        return self.text

    #def clear_txt(self):
    #    self.remove_expressoes()
    #    self.translate_text()
    #    self.to_lower()
    #    self.remove_stopwords_en()
    #    self.remove_stopwords_br()
    #    self.remove_pontuaiton()
    #    self.remove_numbers()
    #    self.tokenize()


    def sentiment_analysis_LMC(self):
        
        self.Loughan_Mc = pd.read_excel('.Loughran_McDonald.xlsx')
        self.Loughan_Mc = self.Loughan_Mc.loc[(self.Loughan_Mc['sentiment'] == 'positive') | (self.Loughan_Mc['sentiment'] == 'negative')]

        self.sentiment_analysis = None
        self.sentiment_analysis = pd.merge(self.token, self.Loughan_Mc, on='token')
        self.sentiment_analysis = self.sentiment_analysis.groupby('sentiment').size()
        self.sentiment_analysis['sentiment'] = ((self.sentiment_analysis['positive'] - self.sentiment_analysis['negative']) / (self.sentiment_analysis['positive'] + self.sentiment_analysis['negative']))

        return self.sentiment_analysis['sentiment']


    def sentiment_analysis_Insider(self):
        self.insider = pd.read_csv('.General_Insider.csv', sep=';')
        self.insider = self.insider.loc[(self.insider['sentiment'] == 'positive') | (self.insider['sentiment'] == 'negative')]

        self.sentiment_analysis = None
        self.sentiment_analysis = pd.merge(self.token, self.insider, on='token')
        self.sentiment_analysis = self.sentiment_analysis.groupby('sentiment').size()
        self.sentiment_analysis['sentiment'] = ((self.sentiment_analysis['positive'] - self.sentiment_analysis['negative']) / (self.sentiment_analysis['positive'] + self.sentiment_analysis['negative']))

        return self.sentiment_analysis['sentiment']


    def summary(self):

        if self.linguagem != "pt":
            
            self.text_temp = self.text_completo
            self.text_temp = re.sub(r'[\n\r-]', ' ', self.text_temp)

            self.resumo = pd.DataFrame()
            self.resumo['texto'] = [self.text_temp]
            self.resumo['tradução'] = self.resumo['texto'].apply(lambda x: TextBlob(x).translate(from_lang= f'{self.linguagem}', to='pt')).astype('str')
            self.resumo = self.resumo['tradução'][0]
            parser = PlaintextParser.from_string(self.resumo, Tokenizer("portuguese"))
            summarizer = LsaSummarizer()
            self.resumo = summarizer(parser.document, 5)  #5 é o número de sentenças no resumo
            
            return self.resumo
        
        else:

            self.text_temp = self.text_completo
            self.text_temp = re.sub(r'[\n\r-]', ' ', self.text_temp)

            self.resumo = [self.text_temp]
            parser = PlaintextParser.from_string(self.resumo, Tokenizer("portuguese"))
            summarizer = LsaSummarizer()
            self.resumo = summarizer(parser.document, self.pages)  #5 é o número de sentenças no resumo
            
            return self.resumo
            

    #FUNÇÕES - OUTPUT
    def most_frequent(self):
        self.frequent = Counter(self.token['token'])

        self.frequent = {'Termo': list(self.frequent.keys()), 'Frequência': list(self.frequent.values())}
        self.df_frequent = pd.DataFrame(self.frequent)
        self.df_frequent = self.df_frequent.sort_values(by='Frequência', ascending=False).reset_index(drop=True)
        self.df_frequent = self.df_frequent.head(10)
        

        if self.linguagem != 'pt':
            self.df_frequent['Termo'] = self.df_frequent['Termo'].map(lambda x: TextBlob(x).translate(from_lang= self.linguagem, to='pt')).astype('str')

        return self.df_frequent