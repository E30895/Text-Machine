import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
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
import PyPDF2
from io import BytesIO

@st.cache_resource
class TextAnalysis:

    """
    Uma classe para análise de arquivos de texto, incluindo tarefas como tradução, transformação para minúsculas, remoção de stopwords, pontuação, números, tokenização, análise de sentimento e sumarização.

    Parâmetros:
        uploaded_file (file): O arquivo de texto a ser analisado.

    Atributos:
        uploaded_file (file): O arquivo de texto enviado.
        text_completo (str): O texto completo extraído do arquivo enviado.
        text (str): Texto processado para análise.
        pages (int): Número de páginas no arquivo PDF enviado.
        linguagem (str): Idioma detectado do texto.

    Métodos:
        read_uploaded_file(): Lê o arquivo enviado e extrai o texto.
        translate_text(): Traduz o texto para inglês, se ainda não estiver em inglês.
        to_lower(): Converte o texto para minúsculas.
        remove_expressoes(): Remove várias expressões como quebra de linha, tabulações, etc.
        remove_stopwords_br(): Remove stopwords do texto em português.
        remove_stopwords_en(): Remove stopwords do texto em inglês.
        remove_pontuaiton(): Remove pontuação do texto.
        remove_numbers(): Remove números do texto.
        tokenize(): Tokeniza o texto.
        clear_txt(): Aplica todos os processos de limpeza de texto.
        sentiment_analysis_LMC(): Realiza análise de sentimento usando o léxico de Loughran-McDonald.
        sentiment_analysis_Insider(): Realiza análise de sentimento usando o léxico Insider.
        summary(): Gera um resumo do texto.
        most_frequent(): Encontra as palavras mais frequentes no texto.

    """

    def __init__(self, uploaded_file):
        nltk.download('stopwords')
        nltk.download('punkt')
        self.uploaded_file = uploaded_file
        self.text_completo, self.text, self.pages = self.read_uploaded_file()
        self.linguagem = str(detect(self.text_completo))

    def read_uploaded_file(self):

        """
        Lê o arquivo enviado e extrai o texto.

        Retorna:
            str: O texto completo extraído do arquivo.
            int: O número de páginas no arquivo PDF.
        """
                
        texto = ""
        pages = 0

        try:
            if self.uploaded_file is not None:
                file_stream = BytesIO(self.uploaded_file.getvalue())
                pdf_reader = PyPDF2.PdfReader(file_stream)
                pages = len(pdf_reader.pages)

                for pagina_num in range(pages):
                    pagina = pdf_reader.pages[pagina_num]
                    texto += pagina.extract_text()

            return texto, texto, pages
        
        except:

            st.error('Erro ao processar arquivo')
            st.stop()

    def translate_text(self):

        """
        Traduz o texto para inglês, se ainda não estiver em inglês.

        Retorna:
            str: O texto traduzido para inglês.
        """
                
        if self.linguagem != 'en':
            corpus = pd.DataFrame()
            corpus['texto'] = [self.text]
            corpus['tradução'] = corpus['texto'].apply(lambda x: TextBlob(x).translate(from_lang=self.linguagem, to='en')).astype('str')
            self.text = corpus['tradução'][0]
        else:
            self.text = self.text

        return self.text

    def to_lower(self):

        """
        Converte o texto para minúsculas.

        Retorna:
            str: O texto em minúsculas.
        """

        self.text = self.text_completo.lower()
        return self.text

    def remove_expressoes(self):

        """
        Remove várias expressões como quebra de linha, tabulações, etc.

        Retorna:
            str: O texto sem as expressões indesejadas.
        """

        self.text = re.sub(r"\n", " ", self.text)
        self.text = re.sub(r'\r', " ", self.text)
        self.text = re.sub(r'-', ' ', self.text)
        self.text = re.sub(r'\t', ' ', self.text)
        self.text = re.sub(r'\s+', ' ', self.text)
        return self.text

    def remove_stopwords_br(self):

        """
        Remove stopwords do texto em português.

        Retorna:
            str: O texto sem as stopwords em português.
        """
                
        stop_words_pt = set(stopwords.words('portuguese'))
        tokens = word_tokenize(self.text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words_pt]
        self.text = ' '.join(filtered_tokens)
        return self.text

    def remove_stopwords_en(self):

        """
        Remove stopwords do texto em inglês.

        Retorna:
            str: O texto sem as stopwords em inglês.
        """

        stop_words_en = set(stopwords.words('english'))    
        tokens = word_tokenize(self.text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words_en]
        self.text = ' '.join(filtered_tokens)
        return self.text

    def remove_pontuaiton(self):

        """
        Remove pontuação do texto.

        Retorna:
            str: O texto sem pontuação.
        """
                
        punctuation = set(string.punctuation)
        for i in self.text:
            if i in punctuation:
                self.text = self.text.replace(i, "")
        return self.text

    def remove_numbers(self):

        """
        Remove números do texto.

        Retorna:
            str: O texto sem números.
        """
                
        self.text = ''.join(filter(lambda z: not z.isdigit(), self.text))
        return self.text

    def tokenize(self):
        
        """
        Tokeniza o texto.

        Retorna:
            DataFrame: Um DataFrame contendo os tokens do texto.
        """
                
        self.token = pd.DataFrame()
        self.token['token'] = word_tokenize(self.text)
        return self.token

    def clear_txt(self):
        
        """
        Aplica todos os processos de limpeza de texto.

        Retorna:
            str: O texto limpo e processado.
        """
        
        self.remove_expressoes()
        self.translate_text()
        self.to_lower()
        self.remove_stopwords_en()
        self.remove_stopwords_br()
        self.remove_pontuaiton()
        self.remove_numbers()
        self.tokenize()
        return self.text

    def sentiment_analysis_LMC(self):

        """
        Realiza análise de sentimento usando o léxico de Loughran-McDonald.

        Retorna:
            float: O valor de sentimento calculado.
            str: Uma mensagem de erro se os elementos forem insuficientes.
        """

        try:
        
            self.Loughan_Mc = pd.read_excel('.Loughran_McDonald.xlsx')
            self.Loughan_Mc = self.Loughan_Mc.loc[(self.Loughan_Mc['sentiment'] == 'positive') | (self.Loughan_Mc['sentiment'] == 'negative')]

            self.sentiment_analysis = None
            self.sentiment_analysis = pd.merge(self.token, self.Loughan_Mc, on='token')
            self.sentiment_analysis = self.sentiment_analysis.groupby('sentiment').size()
            self.sentiment_analysis['sentiment'] = ((self.sentiment_analysis['positive'] - self.sentiment_analysis['negative']) / (self.sentiment_analysis['positive'] + self.sentiment_analysis['negative']))

            return round(self.sentiment_analysis['sentiment'],4)
        
        except: 

            return "Elementos insuficientes"

    def sentiment_analysis_Insider(self):

        """
        Realiza análise de sentimento usando o léxico Insider.

        Retorna:
            float: O valor de sentimento calculado.
            str: Uma mensagem de erro se os elementos forem insuficientes.
        """

        try:

            self.insider = pd.read_csv('.General_Insider.csv', sep=';')
            self.insider = self.insider.loc[(self.insider['sentiment'] == 'positive') | (self.insider['sentiment'] == 'negative')]

            self.sentiment_analysis = None
            self.sentiment_analysis = pd.merge(self.token, self.insider, on='token')
            self.sentiment_analysis = self.sentiment_analysis.groupby('sentiment').size()
            self.sentiment_analysis['sentiment'] = ((self.sentiment_analysis['positive'] - self.sentiment_analysis['negative']) / (self.sentiment_analysis['positive'] + self.sentiment_analysis['negative']))

            return round(self.sentiment_analysis['sentiment'], 4)
        
        except:

            return "Elementos insuficientes"


    def summary(self):

        """
        Gera um resumo do texto.

        Retorna:
            str: O resumo gerado.
        """

        if self.linguagem != "pt":
            
            self.text_temp = self.text_completo
            self.text_temp = re.sub(r'[\n\r-]', ' ', self.text_temp)

            self.resumo = pd.DataFrame()
            self.resumo['texto'] = [self.text_temp]
            self.resumo['tradução'] = self.resumo['texto'].apply(lambda x: TextBlob(x).translate(from_lang= f'{self.linguagem}', to='pt')).astype('str')
            self.resumo = self.resumo['tradução'][0]
            parser = PlaintextParser.from_string(self.resumo, Tokenizer("portuguese"))
            summarizer = LsaSummarizer()
            self.resumo = summarizer(parser.document, self.pages)  
            
            return self.resumo
        
        else:

            self.text_temp = self.text_completo
            self.text_temp = re.sub(r'[\n\r-]', ' ', self.text_temp)

            self.resumo = [self.text_temp]
            parser = PlaintextParser.from_string(self.resumo, Tokenizer("portuguese"))
            summarizer = LsaSummarizer()
            self.resumo = summarizer(parser.document, self.pages)
            return self.resumo
            

    def most_frequent(self):

        """
        Encontra as palavras mais frequentes no texto.

        Retorna:
            DataFrame: Um DataFrame contendo as palavras mais frequentes.
        """
                
        self.frequent = Counter(self.token['token'])

        self.frequent = {'Termo': list(self.frequent.keys()), 'Frequência': list(self.frequent.values())}
        self.df_frequent = pd.DataFrame(self.frequent)
        self.df_frequent = self.df_frequent.sort_values(by='Frequência', ascending=False).reset_index(drop=True)
        self.df_frequent = self.df_frequent.head(10)
        

        if self.linguagem != 'pt':
            self.df_frequent['Termo'] = self.df_frequent['Termo'].map(lambda x: TextBlob(x).translate(from_lang= self.linguagem, to='pt')).astype('str')

        return self.df_frequent
