import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import nltk
import openpyxl
from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import streamlit as st
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import PyPDF2
from io import BytesIO

def sidebar():

    """
    Configura a barra lateral da aplicação.

    A barra lateral inclui informações sobre a aplicação, como título, descrição e opção de fazer upload do arquivo de texto.

    Retorna:
        file_uploader: Um objeto FileUploader do Streamlit para fazer upload de arquivos.
        button: Um botão para iniciar o processamento do texto.
    """
        
    st.set_page_config(page_title="Text Machine", layout='wide')

    st.sidebar.title('Text Machine')
    st.sidebar.header('Sobre')

    st.sidebar.markdown(
        """
        <h3 style='text-align: justify;'>

        Text Machine é uma aplicativo para análises textuais. 
        Ao realizar o upload do arquivo de texto, terá rápido acesso aos recursos da Linguistica Computacional.

        **Nota:** 
        Nesta versão, o aplicativo esta preparado para processar apenas textos em Ingles.
        
        </h3>
        """, 
        unsafe_allow_html=True)

    input = st.sidebar.file_uploader("Faça upload do arquivo de texto", type=["pdf", "txt"])
    processar_button = st.sidebar.button('Processar')

    st.sidebar.markdown(
       """
       <h3 style='text-align: justify;'>

        **Dicionários:** 
        A implementação do índice de sentimento depende exclusivamente do escopo do texto analisado.
        **Loughran McDonald** é o dicionário expecifico para contextos de economia e finanças. Deve ser levado em consideração apenas os textos tratarem especificamente de economia e finanças.
        **General Insider** é um dicionário generalista. Deve ser levado em consideração para textos genéricos, sem um contexto específico.

    
        **Referencial Teórico:** Kim e Hovy (2004), Hu e Liu (2004), Loughran and McDonald (2011), Nopp and Hanbury (2015) e Aprigliano et al. (2023). 
        </h3>
        
        """,
        unsafe_allow_html=True)

    return input, processar_button



def conteudo(lmc, harvard, frequent, summary):

    """
    Exibe o conteúdo principal da aplicação.

    Este método exibe as métricas de sentimentos, os termos mais frequentes e um resumo do texto processado.

    Parâmetros:
        lmc (float): Sentimento calculado usando o léxico Loughran-McDonald.
        harvard (float): Sentimento calculado usando o léxico Insider.
        frequent (DataFrame): DataFrame contendo os termos mais frequentes no texto.
        summary (str): Resumo do texto processado.
    """
        
    container = st.container()
    col1, col2 = st.columns(2)

    with col1:
        st.metric('Loughran and McDonald Sentiment¹', lmc)

    with col2:
        st.metric('General Insider Sentiment²', harvard)

    col4, col5 = st.columns(2)

    with col4:
        st.write('Termos mais frequentes')
        st.table(frequent)
        st.write('Termos mais frequentes')
        st.bar_chart(frequent.set_index('Termo'), use_container_width=True)


    with col5:
        container = st.container()
        container.write('Resumo')
        for item in summary:
            container.write(str(item))
