import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import nltk
import streamlit as st
import openpyxl
from view import sidebar, conteudo
from model import TextAnalysis
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import PyPDF2
from io import BytesIO

def app():

    # ENTRADA
    input, processar_button = sidebar()

    if processar_button:

        # INSTANCIANDO A CLASSE
        Text_Analysis = TextAnalysis(input)


        #PROCESSAMENTO
        Text_Analysis.clear_txt()
        
        #ANÁLISES
        lmc = Text_Analysis.sentiment_analysis_LMC()
        insider = Text_Analysis.sentiment_analysis_Insider()
        frequent = Text_Analysis.most_frequent()
        summary = Text_Analysis.summary()
        

        # SAÍDA
        conteudo(lmc=lmc, harvard=insider, frequent = frequent, summary=summary)
