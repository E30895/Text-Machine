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
from nltk.stem import SnowballStemmer
from collections import Counter
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from langdetect import detect
from view import sidebar, conteudo
from model import TextAnalysis
from controller import app
import PyMuPDF
from fitz import frontend


def main():
    print("Iniciando")
    app()

if __name__ == "__main__":
    main()
