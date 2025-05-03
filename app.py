import streamlit as st
import pandas as pd
import nltk
nltk.download('punkt')  # Correct tokenizer
from nltk import tokenize
from bs4 import BeautifulSoup
import requests
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import io
import docx2txt
from PyPDF2 import PdfReader
import plotly.express as px
from serpapi import GoogleSearch

# ========== Helper Functions ==========

def get_sentences(text):
    return tokenize.sent_tokenize(text)

def get_url_from_serpapi(sentence):
    params = {
        "q": sentence,
        "api_key": "92a3c30132653330b8c1ff58a9aef5fee2bf9f438e96047547859a5cd080d92c",  # Replace with your SerpAPI key
        "engine": "google",
        "num": "1"
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return results['organic_results'][0]['link']
    except:
        return None

def get_text(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        return ' '.join(p.text for p in soup.find_all('p'))
    except:
        return ""

def read_text_file(file):
    with io.open(file.name, 'r', encoding='utf-8') as f:
        return f.read()

def read_docx_file(file):
    return docx2txt.process(file)

def read_pdf_file(file):
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def get_text_from_file(uploaded_file):
    if uploaded_file:
        if uploaded_file.type == "text/plain":
            return read_text_file(uploaded_file)
        elif uploaded_file.type == "application/pdf":
            return read_pdf_file(uploaded_file)
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return read_docx_file(uploaded_file)
    return ""

def get_similarity(text1, text2):
    vectorizer = CountVectorizer().fit([text1, text2])
    count_matrix = vectorizer.transform([text1, text2])
    return cosine_similarity(count_matrix)[0][1]

def get_similarity_list(texts, filenames=None):
    if filenames is None:
        filenames = [f"File {i+1}" for i in range(len(texts))]
    return [
        (filenames[i], filenames[j], get_similarity(texts[i], texts[j]))
        for i in range(len(texts)) for j in range(i+1, len(texts))
    ]

def plot_all(df):
    plot_functions = [px.scatter, px.line, px.bar, px.pie, px.box, px.histogram, px.scatter_3d, px.violin]
    titles = ['Scatter', 'Line', 'Bar', 'Pie', 'Box', 'Histogram', '3D Scatter', 'Violin']
    for func, title in zip(plot_functions, titles):
        try:
            if title == '3D Scatter':
                fig = func(df, x='File 1', y='File 2', z='Similarity', color='Similarity', title=title)
            elif title == 'Pie':
                fig = func(df, values='Similarity', names='File 1', title=title)
            else:
                fig = func(df, x='File 1', y='Similarity' if title != 'Scatter' else 'File 2',
                           color='File 2' if title not in ['Box', 'Histogram', 'Violin'] else 'Similarity',
                           title=title)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.write(f"Could not render {title} plot: {e}")

# ========== Streamlit UI ==========

st.set_page_config(page_title='Plagiarism Detection')
st.title('Plagiarism Detector')

st.write("### Enter the text or upload a file to check for plagiarism or find similarities between files")

option = st.radio("Select input option:", ('Enter text', 'Upload file', 'Find similarities between files'))

if option == 'Enter text':
    text = st.text_area("Enter text here", height=200)
    uploaded_files = []
elif option == 'Upload file':
    uploaded_file = st.file_uploader("Upload file (.docx, .pdf, .txt)", type=["docx", "pdf", "txt"])
    text = get_text_from_file(uploaded_file) if uploaded_file else ""
    uploaded_files = [uploaded_file] if uploaded_file else []
else:
    uploaded_files = st.file_uploader("Upload multiple files", type=["docx", "pdf", "txt"], accept_multiple_files=True)
    texts = []
    filenames = []
    for file in uploaded_files:
        if file:
            texts.append(get_text_from_file(file))
            filenames.append(file.name)
    text = " ".join(texts)

if st.button('Check for plagiarism or find similarities'):
    if not text:
        st.warning("No text found for plagiarism check or similarity analysis.")
        st.stop()

    if option == 'Find similarities between files':
        similarities = get_similarity_list(texts, filenames)
        df = pd.DataFrame(similarities, columns=['File 1', 'File 2', 'Similarity']).sort_values(by='Similarity', ascending=False)
        plot_all(df)
    else:
        with st.spinner("Searching the web for possible plagiarism..."):
            sentences = get_sentences(text)
            urls = [get_url_from_serpapi(sentence) for sentence in sentences]
            filtered = [(s, u) for s, u in zip(sentences, urls) if u]

        if not filtered:
            st.success("No plagiarism detected!")
            st.stop()

        similarity_scores = []
        filtered_sentences, filtered_urls = zip(*filtered)

        with st.spinner("Analyzing similarity..."):
            for sent, url in zip(filtered_sentences, filtered_urls):
                web_text = get_text(url)
                score = get_similarity(sent, web_text)
                if score > 0.3:  # You can tweak the threshold
                    similarity_scores.append((sent, url, score))

        if not similarity_scores:
            st.success("No plagiarism detected above threshold!")
            st.stop()

        df = pd.DataFrame(similarity_scores, columns=['Sentence', 'URL', 'Similarity'])
        df = df.sort_values(by='Similarity', ascending=False).reset_index(drop=True)
        df['URL'] = df['URL'].apply(lambda x: f'<a href="{x}" target="_blank">{x}</a>')
        df_html = df.to_html(escape=False).replace('<th>URL</th>', '<th style="text-align: center;">URL</th>')
        st.write(df_html, unsafe_allow_html=True)
