import streamlit as st
import os
import PyPDF2
import re
import torch
import requests
import nltk
import gensim
import google.generativeai as genai
import numpy as np
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
from keybert import KeyBERT
from gensim.models import KeyedVectors
from scipy.linalg import triu
from dotenv import load_dotenv

load_dotenv()
# Initialize session state
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'papers' not in st.session_state:
    st.session_state.papers = {}
if 'title' not in st.session_state:
    st.session_state.title = ""
if 'abstract' not in st.session_state:
    st.session_state.abstract = ""
if 'keywords' not in st.session_state:
    st.session_state.keywords = ""

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Set page config
st.set_page_config(page_title="Research Literature Assistant", layout="wide")

# --- Helper Functions with Caching ---
@st.cache_resource
def load_models():
    """Load ML models with caching"""
    return {
        'keybert': KeyBERT('distilbert-base-nli-mean-tokens'),
        'word2vec': KeyedVectors.load_word2vec_format(
            'content/word2vec-slim/GoogleNews-vectors-negative300-SLIM.bin.gz',
            binary=True
        )
    }

models = load_models()

# --- PDF Processing Functions ---
def extractTitle(text):
    chat_messages = [
        SystemMessage(content='You are an expert assistant with expertise in extracting titles from academic papers'),
        HumanMessage(content=f'Extract the paper title from this text. Return only the title without any additional text:\n\n{text}')
    ]
    llm = ChatOpenAI(model_name='gpt-4')
    return llm(chat_messages).content

def extractAbstract(text):
    chat_messages = [
        SystemMessage(content='You are an expert assistant with expertise in extracting abstracts from academic papers'),
        HumanMessage(content=f'Extract the abstract from this text. Return only the abstract content without any additional text:\n\n{text}')
    ]
    llm = ChatOpenAI(model_name='gpt-4')
    return llm(chat_messages).content

def extractKeyword(text):
    chat_messages = [
        SystemMessage(content='You are an expert assistant with expertise in extracting keywords from academic papers'),
        HumanMessage(content=f'Extract keywords from this text. Return as comma-separated values or "no" if none found:\n\n{text}')
    ]
    llm = ChatOpenAI(model_name='gpt-4')
    return llm(chat_messages).content

# --- Summarization Functions ---
def summaryUsingGpt4(text, word_count=50):
    chat_messages = [
        SystemMessage(content='You are an expert academic summarizer'),
        HumanMessage(content=f'Summarize this in {word_count} words:\n{text}')
    ]
    llm = ChatOpenAI(model_name='gpt-4')
    return llm(chat_messages).content

def summaryUsingGemini(text, word_count=50):
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(f'Summarize this in {word_count} words: {text}')
    return response.text

def summaryUsingGpt3(text, word_count=50):
    chat_messages = [
        SystemMessage(content='You are an expert academic summarizer'),
        HumanMessage(content=f'Summarize this in {word_count} words:\n{text}')
    ]
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    return llm(chat_messages).content

# def summaryUsingT5(text):
#     tokenizer = T5Tokenizer.from_pretrained('t5-base')
#     model = T5ForConditionalGeneration.from_pretrained('t5-base')
#     inputs = tokenizer.encode("summarize: " + text, return_tensors='pt', max_length=512, truncation=True)
#     outputs = model.generate(inputs, max_length=150)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# def summaryUsingBart(text):
#     tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
#     model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')
#     inputs = tokenizer([text], max_length=1024, return_tensors='pt')
#     summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=150)
#     return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# --- Paper Analysis Functions ---
def summarize_text(text, num_keywords=5):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_freq = Counter(filtered_words)
    common_words = word_freq.most_common(num_keywords)
    return " ".join([w[0] for w in common_words]), [w[0] for w in common_words]

def calculate_wmd_similarity(text1, text2):
    return models['word2vec'].wmdistance(
        word_tokenize(text1.lower()),
        word_tokenize(text2.lower())
    )

def calculate_combined_score(year, citation_count, wmd_similarity):
    return (0.3 * (2024 - year)) + (0.3 * citation_count) + (0.4 * (1 - wmd_similarity))

S2_API_KEY = "91JejA8b7l6c5vlyXLqm145uPbcuKfXQ49pxbmem"
def find_papers(keywords, num_papers=10, offset=0):
    papers = {}
    try:
        response = requests.get(
            'https://api.semanticscholar.org/graph/v1/paper/search',
            headers={'X-API-KEY': S2_API_KEY},
            params={
                'query': " ".join(keywords),
                'fields': 'title,abstract,url,year,citationCount,authors',
                'limit': num_papers,
                'offset': offset
            }
        )
        
        response_json = response.json()
        if 'data' not in response_json:
            st.error(f"Unexpected API response: {response_json}")
            return {}

        for paper in response_json['data']:
            if 'title' in paper and 'abstract' in paper:
                score = calculate_combined_score(
                    paper.get('year', 2023),
                    paper.get('citationCount', 0),
                    calculate_wmd_similarity(st.session_state.title, paper['title'])
                )
                papers[paper.get('paperId', 'Unknown')] = {
                    'title': paper['title'],
                    'abstract': paper['abstract'],
                    'authors': paper.get('authors', []),
                    'year': paper.get('year', 'Unknown'),
                    'citations': paper.get('citationCount', 0),
                    'url': paper.get('url', '#'),
                    'score': score
                }

        return dict(sorted(papers.items(), key=lambda x: x[1]['score'], reverse=True))

    except requests.exceptions.RequestException as e:
        st.error(f"API Request Error: {e}")
    except Exception as e:
        st.error(f"Unexpected Error: {e}")
    
    return {}

# --- Compressive Summary Functions ---
def compressiveSummary(text, year):
    chat_messages = [
        SystemMessage(content='You are an expert assistant with expertize in compressive summarization of text given to you. By extractive summarization i mean that the words/sentences occuring in the text should appear as it in the summary,but only pick the important words and sentences that carry important meaning'),#systemMessages tell out model that it is an expert who can summarize paragraphs
    HumanMessage(content=f'''Please provide a short and concise summary of the following paragraphs, in about 40-60 words.
     that paper has been mentioned some relevant sentence like the authors x et.al. emphasised/discussued/proposed/investigated/analysed to give a start to the summary.Always use this et. al. after author's surname if only two authors in paper write the authors name add authors by & , no need need to use et al when two authors .no need to write the title of the paper in the summary.The authors name should definitely come in the  summary
    Generate in 35-50 words. Also write the  year in brackets immediately after the author's Surname et al.and write the citied paper number  :\n TEXT: {text}{year}''')
  ]
    llm = ChatOpenAI(model_name='gpt-4')
    return llm(chat_messages).content

def summaryFilter(text):
    chat_messages = [
        SystemMessage(content='You are an expert assistant with expertize in filtering the text given to you. The text given to you is the generated literature review by my model and it comprises of few sentences of the form author1 et al.... so this is the right format and any useless sentences having no significance should be removed.'),#systemMessages tell out model that it is an expert who can summarize paragraphs
    HumanMessage(content=f'''Please filter the text given to you. The summary should have sentences like 'author xyz -content...' And if any other irrelevant sentence is present in the text given then that must be removed. The text given to you is the generated related work and that should be generally of the form 'author 1 et al content'   {text}''')
  ]
    llm = ChatOpenAI(model_name='gpt-4')
    return llm(chat_messages).content

# --- UI Components ---
def home_page():
    st.header("📄 ELitSumm: Leveraging Ensemble of LLMs for Automatic Literature Work Generation")
    uploaded_file = st.file_uploader("Upload research paper (PDF)", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Analyzing PDF..."):
            text = ""
            try:
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                for i, page in enumerate(pdf_reader.pages):
                    if i >= 2:  # First 2 pages
                        break
                    text += page.extract_text() or ""
                
                st.session_state.title = extractTitle(text)
                st.session_state.abstract = extractAbstract(text)
                st.session_state.keywords = extractKeyword(text)
                st.session_state.processed = True
                
                with st.expander("Extracted Metadata", expanded=True):
                    cols = st.columns([3,2])
                    cols[0].subheader("Title")
                    cols[0].write(st.session_state.title)
                    cols[1].subheader("Keywords")
                    cols[1].write(st.session_state.keywords)
                    st.subheader("Abstract")
                    st.write(st.session_state.abstract)
                    
            except Exception as e:
                st.error(f"PDF Processing Error: {str(e)}")

def summary_page():
    st.header("📝 Summary Generation")
    if not st.session_state.processed:
        st.warning("Please process a PDF first on the Home page")
        return
    
    col1, col2 = st.columns([3, 1])
    with col1:
        method = st.selectbox("Select Summarization Method", [
            "GPT-4", "Gemini", "GPT-3.5", "T5", "BART"
        ])
    with col2:
        word_count = st.number_input("Summary Length (words)", 
                                    min_value=20, 
                                    max_value=500, 
                                    value=50,
                                    step=10,
                                    help="Set the desired length of your summary in words")
    
    if st.button("Generate Summary"):
        combined_text = f"{st.session_state.title}\n{st.session_state.abstract}"
        with st.spinner(f"Generating {word_count}-word summary with {method}..."):
            try:
                if method == "GPT-4":
                    result = summaryUsingGpt4(combined_text, word_count)
                elif method == "Gemini":
                    result = summaryUsingGemini(combined_text, word_count)
                elif method == "GPT-3.5":
                    result = summaryUsingGpt3(combined_text, word_count)
                elif method == "T5":
                    # Note: For T5 and BART implementations, you'll need to add length control
                    # This is a placeholder that temporarily uses Gemini
                    result = summaryUsingGemini(combined_text, word_count)
                elif method == "BART":
                    # Placeholder using Gemini until BART is implemented with length control
                    result = summaryUsingGemini(combined_text, word_count)
                
                st.subheader("Generated Summary")
                st.text_area("Generated Summary", result, height=150)
                
                # Word count verification
                actual_word_count = len(result.split())
                st.caption(f"Generated summary contains {actual_word_count} words")
                
            except Exception as e:
                st.error(f"Summarization Error: {str(e)}")

def related_papers_page():
    st.header("🔍 Related Papers")
    if not st.session_state.processed:
        st.warning("Please process a PDF first on the Home page")
        return
    
    if st.button("Find Related Research"):
        with st.spinner("Searching Semantic Scholar..."):
            try:
                _, keywords = summarize_text(f"{st.session_state.title} {st.session_state.abstract}")
                papers = find_papers(keywords)
                st.session_state.papers = papers
                
                st.subheader(f"Top {len(papers)} Related Papers")
                for pid, paper in papers.items():
                    with st.expander(f"{paper['title']} ({paper['year']})"):
                        st.write(f"**Citations:** {paper['citations']}")
                        st.write(f"**Abstract:** {paper['abstract']}")
                        st.markdown(f"[📄 Paper Link]({paper['url']})")
                        
            except Exception as e:
                st.error(f"Search Error: {str(e)}")

def compressive_summary_page():
    st.header("📚 Literature Review/Related Work")
    if not st.session_state.processed:
        st.warning("Please process a PDF first on the Home page")
        return

    # User controls
    num_papers = st.number_input("Number of Papers to Include", 
                               min_value=1, max_value=50, value=10)
    sort_by = st.selectbox("Sort Papers By", ["Relevance Score", "Recent First", "Citation Count"])

    if st.button("Generate Literature Review"):
        with st.spinner("🔍 Finding relevant papers and generating review..."):
            try:
                # Fetch papers with user-specified quantity
                _, keywords = summarize_text(f"{st.session_state.title} {st.session_state.abstract}")
                papers = find_papers(keywords, num_papers=num_papers)
                
                if not papers:
                    st.error("No relevant papers found.")
                    return

                # Sort papers based on user selection
                sort_keys = {
                    "Relevance Score": lambda x: x[1]['score'],
                    "Recent First": lambda x: -x[1]['year'],
                    "Citation Count": lambda x: -x[1]['citations']
                }
                sorted_papers = sorted(papers.items(), 
                                     key=sort_keys[sort_by], 
                                     reverse=True)

                # Generate formatted summaries
                literature_entries = []
                for pid, paper in sorted_papers[:num_papers]:
                    # Author formatting
                    authors = paper.get('authors', [])
                    if authors:
                        surnames = [a['name'].split()[-1] for a in authors if 'name' in a]
                        if len(surnames) > 2:
                            authors_str = f"{surnames[0]} et al."
                        else:
                            authors_str = " & ".join(surnames)
                    else:
                        authors_str = "Unknown Authors"

                    # Generate compressive summary
                    paper_text = f"Title: {paper['title']}\nAbstract: {paper['abstract']}"
                    summary = compressiveSummary(paper_text, paper.get('year', 'N/A'))
                    filtered_summary = summaryFilter(summary)

                    literature_entries.append(
                        f"**{authors_str} ({paper.get('year', 'N/A')})** " 
                        f"[{paper['title']}]({paper['url']}): {filtered_summary}"
                    )

                # Combine entries with optional LLM synthesis
                combined_review = "\n\n".join(literature_entries)
                
                # Optional coherence enhancement
                if len(literature_entries) > 3:
                    combined_review = enhance_coherence(combined_review)

                st.subheader("Generated Literature Review")
                st.markdown(combined_review, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error generating review: {str(e)}")
def enhance_coherence(text):
    """Improve flow between paragraphs using LLM"""
    chat_messages = [
        SystemMessage(content='You are an academic editor. Improve the flow between these paragraphs while preserving all key information.'),
        HumanMessage(content=f"Make this literature review more cohesive:\n\n{text}")
    ]
    llm = ChatOpenAI(model_name='gpt-4')
    return llm(chat_messages).content

# --- Navigation ---
pages = {
    "Home": home_page,
    "Summarization": summary_page,
    "Related Papers": related_papers_page,
    "Literature Review": compressive_summary_page
}

# --- Sidebar ---
st.sidebar.header("🔑 API Configuration")
openai_key = st.sidebar.text_input("OpenAI Key", type="password")
os.environ["OPENAI_API_KEY"] = openai_key
google_key = st.sidebar.text_input("Google AI Key", type="password")
genai.configure(api_key=google_key)

st.sidebar.header("🧭 Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()