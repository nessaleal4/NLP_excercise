import streamlit as st
import spacy
import fitz  # PyMuPDF for PDF extraction
import networkx as nx
from pyvis.network import Network
import tempfile
import os
from bs4 import BeautifulSoup

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF"""
    doc = fitz.open(pdf_file)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

def extract_entities(text):
    """Extract authors, organizations, and citations using spaCy NER"""
    doc = nlp(text)
    authors = set()
    organizations = set()
    citations = set()
    
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            authors.add(ent.text)
        elif ent.label_ == "ORG":
            organizations.add(ent.text)
        elif ent.label_ in ["WORK_OF_ART", "MISC"]:  # Citations may be tagged under different labels
            citations.add(ent.text)
    
    return authors, organizations, citations

def create_knowledge_graph(authors, organizations, citations):
    """Create a knowledge graph using NetworkX and Pyvis"""
    G = nx.Graph()
    
    for author in authors:
        G.add_node(author, label="Author", color="blue")
    
    for org in organizations:
        G.add_node(org, label="Organization", color="red")
        for author in authors:  # Assume authors may be affiliated with organizations
            G.add_edge(author, org)
    
    for citation in citations:
        G.add_node(citation, label="Citation", color="green")
        for author in authors:  # Assume authors cite works
            G.add_edge(author, citation)
    
    return G

def render_graph(G):
    """Render an interactive knowledge graph with Pyvis"""
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
    net.from_nx(G)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
        net.save_graph(tmpfile.name)
        html_code = open(tmpfile.name, "r", encoding="utf-8").read()
        os.unlink(tmpfile.name)  # Delete temp file after reading
        return html_code

def main():
    st.set_page_config(page_title="Knowledge Graph from PDFs", layout="wide")
    st.title("📄 Technical Paper Knowledge Graph")
    st.write("Upload a technical paper in PDF format, and we will extract key entities and visualize their relationships.")
    
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_file)
            authors, organizations, citations = extract_entities(text)
            
        st.subheader("Extracted Entities")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("### Authors")
            st.write(authors if authors else "No authors found")
        with col2:
            st.write("### Organizations")
            st.write(organizations if organizations else "No organizations found")
        with col3:
            st.write("### Citations")
            st.write(citations if citations else "No citations found")
        
        st.subheader("📊 Knowledge Graph")
        G = create_knowledge_graph(authors, organizations, citations)
        graph_html = render_graph(G)
        
        st.components.v1.html(graph_html, height=600)
    
if __name__ == "__main__":
    main()
