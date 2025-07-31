# Complaint Data Analysis Project
## Overview
This project analyzes consumer complaints from the Consumer Financial Protection Bureau (CFPB) database to provide insights into financial product issues and support query resolution for financial analysts. The dataset, originally containing 9,609,797 records, was filtered to 527,570 records with non-empty narratives, focusing on products such as credit cards, personal loans, Buy Now Pay Later services, savings accounts, and money transfers. The analysis involves exploratory data analysis (EDA), data preprocessing, vector store creation for Retrieval-Augmented Generation (RAG), and the development of a Streamlit-based web application for querying complaint data.

## Project Structure
### Notebooks:
**eda.ipynb**: Performs exploratory data analysis to understand dataset structure, missing values, and complaint distributions.
**data_preprocessing.ipynb:** Filters and cleans the dataset, focusing on relevant products and narratives.
**chunk_embed_index.ipynb:** Segments narratives, generates embeddings, and indexes them in a ChromaDB vector store.
**rag_pipeline.ipynb:** Implementation of the RAG pipeline for query resolution.
**app.ipynb:** Development of the Streamlit web application.

**vector_store/**: Stores the ChromaDB collection for indexed complaint chunks.

### Data:
**complaints.csv:** Original dataset with 9,609,797 records and 18 columns.
**preprocessed_complaints.csv:** Filtered dataset with 527,570 records containing cleaned narratives.
**./vector_store/:** Directory containing the ChromaDB vector store with 200 indexed chunks (from a 100-record sample).


### Prerequisites
To run the project, ensure the following are installed:

Python 3.8 or higher

**Libraries**: pandas, numpy, nltk, sentence-transformers, chromadb, huggingface_hub, streamlit

Jupyter Notebook or Google Colab for running .ipynb files
Access to the CFPB dataset (complaints.csv)

### Dependencies:
Install required packages using:pip install pandas numpy nltk langchain sentence-transformers chromadb matplotlib seaborn


**Hardware**: Sufficient memory for processing large datasets (e.g., 16GB RAM). For full dataset processing, consider cloud-based solutions.
**Access**: Google Drive access to complaints.csv (as used in Colab environment).

### Installation

**Clone the repository**:git clone <repository-url>
cd <repository-directory>


**Install dependencies**:pip install -r requirements.txt


Ensure complaints.csv is accessible or download it from the CFPB website.

### Usage

**1. Exploratory Data Analysis:**

Run eda.ipynb to explore dataset characteristics, including product distribution and missing values.

**2. Data Preprocessing:**

Execute data_preprocessing.ipynb to filter records, clean narratives, and save the preprocessed dataset as preprocessed_complaints.csv.

**3. Chunking, Embedding, and Indexing:**

Run chunk_embed_index.ipynb to create text chunks, generate embeddings using all-MiniLM-L6-v2, and index them in a ChromaDB vector store.

**4. RAG Pipeline:**

Use rag_pipeline.ipynb to test the Retrieval-Augmented Generation pipeline with sample questions.

**5. Web Application:**

Launch the Streamlit app by running:

### Preprocess Data:
jupyter notebook data_preprocessing.ipynb


Filters for products: Credit card, Personal loan, Buy Now, Pay Later, Savings account, Money transfer.
Cleans narratives by removing special characters, stopwords, and boilerplate text.
Saves preprocessed_complaints.csv.


### Index for RAG:
jupyter notebook chunk_embed_index.ipynb


Samples 100 records, splits narratives into ~500-character chunks, generates embeddings using all-MiniLM-L6-v2, and indexes in ChromaDB.
Stores results in ./vector_store/.



### Key Findings

**EDA**: Visualizations of complaint distribution by product and narrative presence.

**Preprocessing**: Filters dataset to relevant products and cleans narratives by removing special characters, stopwords, and boilerplate text.

**Vector Store**: Indexes complaint chunks in ChromaDB for semantic search using all-MiniLM-L6-v2 embeddings.

**RAG Pipeline**: Retrieves relevant complaint excerpts and generates answers using the google/flan-t5-base model.

**Streamlit App**: Provides a user-friendly interface for querying complaints, displaying results with source excerpts, and maintaining conversation history.

**Indexing**: 200 chunks from 100 sampled records indexed in ChromaDB for RAG applications.

**Visualizations**
The project includes the following visualizations (see report for details):

Complaint Distribution by Product: Bar chart showing complaint volumes for filtered products.
Narrative Presence: Pie chart displaying the proportion of complaints with/without narratives.
Top Sub-products: Bar chart highlighting leading sub-products by complaint volume.

### Limitations

High missing narrative rate (69.0%) limits text-based analysis.
Memory constraints required sampling 100 records for indexing.
Full dataset processing may require distributed computing.

