# Complaint Data Analysis Project
## Overview
This project analyzes consumer complaints from the Consumer Financial Protection  database, comprising 9,609,797 records. The workflow includes exploratory data analysis (EDA), data preprocessing, and indexing for a Retrieval-Augmented Generation (RAG) system. The dataset is filtered to focus on specific financial products, resulting in 527,570 records with non-empty narratives, which are cleaned, segmented, and stored in a ChromaDB vector store. Visualizations highlight complaint distributions and data characteristics.

## Project Structure
### Notebooks:
**eda.ipynb**: Performs exploratory data analysis to understand dataset structure, missing values, and complaint distributions.
**data_preprocessing.ipynb:** Filters and cleans the dataset, focusing on relevant products and narratives.
**chunk_embed_index.ipynb:** Segments narratives, generates embeddings, and indexes them in a ChromaDB vector store.


### Data:
**complaints.csv:** Original dataset with 9,609,797 records and 18 columns.
**preprocessed_complaints.csv:** Filtered dataset with 527,570 records containing cleaned narratives.
**./vector_store/:** Directory containing the ChromaDB vector store with 200 indexed chunks (from a 100-record sample).


### Prerequisites

Python: Version 3.8 or higher.
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

Run EDA:
jupyter notebook eda.ipynb


Analyzes dataset structure, missing values, and complaint distributions.
Outputs visualizations (e.g., product distribution bar charts, narrative presence pie chart).


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

Dataset: 9,609,797 records, with 69.0% missing narratives.
Filtered Dataset: 527,570 records with non-empty narratives after focusing on select products.
**Product Distribution:**
Checking or savings account: 26.7%
Credit card: 20.8%
Credit card or prepaid card: 18.9%


### Visualizations:
Bar charts show complaint volumes by product and sub-product.
Pie chart illustrates narrative presence (31.0% with narratives).


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

