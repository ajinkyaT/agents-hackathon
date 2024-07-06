import os
import typing as t
from typing import Any
from pydantic import BaseModel
import jsonlines
import uuid

from langchain.schema import Document
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.csv import partition_csv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document

from agents.summarize_chain import summarize_chain

class Element(BaseModel):
    type: str
    text: Any
    source: str

class VectorDB:
    def __init__(self, doc_store_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.doc_store_path = doc_store_path
        self.chunk_size = chunk_size
        self.vector_store_dir: str = "stihl_db"
        self.processed_docs_dir: str = "stihl_doc_store"
        self.chunk_overlap = chunk_overlap
        self.docs_summary = None
        self.docs_texts = None
        self.doc_ids = None
        self.vectorstore = None
        self.retriever = None
        if not os.path.exists(self.processed_docs_dir):
            self.load_documents()
        else: self.docs_summary, self.docs_texts, self.doc_ids = self.load_docs_from_jsonl()
        self.create_vectorstore()

    def save_docs_to_jsonl(self, documents: t.Iterable[t.Tuple[Document, str, str]]) -> None:
        print(f"Saving processed docs to: {self.processed_docs_dir}")
        if not os.path.exists(self.processed_docs_dir):
            os.makedirs(self.processed_docs_dir, exist_ok=True)
        with jsonlines.open(os.path.join(self.processed_docs_dir, "docs_backup.json"), mode="w") as writer:
            for doc, text, uuid in documents:
                writer.write((doc.dict(), text, uuid))

    def load_docs_from_jsonl(self) -> t.Iterable[t.Tuple[Document, str,str]]:
        documents = []
        texts = []
        uuids = []
        with jsonlines.open(os.path.join(self.processed_docs_dir, "docs_backup.json"), mode="r") as reader:
            for doc_dict, text, uuid in reader:
                doc = Document(**doc_dict)
                documents.append(doc)
                texts.append(text)
                uuids.append(uuid)
        return documents, texts, uuids
    
    def get_raw_pdf_elements(self, pdf_file_path):
        print(f"Processing PDF: {pdf_file_path}")
        raw_pdf_elements = partition_pdf(
    filename=os.path.join(self.doc_store_path, pdf_file_path),
    extract_images_in_pdf=False,
    # Titles are any sub-section of the document
    infer_table_structure=True,
    # Post processing to aggregate text once we have the title
    chunking_strategy="by_title",
    # Chunking params to aggregate text blocks
    # Attempt to create a new chunk 3800 chars
    # Attempt to keep chunks > 2000 chars
    max_characters=4000,
    new_after_n_chars=3800,
    combine_text_under_n_chars=2000,
)
        categorized_elements = []
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                categorized_elements.append(Element(type="table", text=str(element), source=element.metadata.filename))
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                categorized_elements.append(Element(type="text", text=str(element), source=element.metadata.filename))
        # Tables
        table_elements = [e for e in categorized_elements if e.type == "table"]
        print(f"Total tables found: {len(table_elements)} \n")
        # Text
        text_elements = [e for e in categorized_elements if e.type == "text"]
        print(f"Total texts found: {len(text_elements)} \n")
        return text_elements, table_elements

    def get_summaries(self, elements):
        text_summaries = summarize_chain.batch(elements, {"max_concurrency": 5})
        return text_summaries
    
    def load_documents(self):
        pdf_file_paths = [f for f in os.listdir(self.doc_store_path) if f.endswith('.pdf')]
        csv_files = [f for f in os.listdir(self.doc_store_path) if f.endswith('.csv')]
        text_elements = []
        table_elements = []
        for csv_file in csv_files:
            print(f'Processing csv file: {csv_file}')
            raw_tables = partition_csv(filename=os.path.join(self.doc_store_path, csv_file))
            for table in raw_tables:
                                table_elements.append(Element(type="table", text=str(table), source=table.metadata.filename))
        for pdf_file_path in pdf_file_paths:
            text_elements_batch, table_elements_batch = self.get_raw_pdf_elements(pdf_file_path)
            if text_elements_batch:
                text_elements.extend(text_elements_batch)
            if table_elements_batch:
                table_elements.extend(table_elements_batch)
        combined_docs = text_elements + table_elements
        self.docs_texts = [e.text for e in combined_docs]
        print(f"length of combined_docs list: {len(combined_docs)}")
        self.doc_ids = [str(uuid.uuid4()) for _ in self.docs_texts]
        summary_texts = self.get_summaries(self.docs_texts)
        id_key = "doc_id"
        self.docs_summary = [
            Document(page_content=f"{s} \n Source File: {combined_docs[i].source}", metadata={id_key: self.doc_ids[i],"source_file":combined_docs[i].source})
            for i, s in enumerate(summary_texts)
        ]
        self.save_docs_to_jsonl(list(zip(self.docs_summary, self.docs_texts, self.doc_ids)))

    def create_vectorstore(self):
        self.vectorstore = vectorstore = Chroma(collection_name="summaries", embedding_function=OpenAIEmbeddings(), persist_directory=self.vector_store_dir)
        store = InMemoryStore()
        id_key = "doc_id"
        self.retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=store,
            id_key=id_key,
            search_kwargs={"k": 3}
        )
        self.retriever.vectorstore.add_documents(self.docs_summary)
        text_docs= [
            Document(page_content=f"{s} \n Source File: {self.docs_summary[i].metadata['source_file']}", metadata={id_key: self.doc_ids[i]})
            for i, s in enumerate(self.docs_texts)
        ]
        self.retriever.docstore.mset(list(zip(self.doc_ids, text_docs)))
        print(f"Initialized vector store: \n Number of docs{len(self.docs_texts)}")

    def get_documents(self):
        return self.docs_summary

