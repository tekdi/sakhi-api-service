import argparse
import re
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import SimpleDirectoryReader
from vectorstores.marqo import MarqoVectorStore 


def document_loader(input_dir: str) -> List[Document]:
    """Load data from the input directory.

    Args:
        input_dir (str): Path to the directory.

    Returns:
        List[Document]: A list of documents.
    """
    return SimpleDirectoryReader(
        input_dir=input_dir, recursive=True).load_data()

def split_documents(documents: List[Document], chunk_size: int = 4000, chunk_overlap=200) -> List[Document]:
    """Split documents.

    Args:
        documents: List of documents
        chunk_size: Maximum size of chunks to return
        chunk_overlap: Overlap in characters between chunks

    Returns:
        List[Document]: A list of documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Pattern to remove unsupported hexadecimal characters for Marqo.
    pattern = re.compile(r'[\x00-\x1F\x7F\u00A0]')
    splited_docs = []
    for document in documents:
        for chunk in text_splitter.split_text(document.text):
            chunk = re.sub(pattern, '', chunk)
            splited_docs.append(Document(page_content=chunk, metadata={
                "page_label": document.metadata.get("page_label"),
                "file_name": document.metadata.get("file_name"),
                "file_type": document.metadata.get("file_type")
            }))
    return splited_docs

def load_documents(folder_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    documents = document_loader(folder_path)
    splitted_documents = split_documents(documents, chunk_size, chunk_overlap)
    return splitted_documents

def indexer_main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path',
                        type=str,
                        required=True,
                        help='Path to the folder',
                        default="input_data"
                        )
    parser.add_argument('--chunk_size',
                        type=int,
                        required=False,
                        help='documents chunk size',
                        default=1024
                        )
    parser.add_argument('--chunk_overlap',
                        type=int,
                        required=False,
                        help='documents chunk overlap size',
                        default=200
                        )
    parser.add_argument('--fresh_index',
                        action='store_true',
                        help='Is the indexing fresh'
                        )
    parser.add_argument('--index_name',
                        type=str,
                        required=True,
                        help='Name of the vector collection'
                        )

    args = parser.parse_args()

    FOLDER_PATH = args.folder_path
    FRESH_INDEX = args.fresh_index
    CHUNK_SIZE = args.chunk_size
    CHUNK_OVERLAP = args.chunk_overlap
    INDEX_NAME = args.index_name

    documents = load_documents(FOLDER_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
    print("Total documents :: =>", len(documents))
    
    print("Adding documents...")
    vector_store = MarqoVectorStore(index_name=INDEX_NAME)
    results = vector_store.add_documents(documents, FRESH_INDEX)
    print("results =======>", results)
    
    print("============ INDEX DONE =============")


if __name__ == "__main__":
    indexer_main()
    
# For Fresh collection
# python3 index_documents.py --folder_path=Documents --fresh_index

# For appending documents to existing collection
# python3 index_documents.py --folder_path=Documents