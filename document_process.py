from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import pdfplumber
import os

class DocumentProcessor:
    def __init__(self, data_path: str, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load_documents(self):
        documents = []
        for file_name in os.listdir(self.data_path):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(self.data_path, file_name)
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        text = text.encode('utf-8', errors='ignore').decode('utf-8')
                        metadata = {"source": file_name, "page": page_num + 1}
                        document = Document(page_content=text, metadata=metadata)
                        documents.append(document)
        return documents

    def split_documents(self, documents: list[Document]):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
        )
        return splitter.split_documents(documents)

    def process_documents(self):
        documents = self.load_documents()
        splitted_documents = self.split_documents(documents)
        return splitted_documents
    
if __name__ == "__main__":
    processor = DocumentProcessor(data_path="data")
    result = processor.load_documents()
    print(len(result))
    result = processor.split_documents(result)
