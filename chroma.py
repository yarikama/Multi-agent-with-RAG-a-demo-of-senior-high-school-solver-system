import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.tools import tool
from embedding import get_embedding

def record_metadata_chunks(chunks: list[Document]):
    previous_page_id = None
    current_chunk = 0
    for chunk in chunks:
        file_name = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{file_name}_{page}"
        current_chunk = current_chunk + 1 if previous_page_id == current_page_id else 0
        chunk_id = f"{current_page_id}_{current_chunk}"
        previous_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks


# 教科書分析師使用 chroma_answer 資料庫
# 作業批改員使用 chroma_question 資料庫
# 錯題整理師使用 chroma_textbook 資料庫
def chroma_add_chunks(chunks: list[Document], database_directory: str):
    vector_database = Chroma(
        persist_directory=database_directory,
        embedding_function=get_embedding(),
    )
    chunks_with_metadata = record_metadata_chunks(chunks)
    existing_items = vector_database.get(include=[])
    existing_ids = set(existing_items["ids"])
    print(f"現在有的檔案數為: {len(existing_ids)}")
    
    # 只添加新的檔案
    new_chunks = [chunk for chunk in chunks_with_metadata if chunk.metadata["id"] not in existing_ids]
    if len(new_chunks) > 0:
        print(f"新增的檔案數為: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vector_database.add_documents(new_chunks, ids=new_chunk_ids)
        vector_database.persist()
    else:
        print("沒有新增的檔案")

def clean_vector_database():
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    print("已清空資料庫")
    
    
if __name__ == "__main__":
    from document_process import DocumentProcessor
    data_list = ["answer", "exam", "textbook"]
    for data_element in data_list:
        data_path = "data/" + data_element
        processor = DocumentProcessor(data_path)
        processed_documents = processor.process_documents()
        add_to_database = "chroma/chroma_" + data_element + "_data"
        chroma_add_chunks(processed_documents, add_to_database)
        print(f"已添加 {data_path} 到 {add_to_database}")  