import os
import shutil
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.tools import tool
from embedding import get_embedding

@tool("查詢與給定文本相似的文件")
def query_chroma(query_text: str, database_directory: str, number_of_most_similar_results: int = 3) -> str:
    """
    Args:
        query_text (str): 查詢文本。
        database_directory (str): 資料庫目錄，若是作業批改員，指定 chroma_answer、若為錯題整理師指定 chroma_question、若為教科書分析師指定 chroma_textbook 。
        number_of_most_similar_results (int, optional): 要返回的最相似文件的數量。你可以自己決定，但沒有設定下 default 為 3。

    Returns:
        str: 包含最相似文件內容的字符串，每個文件之間用 "\n\n---\n\n" 分隔。

    """
    database_directory = "chroma/" + database_directory
    
    vector_database = Chroma(
        persist_directory=database_directory,
        embedding_function=get_embedding(),
    )
    
    results = vector_database.similarity_search_with_score(
        query_text,
        k=number_of_most_similar_results
    )
    
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    context_text = f"###來源: {sources}\n\n{context_text}"
    return context_text