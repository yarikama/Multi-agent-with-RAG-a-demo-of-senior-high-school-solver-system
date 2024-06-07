from langchain.tools import tool

@tool("output .md file")
def outputMD(result, filename):
    """
    Args:
        result: 要寫入文件的內容,通常是一個字串。
        filename: 要創建的 Markdown 文件的名稱,需要包含 .md 後綴。

    Returns:
        file: 一個 md 類型的文件。
    """
    with open(filename, "w") as file:
        file.write(result)
    return file