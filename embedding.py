import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import LlamaCppEmbeddings
from langchain_community.embeddings import GPT4AllEmbeddings


load_dotenv()

def get_embedding():
    # embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # embeddings = LlamaCppEmbeddings(model_path="/path/to/model")
    embeddings = GPT4AllEmbeddings()
    return embeddings

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import PCA
    from gensim.models import KeyedVectors
    
    sentences = [
        "國文科包括閱讀理解與寫作測驗",
        "英文科測驗學生聽說讀寫的能力",
        "數學科考試包含選擇題和計算題",
        "自然科測驗物理、化學、生物、地球科學",
        "社會科包括歷史、地理、公民與社會",
        "數學跟自然科都需要計算能力",
        "考生可以選考自然科或社會科",
        "指考成績採計各科級分轉換成分數"
    ]
    
    embeddings = get_embedding()
    similarities = []
    
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            emb1 = embeddings.embed_query(sentences[i])
            emb2 = embeddings.embed_query(sentences[j])
            similarity = cosine_similarity([emb1], [emb2])[0][0]
            similarities.append((sentences[i], sentences[j], similarity))

    print("語義相似度測試:")        
    for s1, s2, sim in sorted(similarities, key=lambda x: x[2], reverse=True):
        print(f"{s1} 和 {s2} 的餘弦相似度為: {sim:.3f}")
        
    # 2. 向量運算測試
    print("\n向量運算測試:")


    king = np.array(embeddings.embed_query("數學"))
    man = np.array(embeddings.embed_query("代數"))
    woman = np.array(embeddings.embed_query("幾何"))
    queen = king - man + woman
    queen = king - man + woman

    most_similar = None
    max_similarity = -1
    for sent in sentences:
        similarity = cosine_similarity([embeddings.embed_query(sent)], [queen])[0][0]
        if similarity > max_similarity:
            most_similar = sent
            max_similarity = similarity

    print(f"'數學' - '代數' + '幾何' ≈ '{most_similar}', 餘弦相似度為 {max_similarity:.3f}")

    # 3. 下游任務表現測試 (以文本分類為例)
    print("\n下游任務表現測試:")
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())

    def preprocess(text):
        return ' '.join([str(v) for v in embeddings.embed_query(text)])

    X = vectorizer.fit_transform([preprocess(text) for text in sentences])  
    print(f"TF-IDF 向量維度: {X.shape[1]}")

    # 4. 鑑別能力測試
    print("\n鑑別能力測試:")
    print("餘弦相似度:")
    print("'國文科包括閱讀理解與寫作測驗' vs '英文科測驗學生聽說讀寫的能力': ", 
        cosine_similarity([embeddings.embed_query("國文科包括閱讀理解與寫作測驗")], 
                            [embeddings.embed_query("英文科測驗學生聽說讀寫的能力")])[0][0])
    print("'國文科包括閱讀理解與寫作測驗' vs '英文科包括閱讀理解與寫作測驗': ",
        cosine_similarity([embeddings.embed_query("國文科包括閱讀理解與寫作測驗")],
                            [embeddings.embed_query("英文科包括閱讀理解與寫作測驗")])[0][0])
                            
    # 5. OOV 詞表現測試
    print("\nOOV 詞表現測試:")
    oov_word = "學測指考" 
    print(f"OOV詞 '{oov_word}' 的 embedding: {embeddings.embed_query(oov_word)}")

    most_similar_to_oov = None
    max_similarity_to_oov = -1
    for sent in sentences:
        similarity = cosine_similarity([embeddings.embed_query(sent)], [embeddings.embed_query(oov_word)])[0][0]
        if similarity > max_similarity_to_oov:
            most_similar_to_oov = sent
            max_similarity_to_oov = similarity
    
    print(f"跟 OOV 詞 '{oov_word}' 最相似的句子是: '{most_similar_to_oov}', 餘弦相似度為 {max_similarity_to_oov:.3f}")

# 6. 維度可解釋性測試 - 可視化
    print("\n維度可解釋性測試 - 可視化:")
    embeddings_list = [embeddings.embed_query(sent) for sent in sentences]
    pca = PCA(n_components=2)
    embeddings_pca = pca.fit_transform(embeddings_list)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=embeddings_pca[:, 0], y=embeddings_pca[:, 1], 
                    hue=sentences, palette="viridis", legend=False)
    for i in range(len(sentences)):
        plt.annotate(sentences[i], xy=(embeddings_pca[i, 0], embeddings_pca[i, 1]), xytext=(5, 2), 
                     textcoords='offset points', ha='right', va='bottom')
    plt.title("Embedding 可視化 (PCA)")
    
    # 將圖片儲存到檔案
    # plt.savefig("embedding_visualization.png")
    # print("可視化圖片已儲存為 embedding_visualization.png")


    # 7. 穩定性測試
    print("\n穩定性測試:")
    emb1 = embeddings.embed_query("國文科包括閱讀理解與寫作測驗")
    emb2 = embeddings.embed_query("國文科包括閱讀理解與寫作測驗")
    stability = cosine_similarity([emb1], [emb2])[0][0]
    print(f"多次 Embedding 的穩定性(餘弦相似度): {stability:.3f}")
    
    
