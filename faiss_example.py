import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Шаг 1: Загрузка и подготовка данных из CSV
def load_data_from_csv(csv_file):
    """Загрузить данные из CSV файла и подготовить для индексации FAISS"""
    df = pd.read_csv(csv_file)
    
    # Загружаем предобученный энкодер
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Создаем эмбеддинги для всех текстов
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)
    embeddings = embeddings.astype('float32')
    
    ids = df['id'].values
    
    return embeddings, ids, df, model

# Шаг 2: Построение индекса FAISS
def build_faiss_index(embeddings):
    """Построить индекс FAISS из векторов-эмбеддингов"""
    embeddings = embeddings.astype('float32')
    
    # Нормализуем векторы для косинусного сходства
    faiss.normalize_L2(embeddings)
    
    # Используем Inner Product для косинусного сходства
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    print('\n'*3, '='*3, 'SUCCESSFULLY BUILD INDEX', '='*3)
    print('EXAMPLE #128:', df.iloc[128]['text'][:50], "...\n-->\n", index.reconstruct(128)[:8])
    return index

# Шаг 3: Функция поиска
def search_similar_vectors(index, query_vector, k=5, ids=None, original_df=None):
    """Поиск похожих векторов в индексе FAISS"""
    query_vector = query_vector.astype('float32')
    
    # Нормализуем query вектор
    faiss.normalize_L2(query_vector)
    
    # Ищем похожие векторы (cosine similarity)
    similarities, indices = index.search(query_vector, k)
    
    results = []
    for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
        if idx != -1:
            result = {
                'rank': i + 1,
                'similarity': similarity,
                'index': idx
            }
            
            if ids is not None:
                result['id'] = ids[idx]
            if original_df is not None:
                result['data'] = original_df.iloc[idx].to_dict()
            
            results.append(result)
    
    return results

if __name__ == "__main__":
    csv_file = "data.csv"
    
    # Загружаем данные из CSV
    embeddings, ids, df, model = load_data_from_csv(csv_file)
    
    # Строим индекс FAISS
    index = build_faiss_index(embeddings)
    
    # Простой поиск через input
    search_query = input("\nВведите слово для поиска: ")
    
    # Создаем эмбеддинг для запроса
    query_embedding = model.encode([search_query]).astype('float32')
    
    # Выполняем поиск
    results = search_similar_vectors(index, query_embedding, k=3, ids=ids, original_df=df)
    
    print(f"\nТоп-3 для запроса '{search_query}':")
    for result in results:
        data = result['data']
        print(f"{result['rank']}. ID: {data['id']}, Дата: {data['date']}")
        print(f"   Текст: {data['text']}")
        print(f"   Сходство: {result['similarity']:.4f}\n")