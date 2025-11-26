import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

from common_utils import prepare_vectors


# Шаг 1: Загрузка и подготовка данных из CSV
def embed_data(csv_file):
    """Подготовка данных для индексации FAISS"""
    
    # Загружаем предобученный энкодер
    model = SentenceTransformer('cointegrated/LaBSE-en-ru')
    # это урезанная версия LaBSE, сохранившая только русские и английские токены

    # Создаем эмбеддинги для всех текстов
    def encode_fn(texts):
        emb = model.encode(texts, show_progress_bar=True)
        return emb.astype('float32')

    embeddings, ids, df, _ = prepare_vectors(
        csv_file=csv_file,
        vectorizer_factory=encode_fn,
        text_column='text',
        id_column='id'
    )

    return embeddings, ids, df, model

# Шаг 2: Построение индекса FAISS
def build_faiss_index(embeddings, df):
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