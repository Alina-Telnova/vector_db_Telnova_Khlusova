import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from common_utils import prepare_vectors

# Шаг 1: Загрузка и подготовка данных из CSV
def vectorize_data(csv_file):
    """Подготовить данные для TF-IDF"""
    
    # Создаем TF-IDF векторизатор и матрицу
    vectorizer = TfidfVectorizer()
    
    tfidf_matrix, ids, df, _ = prepare_vectors(
        csv_file=csv_file,
        vectorizer_factory=vectorizer,
        text_column='text',
        id_column='id'
    )
    
    return tfidf_matrix, ids, df, vectorizer

# Шаг 2: Построение TF-IDF матрицы
def build_tfidf_index(tfidf_matrix, df):
    """Построить TF-IDF матрицу и показать пример"""
    print('\n'*3, '='*3, 'SUCCESSFULLY BUILT TF-IDF MATRIX', '='*3)
    
    # Показываем пример для 128-го документа
    if tfidf_matrix.shape[0] > 128:
        sample_vector = tfidf_matrix[128].toarray()[0]
        non_zero_indices = np.nonzero(sample_vector)[0]
        print('EXAMPLE #128:', df.iloc[128]['text'][:50], "...")
        print('-->')
        if len(non_zero_indices) > 0:
            print(f"TF-IDF features (first 8 non-zero): {sample_vector[non_zero_indices[:8]]}")
        else:
            print("No non-zero TF-IDF features")
    
    return tfidf_matrix

# Шаг 3: Функция поиска
def search_similar_texts(tfidf_matrix, query_vector, k=5, ids=None, original_df=None):
    """Поиск похожих текстов используя косинусное сходство"""
    
    # Вычисляем косинусное сходство
    similarities = cosine_similarity(query_vector, tfidf_matrix)
    
    # Получаем индексы топ-k наиболее похожих документов
    top_indices = similarities[0].argsort()[-k:][::-1]
    
    results = []
    for i, idx in enumerate(top_indices):
        result = {
            'rank': i + 1,
            'similarity': similarities[0][idx],
            'index': idx
        }
        
        if ids is not None:
            result['id'] = ids[idx]
        if original_df is not None:
            result['data'] = original_df.iloc[idx].to_dict()
        
        results.append(result)
    
    return results