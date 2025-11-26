from tfidf_example import build_tfidf_index, search_similar_texts, vectorize_data
from faiss_example import build_faiss_index, search_similar_vectors, embed_data

csv_file = "data.csv"

# Загружаем данные из CSV
tfidf_matrix, ids, df, vectorizer = vectorize_data(csv_file)
embeddings, ids, df, model = embed_data(csv_file)

# Строим TF-IDF матрицу
tfidf_matrix = build_tfidf_index(tfidf_matrix, df)
# Строим индекс FAISS
index = build_faiss_index(embeddings)

# Простой поиск через input
search_query = input("\nВведите слово для поиска: ")

# Преобразуем запрос в TF-IDF вектор
query_vector = vectorizer.transform([search_query])
# Создаем эмбеддинг для запроса в FAISS
query_embedding = model.encode([search_query]).astype('float32')

# Выполняем поиск
results_tfidf = search_similar_texts(tfidf_matrix, query_vector, k=3, ids=ids, original_df=df)
results_vectors = search_similar_vectors(index, query_embedding, k=3, ids=ids, original_df=df)

print(f"\nТоп-3 для запроса с tf-idf'{search_query}':")
for result in results_tfidf:
    data = result['data']
    print(f"{result['rank']}. ID: {data['id']}, Дата: {data['date']}")
    print(f"   Текст: {data['text'][:50]}")
    print(f"   Сходство: {result['similarity']:.8f}\n")


print(f"\nТоп-3 для запроса с FAISS'{search_query}':")
for result in results_vectors:
    data = result['data']
    print(f"{result['rank']}. ID: {data['id']}, Дата: {data['date']}")
    print(f"   Текст: {data['text']}")
    print(f"   Сходство: {result['similarity']:.4f}\n")