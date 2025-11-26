import pandas as pd

def prepare_vectors(csv_file, vectorizer_factory, text_column='text', id_column='id'):
    """
    Общая функция для подготовки векторов из CSV файла.

    Args:
        csv_file (str): Путь к CSV-файлу.
        vectorizer_factory (callable or object): 
            - Если callable: функция, принимающая список текстов и возвращающая векторы (например, model.encode).
            - Если объект с методами `fit_transform`/`transform`: будет вызван .fit_transform(texts).
        text_column (str): Название колонки с текстами.
        id_column (str): Название колонки с ID.

    Returns:
        tuple: (vectors, ids, df, fitted_vectorizer_or_model)
    """
    df = pd.read_csv(csv_file)
    texts = df[text_column].tolist()
    ids = df[id_column].values

    # Определяем, как применить векторизатор/модель
    if callable(vectorizer_factory):
        vectors = vectorizer_factory(texts)
        fitted_obj = None
    else:
        vectors = vectorizer_factory.fit_transform(texts)
        fitted_obj = vectorizer_factory

    return vectors, ids, df, fitted_obj