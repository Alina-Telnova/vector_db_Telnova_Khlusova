"""
Модуль для загрузки, обработки и сохранения текстовых данных из JSON файлов
для последующей векторизации текста
"""

import os
import json
import re
import pandas as pd
from typing import List, Dict, Any
import hashlib


class TextDataLoader:
    """Класс для загрузки, обработки и сохранения текстовых данных из JSON файлов для векторизации."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data: Dict[str, Any] = {}
        self.cleaned_posts: List[Dict[str, Any]] = []
        
        # Предкомпилированные регулярные выражения для скорости
        self._url_pattern = re.compile(r'http\S+')
        self._mention_pattern = re.compile(r'[@#]\w+')
        self._clean_pattern = re.compile(r'[^\w\s.,!?]')
        self._space_pattern = re.compile(r'\s+')
        
        self._initialize_loader()
    
    def _initialize_loader(self) -> None:
        """Инициализация загрузчика: загрузка и обработка данных."""
        self.data = self._load_data(self.file_path)
        self.cleaned_posts = self._extract_and_clean_posts()
    
    def _load_data(self, file_path: str) -> Dict[str, Any]:
        """Загрузка JSON данных из файла."""
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Файл '{file_path}' не существует")
        
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def _generate_document_id(self, text: str, date: str) -> str:
        """Генерация уникального ID для документа."""
        return hashlib.md5(f"{text}_{date}".encode('utf-8')).hexdigest()[:16]
    
    def _extract_text_from_message(self, message: Dict[str, Any]) -> str:
        """Извлечение текстового содержимого из сообщения."""
        text = message.get('text', '')
        
        if isinstance(text, str):
            return text.strip()
        
        if isinstance(text, list):
            text_parts = []
            for item in text:
                if isinstance(item, str):
                    text_parts.append(item.strip())
                elif isinstance(item, dict):
                    nested_text = item.get('text', '')
                    if isinstance(nested_text, str):
                        text_parts.append(nested_text.strip())
            return ' '.join(text_parts)
        
        return ''
    
    def _clean_text(self, text: str) -> str:
        """Очистка текста."""
        cleaned_text = text.lower()
        cleaned_text = self._url_pattern.sub('', cleaned_text)
        cleaned_text = self._mention_pattern.sub('', cleaned_text)
        cleaned_text = self._clean_pattern.sub('', cleaned_text)
        cleaned_text = self._space_pattern.sub(' ', cleaned_text).strip()
        return cleaned_text
    
    def _extract_and_clean_posts(self) -> List[Dict[str, Any]]:
        """Извлечение и очистка постов в одном проходе."""
        cleaned_posts = []
        
        for message in self.data.get('messages', []):
            text_content = self._extract_text_from_message(message)
            
            if len(text_content) > 10:
                date = message.get('date', '')[:10]
                cleaned_text = self._clean_text(text_content)
                
                if len(cleaned_text) > 10:
                    doc_id = self._generate_document_id(cleaned_text, date)
                    
                    cleaned_posts.append({
                        'id': doc_id,
                        'date': date,
                        'text': cleaned_text
                    })
        
        return cleaned_posts
    
    def save_to_csv(self, output_file: str = 'data.csv') -> str:
        """Сохранение очищенных постов в CSV файл для векторизации."""
        df = pd.DataFrame(self.cleaned_posts)
        df.to_csv(output_file, index=False, encoding='utf-8')
        print('='*3, 'SUCCESSFULLY PARSED YOUR DATA', '='*3)
        print(df.head())
        return output_file


if __name__ == "__main__":
    loader = TextDataLoader('result.json')
    output_file = loader.save_to_csv()
