import asyncio
import logging
import os
import json
from dotenv import load_dotenv
from yandex_neural_api.client import YandexNeuralAPIClient

def main():
    # Загрузка переменных окружения из файла .env (если используется)
    load_dotenv()
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)

    # Замените 'YOUR_IAM_TOKEN' и 'YOUR_FOLDER_ID' на реальные значения
    iam_token = os.environ['IAM_TOKEN']
    folder_id = os.environ['FOLDER_ID']

    # Инициализация клиента YandexNeuralAPIClient с заданными параметрами
    client = YandexNeuralAPIClient(
        iam_token,
        folder_id,
        model_type='pro',           # Используем модель 'pro' для генерации текста
        llm_temperature=0.7,        # Устанавливаем температуру генерации текста
        llm_max_tokens=1000,        # Максимальное количество токенов для генерации текста
        log_level=logging.DEBUG     # Устанавливаем уровень логирования на DEBUG для подробной информации
    )

    # Синхронная генерация текста
    print("Синхронная генерация текста:")
    prompt_text_sync = "Напиши короткое стихотворение о программировании."
    text_sync = client.generate_text(prompt_text_sync)
    print(text_sync)
    print("="*50)

    # Асинхронная генерация текста
    # async def async_text_generation():
    #     print("Асинхронная генерация текста:")
    #     prompt_text_async = "Напиши краткое описание об искусственном интеллекте."
    #     text_async = await client.generate_text_async(prompt_text_async)
    #     print(text_async)
    #     print("="*50)
    # asyncio.run(async_text_generation())

    # Синхронная стриминговая генерация текста
    print("Синхронная стриминговая генерация текста:")
    def sync_streaming_callback(data):
        data_json = json.loads(data)
        print(data_json["result"]["alternatives"][0]["message"]["text"], end='\n', flush=True)

    prompt_stream_sync = "Расскажи интересный факт о Вселенной."
    # Генерация текста в режиме стриминга с использованием callback-функции
    client.generate_text(prompt_stream_sync, stream=True, callback=sync_streaming_callback)
    print("\n" + "="*50)

    # Асинхронная стриминговая генерация текста
    # async def async_streaming():
    #     print("Асинхронная стриминговая генерация текста:")
    #     def async_streaming_callback(data):
    #         print(data, end='', flush=True)

    #     prompt_stream_async = "Опиши закат над морем."
    #     await client.generate_text_async(prompt_stream_async, stream=True, callback=async_streaming_callback)
    #     print("\n" + "="*50)
    # asyncio.run(async_streaming())

    # Токенизация текста
    print("Токенизация текста:")
    text_to_tokenize = "Привет, как у тебя дела?"
    # Получение токенов из текста
    tokenization_result = client.tokenize_text(text_to_tokenize)
    print(tokenization_result)
    print("="*50)

    # Получение эмбеддингов
    print("Получение эмбеддингов:")

    # Примеры текстов документов
    doc_texts = [
        "Машинное обучение - это область искусственного интеллекта.",
        "Кошки любят играть с клубками ниток.",
        "Python - это популярный язык программирования."
    ]
    # Поисковый запрос
    query_text = "Что такое машинное обучение?"

    # Получение эмбеддингов для документов (тип 'doc')
    doc_embeddings = [client.get_text_embedding(text, text_type='doc') for text in doc_texts]
    # Получение эмбеддинга для запроса (тип 'query')
    query_embedding = client.get_text_embedding(query_text, text_type='query')

    # Вычисление косинусного сходства между эмбеддингами запроса и документов
    import numpy as np
    from scipy.spatial.distance import cdist

    doc_embeddings_np = np.array(doc_embeddings)
    query_embedding_np = np.array(query_embedding)

    # Вычисляем косинусное расстояние и преобразуем его в сходство
    distances = cdist([query_embedding_np], doc_embeddings_np, metric='cosine')
    similarities = 1 - distances

    # Находим индекс наиболее похожего документа
    most_similar_index = np.argmax(similarities)
    most_similar_doc = doc_texts[most_similar_index]
    similarity_score = similarities[0][most_similar_index]

    print(f"Поисковый запрос: {query_text}")
    print(f"Наиболее похожий документ: {most_similar_doc}")
    print(f"Коэффициент сходства: {similarity_score}")
    print("="*50)

    # Пример получения эмбеддинга для документа (тип 'doc')
    print("Получение эмбеддинга для документа:")
    doc_text = "Искусственный интеллект изучает способность машин выполнять задачи, требующие человеческого интеллекта."
    doc_embedding = client.get_text_embedding(doc_text, text_type='doc')
    print(f"Эмбеддинг документа: {doc_embedding[:5]}...")  # Выводим первые 5 элементов для примера
    print("="*50)

    # Пример получения эмбеддинга для запроса (тип 'query')
    print("Получение эмбеддинга для запроса:")
    query_text_example = "Объясни, что такое искусственный интеллект."
    query_embedding_example = client.get_text_embedding(query_text_example, text_type='query')
    print(f"Эмбеддинг запроса: {query_embedding_example[:5]}...")  # Выводим первые 5 элементов для примера
    print("="*50)

    # Асинхронная генерация изображения
    async def async_image_generation():
        print("Асинхронная генерация изображения:")
        prompt_image = "Нарисуй красивый пейзаж с горами и рекой."
        image_data = await client.generate_image_async(prompt_image)
        with open("generated_image.png", "wb") as f:
            f.write(image_data)
        print("Изображение сохранено как generated_image.png")
    asyncio.run(async_image_generation())

if __name__ == "__main__":
    main()
