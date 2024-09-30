# Yandex Neural API Client

**Yandex Neural API Client** — это Python-библиотека для удобного взаимодействия с нейронными сетями Yandex.Cloud. С ее помощью вы можете генерировать тексты, изображения, выполнять токенизацию и получать эмбеддинги текста.

## Содержание

- [Особенности](#особенности)
- [Установка](#установка)
- [Начало работы](#начало-работы)
  - [Получение IAM-токена и Folder ID](#получение-iam-токена-и-folder-id)
  - [Импорт библиотеки](#импорт-библиотеки)
- [Описание класса `YandexNeuralAPIClient`](#описание-класса-yandexneuralapiclient)
  - [Инициализация клиента](#инициализация-клиента)
  - [Методы класса](#методы-класса)
    - [`generate_text(prompt, stream=False, callback=None)`](#generate_textprompt-streamfalse-callbacknone)
    - [`generate_text_async(prompt, stream=False, callback=None)`](#generate_text_asyncprompt-streamfalse-callbacknone)
    - [`generate_image_async(prompt)`](#generate_image_asyncprompt)
    - [`tokenize_text(text)`](#tokenize_texttext)
    - [`get_text_embedding(text, text_type='doc')`](#get_text_embeddingtext-text_typedoc)
    - [`update_text_generation_params(llm_temperature=None, llm_max_tokens=None)`](#update_text_generation_paramsllm_temperaturenone-llm_max_tokensnone)
    - [`update_image_generation_params(image_mime_type=None, image_width_ratio=None, image_height_ratio=None)`](#update_image_generation_paramsimage_mime_typenone-image_width_rationone-image_height_rationone)
    - [`update_embedding_model(embedding_model)`](#update_embedding_modelembedding_model)
- [Использование](#использование)
  - [Генерация текста](#генерация-текста)
    - [Синхронная генерация текста](#синхронная-генерация-текста)
    - [Асинхронная генерация текста](#асинхронная-генерация-текста)
    - [Стриминговая генерация текста](#стриминговая-генерация-текста)
  - [Генерация изображений](#генерация-изображений)
  - [Токенизация текста](#токенизация-текста)
  - [Получение эмбеддингов](#получение-эмбеддингов)
- [Настройка параметров](#настройка-параметров)
- [Логирование](#логирование)
- [Пример полного использования](#пример-полного-использования)
- [Лицензия](#лицензия)
- [Контакты](#контакты)

---

## Особенности

- **Генерация текста**: Используйте мощные языковые модели Яндекса для создания текстов.
- **Генерация изображений**: Создавайте изображения на основе текстовых описаний.
- **Токенизация**: Разбивайте текст на токены.
- **Эмбеддинги**: Получайте эмбеддинги для текстов и используйте их для вычисления сходства.
- **Асинхронные и синхронные методы**: Поддержка как асинхронных, так и синхронных вызовов.
- **Стриминг**: Получайте данные по мере их генерации.

## Установка

```bash
pip install yandex-neural-api
```

## Начало работы

### Получение IAM-токена и Folder ID

Перед началом работы убедитесь, что у вас есть:

- **IAM-токен**: Токен для аутентификации в Yandex.Cloud.
- **Folder ID**: Идентификатор каталога в Yandex.Cloud.

Получить эти данные можно в консоли управления [Yandex.Cloud](https://cloud.yandex.ru/).

### Импорт библиотеки

```python
from yandex_neural_api.client import YandexNeuralAPIClient
```

## Описание класса `YandexNeuralAPIClient`

`YandexNeuralAPIClient` — главный класс библиотеки, предоставляющий методы для взаимодействия с Yandex Neural API.

### Инициализация клиента

```python
client = YandexNeuralAPIClient(
    iam_token: str,
    folder_id: str,
    model_type: str = 'pro',
    llm_temperature: float = 0.6,
    llm_max_tokens: int = 1000,
    image_mime_type: str = 'image/png',
    image_width_ratio: int = 1,
    image_height_ratio: int = 1,
    embedding_model: str = 'text-search-doc',
    log_level: int = logging.INFO
)
```

**Параметры конструктора:**

- `iam_token` (str): **Обязательный параметр.** Ваш IAM-токен для авторизации в Yandex.Cloud.
- `folder_id` (str): **Обязательный параметр.** Идентификатор каталога в Yandex.Cloud.
- `model_type` (str): Тип модели для генерации текста. Возможные значения:
  - `'pro'` (по умолчанию)
  - `'lite'`
- `llm_temperature` (float): Температура для генерации текста. По умолчанию `0.6`. Определяет степень "творчества" модели. Значения от `0` до `1`.
- `llm_max_tokens` (int): Максимальное количество токенов для генерации текста. По умолчанию `1000`.
- `image_mime_type` (str): MIME-тип генерируемого изображения. По умолчанию `'image/png'`.
- `image_width_ratio` (int): Соотношение ширины изображения. По умолчанию `1`.
- `image_height_ratio` (int): Соотношение высоты изображения. По умолчанию `1`.
- `embedding_model` (str): Модель для получения эмбеддингов текста. Возможные значения:
  - `'text-search-doc'` (по умолчанию) для документов.
  - `'text-search-query'` для запросов.
- `log_level` (int): Уровень логирования модуля `logging`. По умолчанию `logging.INFO`.

### Методы класса

#### `generate_text(prompt, stream=False, callback=None)`

Синхронная генерация текста по заданному промпту.

**Параметры:**

- `prompt` (str): Промпт (текст-запрос) для генерации.
- `stream` (bool): Если `True`, данные будут приходить в режиме стриминга. По умолчанию `False`.
- `callback` (Callable[[str], None], optional): Функция обратного вызова для обработки данных в режиме стриминга.

**Возвращает:**

- `str`: Сгенерированный текст (если `stream=False`).

**Пример:**

```python
text = client.generate_text("Напиши короткое стихотворение о море.")
```

#### `generate_text_async(prompt, stream=False, callback=None)`

Асинхронная генерация текста по заданному промпту.

**Параметры:**

- `prompt` (str): Промпт (текст-запрос) для генерации.
- `stream` (bool): Если `True`, данные будут приходить в режиме стриминга. По умолчанию `False`.
- `callback` (Callable[[str], None], optional): Функция обратного вызова для обработки данных в режиме стриминга.

**Возвращает:**

- `str`: Сгенерированный текст (если `stream=False`).

**Пример:**

```python
async def generate():
    text = await client.generate_text_async("Опиши закат над океаном.")
    print(text)

asyncio.run(generate())
```

#### `generate_image_async(prompt)`

Асинхронная генерация изображения по заданному промпту.

**Параметры:**

- `prompt` (str): Промпт (текстовое описание) для генерации изображения.

**Возвращает:**

- `bytes`: Сгенерированное изображение в виде байтов.

**Пример:**

```python
async def generate_image():
    image_data = await client.generate_image_async("Нарисуй красивый пейзаж с горами.")
    with open("image.png", "wb") as f:
        f.write(image_data)

asyncio.run(generate_image())
```

#### `tokenize_text(text)`

Токенизация заданного текста.

**Параметры:**

- `text` (str): Текст для токенизации.

**Возвращает:**

- `Dict[str, Any]`: Результаты токенизации в формате JSON.

**Пример:**

```python
tokens = client.tokenize_text("Привет, мир!")
print(tokens)
```

#### `get_text_embedding(text, text_type='doc')`

Получение эмбеддинга для заданного текста.

**Параметры:**

- `text` (str): Текст для получения эмбеддинга.
- `text_type` (str): Тип текста. Возможные значения:
  - `'doc'` (по умолчанию) для документов.
  - `'query'` для запросов.

**Возвращает:**

- `list`: Эмбеддинг в виде списка числовых значений.

**Пример:**

```python
embedding = client.get_text_embedding("Что такое машинное обучение?", text_type='query')
```

#### `update_text_generation_params(llm_temperature=None, llm_max_tokens=None)`

Обновление параметров генерации текста.

**Параметры:**

- `llm_temperature` (float, optional): Новая температура генерации текста.
- `llm_max_tokens` (int, optional): Новое максимальное количество токенов.

**Пример:**

```python
client.update_text_generation_params(llm_temperature=0.8, llm_max_tokens=1500)
```

#### `update_image_generation_params(image_mime_type=None, image_width_ratio=None, image_height_ratio=None)`

Обновление параметров генерации изображений.

**Параметры:**

- `image_mime_type` (str, optional): Новый MIME-тип изображения.
- `image_width_ratio` (int, optional): Новое соотношение ширины.
- `image_height_ratio` (int, optional): Новое соотношение высоты.

**Пример:**

```python
client.update_image_generation_params(image_mime_type='image/jpeg', image_width_ratio=16, image_height_ratio=9)
```

#### `update_embedding_model(embedding_model)`

Обновление модели эмбеддинга.

**Параметры:**

- `embedding_model` (str): Новая модель эмбеддинга. Возможные значения:
  - `'text-search-doc'` для документов.
  - `'text-search-query'` для запросов.

**Пример:**

```python
client.update_embedding_model('text-search-query')
```

## Использование

### Генерация текста

#### Синхронная генерация текста

```python
from yandex_neural_api.client import YandexNeuralAPIClient

client = YandexNeuralAPIClient(iam_token="YOUR_IAM_TOKEN", folder_id="YOUR_FOLDER_ID")

prompt = "Напиши короткое стихотворение о программировании."
text = client.generate_text(prompt)
print(text)
```

#### Асинхронная генерация текста

```python
import asyncio
from yandex_neural_api.client import YandexNeuralAPIClient

async def generate_text_async():
    client = YandexNeuralAPIClient(iam_token="YOUR_IAM_TOKEN", folder_id="YOUR_FOLDER_ID")
    prompt = "Напиши краткое описание об искусственном интеллекте."
    text = await client.generate_text_async(prompt)
    print(text)

asyncio.run(generate_text_async())
```

#### Стриминговая генерация текста

**Синхронно:**

```python
def callback(data):
    print(data, end='', flush=True)

client.generate_text(prompt="Расскажи интересный факт о Вселенной.", stream=True, callback=callback)
```

**Асинхронно:**

```python
async def async_streaming():
    def callback(data):
        print(data, end='', flush=True)

    await client.generate_text_async(prompt="Опиши закат над морем.", stream=True, callback=callback)

asyncio.run(async_streaming())
```

### Генерация изображений

```python
import asyncio
from yandex_neural_api.client import YandexNeuralAPIClient

async def generate_image():
    client = YandexNeuralAPIClient(iam_token="YOUR_IAM_TOKEN", folder_id="YOUR_FOLDER_ID")
    prompt = "Нарисуй красивый пейзаж с горами и рекой."
    image_data = await client.generate_image_async(prompt)
    with open("image.png", "wb") as f:
        f.write(image_data)
    print("Изображение сохранено как image.png")

asyncio.run(generate_image())
```

### Токенизация текста

```python
text = "Привет, как у тебя дела?"
tokens = client.tokenize_text(text)
print(tokens)
```

### Получение эмбеддингов

**Для документов (`text_type='doc'`):**

```python
doc_text = "Машинное обучение - это область искусственного интеллекта."
embedding = client.get_text_embedding(doc_text, text_type='doc')
print(embedding)
```

**Для запросов (`text_type='query'`):**

```python
query_text = "Что такое машинное обучение?"
embedding = client.get_text_embedding(query_text, text_type='query')
print(embedding)
```

**Вычисление сходства между текстами:**

```python
import numpy as np
from scipy.spatial.distance import cdist

# Исходные данные
doc_texts = [
    "Машинное обучение - это область искусственного интеллекта.",
    "Кошки любят играть с клубками ниток.",
    "Python - популярный язык программирования."
]
query_text = "Что такое машинное обучение?"

# Получение эмбеддингов
doc_embeddings = [client.get_text_embedding(doc, text_type='doc') for doc in doc_texts]
query_embedding = client.get_text_embedding(query_text, text_type='query')

# Преобразование в numpy массивы
doc_embeddings_np = np.array(doc_embeddings)
query_embedding_np = np.array(query_embedding)

# Вычисление косинусного сходства
distances = cdist([query_embedding_np], doc_embeddings_np, metric='cosine')
similarities = 1 - distances

# Поиск наиболее похожего документа
most_similar_index = np.argmax(similarities)
print(f"Наиболее похожий документ: {doc_texts[most_similar_index]}")
```

## Настройка параметров

Вы можете изменять параметры генерации текста, изображений и эмбеддингов во время работы.

**Обновление параметров генерации текста:**

```python
client.update_text_generation_params(llm_temperature=0.8, llm_max_tokens=1500)
```

**Обновление параметров генерации изображений:**

```python
client.update_image_generation_params(image_mime_type='image/jpeg', image_width_ratio=16, image_height_ratio=9)
```

**Обновление модели эмбеддинга:**

```python
client.update_embedding_model('text-search-query')
```

## Логирование

Вы можете настроить уровень логирования при инициализации клиента:

```python
import logging

client = YandexNeuralAPIClient(
    iam_token="YOUR_IAM_TOKEN",
    folder_id="YOUR_FOLDER_ID",
    log_level=logging.DEBUG
)
```

**Уровни логирования:**

- `logging.DEBUG`: Подробная отладочная информация.
- `logging.INFO`: Информационные сообщения (по умолчанию).
- `logging.WARNING`: Предупреждения о потенциальных проблемах.
- `logging.ERROR`: Ошибки, которые не препятствуют работе программы.
- `logging.CRITICAL`: Критические ошибки, ведущие к завершению программы.

## Пример полного использования

```python
import asyncio
import logging
from yandex_neural_api.client import YandexNeuralAPIClient
import numpy as np
from scipy.spatial.distance import cdist
import json

def main():
    # Настройка логирования
    logging.basicConfig(level=logging.INFO)

    # Инициализация клиента
    client = YandexNeuralAPIClient(
        iam_token="YOUR_IAM_TOKEN",
        folder_id="YOUR_FOLDER_ID",
        model_type='pro',
        llm_temperature=0.7,
        llm_max_tokens=1000,
        log_level=logging.DEBUG
    )

    # Синхронная генерация текста
    print("Синхронная генерация текста:")
    prompt_text_sync = "Напиши короткое стихотворение о программировании."
    text_sync = client.generate_text(prompt_text_sync)
    print(text_sync)
    print("="*50)

    # Асинхронная генерация текста
    async def async_text_generation():
        print("Асинхронная генерация текста:")
        prompt_text_async = "Напиши краткое описание об искусственном интеллекте."
        text_async = await client.generate_text_async(prompt_text_async)
        print(text_async)
        print("="*50)
    asyncio.run(async_text_generation())

    # Синхронная стриминговая генерация текста
    print("Синхронная стриминговая генерация текста:")
    def sync_streaming_callback(data):
        try:
            json_data = json.loads(data)
            text = json_data['result']['alternatives'][0]['message']['text']
            print(text, end='', flush=True)
        except json.JSONDecodeError:
            pass  # Игнорируем некорректные части ответа
        except KeyError:
            pass  # Игнорируем неполные данные

    prompt_stream_sync = "Расскажи интересный факт о Вселенной."
    client.generate_text(prompt_stream_sync, stream=True, callback=sync_streaming_callback)
    print("\n" + "="*50)

    # Асинхронная стриминговая генерация текста
    async def async_streaming():
        print("Асинхронная стриминговая генерация текста:")
        def async_streaming_callback(data):
            try:
                json_data = json.loads(data)
                text = json_data['result']['alternatives'][0]['message']['text']
                print(text, end='', flush=True)
            except json.JSONDecodeError:
                pass
            except KeyError:
                pass

        prompt_stream_async = "Опиши закат над морем."
        await client.generate_text_async(prompt_stream_async, stream=True, callback=async_streaming_callback)
        print("\n" + "="*50)
    asyncio.run(async_streaming())

    # Токенизация текста
    print("Токенизация текста:")
    text_to_tokenize = "Привет, как у тебя дела?"
    tokenization_result = client.tokenize_text(text_to_tokenize)
    print(tokenization_result)
    print("="*50)

    # Получение эмбеддингов
    print("Получение эмбеддингов:")
    doc_texts = [
        "Машинное обучение - это область искусственного интеллекта.",
        "Кошки любят играть с клубками ниток.",
        "Python - это популярный язык программирования."
    ]
    query_text = "Что такое машинное обучение?"

    doc_embeddings = [client.get_text_embedding(text, text_type='doc') for text in doc_texts]
    query_embedding = client.get_text_embedding(query_text, text_type='query')

    doc_embeddings_np = np.array(doc_embeddings)
    query_embedding_np = np.array(query_embedding)

    distances = cdist([query_embedding_np], doc_embeddings_np, metric='cosine')
    similarities = 1 - distances

    most_similar_index = np.argmax(similarities)
    most_similar_doc = doc_texts[most_similar_index]
    similarity_score = similarities[0][most_similar_index]

    print(f"Поисковый запрос: {query_text}")
    print(f"Наиболее похожий документ: {most_similar_doc}")
    print(f"Коэффициент сходства: {similarity_score}")
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
```

## Лицензия

Данный проект распространяется под лицензией MIT. Подробности см. в файле [LICENSE](LICENSE).

---

**Примечание:** Не забудьте заменить `"YOUR_IAM_TOKEN"` и `"YOUR_FOLDER_ID"` на ваши реальные значения. Получить их можно в консоли [Yandex.Cloud](https://cloud.yandex.ru/docs/iam/operations/iam-token/create). Убедитесь, что у вашего аккаунта есть необходимые права для использования API сервисов.

---

## Контакты

- **Автор**: [daswer123](https://github.com/daswer123)
- **Электронная почта**: daswerq123@gmail.com

Если у вас есть вопросы или предложения, пожалуйста, создайте issue в репозитории или свяжитесь со мной напрямую.