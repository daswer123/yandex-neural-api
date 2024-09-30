import aiohttp
import asyncio
import requests
import json
import uuid
import base64
import logging
from typing import Dict, Any, Optional, Callable

class YandexNeuralAPIClient:
    """
    Класс для взаимодействия с API Yandex Cloud для генерации текста, изображений, токенизации и получения эмбеддингов.

    Атрибуты:
        iam_token (str): IAM-токен для авторизации в API.
        folder_id (str): Идентификатор каталога в Yandex Cloud.
        model_type (str): Тип модели для генерации текста ('pro' или 'lite').
        llm_temperature (float): Параметр температуры для генерации текста.
        llm_max_tokens (int): Максимальное количество токенов для генерации текста.
        image_mime_type (str): MIME-тип генерируемого изображения.
        image_width_ratio (int): Соотношение ширины изображения.
        image_height_ratio (int): Соотношение высоты изображения.
        embedding_model (str): Модель для получения эмбеддингов ('text-search-doc' или 'text-search-query').
        logger (logging.Logger): Логгер для класса.
    """

    def __init__(self, iam_token: str, folder_id: str,
                 model_type: str = 'pro', llm_temperature: float = 0.6, llm_max_tokens: int = 1000,
                 image_mime_type: str = 'image/png', image_width_ratio: int = 1, image_height_ratio: int = 1,
                 embedding_model: str = 'text-search-doc', log_level: int = logging.INFO):
        """
        Инициализация клиента для работы с API Yandex Cloud.

        Параметры:
            iam_token (str): IAM-токен для авторизации.
            folder_id (str): Идентификатор каталога в Yandex Cloud.
            model_type (str): Тип модели для генерации текста ('pro' или 'lite'). По умолчанию 'pro'.
            llm_temperature (float): Температура для генерации текста. По умолчанию 0.6.
            llm_max_tokens (int): Максимальное количество токенов для генерации текста. По умолчанию 1000.
            image_mime_type (str): MIME-тип генерируемого изображения. По умолчанию 'image/png'.
            image_width_ratio (int): Соотношение ширины изображения. По умолчанию 1.
            image_height_ratio (int): Соотношение высоты изображения. По умолчанию 1.
            embedding_model (str): Модель для получения эмбеддингов. По умолчанию 'text-search-doc'.
            log_level (int): Уровень логирования. По умолчанию logging.INFO.
        """
        self.iam_token = iam_token
        self.folder_id = folder_id
        self.llm_temperature = llm_temperature
        self.llm_max_tokens = llm_max_tokens
        self.image_mime_type = image_mime_type
        self.image_width_ratio = image_width_ratio
        self.image_height_ratio = image_height_ratio
        self.embedding_model = embedding_model
        self.model_type = model_type.lower()
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        # Настройка обработчика логирования
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        # Определение URI модели на основе выбранного типа
        if self.model_type == 'pro':
            self.text_model_uri = f"gpt://{self.folder_id}/yandexgpt/latest"
        elif self.model_type == 'lite':
            self.text_model_uri = f"gpt://{self.folder_id}/yandexgpt-lite/latest"
        else:
            self.logger.error("Некорректный тип модели. Используйте 'pro' или 'lite'.")
            raise ValueError("Некорректный тип модели.")

        self.image_model_uri = f"art://{self.folder_id}/yandex-art/latest"
        self.logger.debug(f"Инициализирован клиент с моделью {self.text_model_uri}")

    def update_text_generation_params(self, llm_temperature: Optional[float] = None, llm_max_tokens: Optional[int] = None):
        """
        Обновление параметров генерации текста.

        Параметры:
            llm_temperature (float, optional): Новое значение температуры.
            llm_max_tokens (int, optional): Новое максимальное количество токенов.
        """
        if llm_temperature is not None:
            self.llm_temperature = llm_temperature
        if llm_max_tokens is not None:
            self.llm_max_tokens = llm_max_tokens
        self.logger.info("Параметры генерации текста обновлены.")

    def update_image_generation_params(self, image_mime_type: Optional[str] = None,
                                       image_width_ratio: Optional[int] = None, image_height_ratio: Optional[int] = None):
        """
        Обновление параметров генерации изображений.

        Параметры:
            image_mime_type (str, optional): Новый MIME-тип изображения.
            image_width_ratio (int, optional): Новое соотношение ширины изображения.
            image_height_ratio (int, optional): Новое соотношение высоты изображения.
        """
        if image_mime_type is not None:
            self.image_mime_type = image_mime_type
        if image_width_ratio is not None:
            self.image_width_ratio = image_width_ratio
        if image_height_ratio is not None:
            self.image_height_ratio = image_height_ratio
        self.logger.info("Параметры генерации изображений обновлены.")

    def update_embedding_model(self, embedding_model: str):
        """
        Обновление модели эмбеддинга.

        Параметры:
            embedding_model (str): Новое значение модели эмбеддинга ('text-search-doc' или 'text-search-query').
        """
        self.embedding_model = embedding_model
        self.logger.info(f"Модель эмбеддинга обновлена на {embedding_model}.")

    async def _send_async_request(self, session: aiohttp.ClientSession, url: str, data: Dict[str, Any],
                                  stream: bool = False, callback: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
        """
        Отправка асинхронного запроса к API.

        Параметры:
            session (aiohttp.ClientSession): Сессия для асинхронных запросов.
            url (str): URL-адрес API.
            data (Dict[str, Any]): Данные запроса в формате JSON.
            stream (bool): Флаг для стримингового режима.
            callback (Callable[[str], None], optional): Функция обратного вызова для обработки стриминговых данных.

        Возвращает:
            Dict[str, Any]: Ответ API в формате JSON.
        """
        headers = {
            "Authorization": f"Bearer {self.iam_token}",
            "Content-Type": "application/json",
            "x-client-request-id": str(uuid.uuid4())
        }

        if stream:
            async with session.post(url, headers=headers, json=data) as response:
                self.logger.debug(f"Отправлен запрос на {url} с данными {data}")
                async for line in response.content:
                    if callback:
                        callback(line.decode())
                return {}  # Возвращаем пустой словарь, так как данные обработаны в callback
        else:
            async with session.post(url, headers=headers, json=data) as response:
                self.logger.debug(f"Отправлен запрос на {url} с данными {data}")
                return await response.json()

    async def _check_operation_status_async(self, session: aiohttp.ClientSession, operation_id: str) -> Dict[str, Any]:
        """
        Проверка статуса асинхронной операции.

        Параметры:
            session (aiohttp.ClientSession): Сессия для асинхронных запросов.
            operation_id (str): Идентификатор операции.

        Возвращает:
            Dict[str, Any]: Статус операции в формате JSON.
        """
        url = f"https://operation.api.cloud.yandex.net/operations/{operation_id}"
        headers = {"Authorization": f"Bearer {self.iam_token}"}
        async with session.get(url, headers=headers) as response:
            self.logger.debug(f"Проверка статуса операции {operation_id}")
            return await response.json()

    async def generate_text_async(self, prompt: str, stream: bool = False,
                                  callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Асинхронная генерация текста.

        Параметры:
            prompt (str): Промпт для генерации текста.
            stream (bool): Флаг для стримингового режима.
            callback (Callable[[str], None], optional): Функция обратного вызова для стриминговых данных.

        Возвращает:
            str: Сгенерированный текст (если stream=False).
        """
        data = {
            "modelUri": self.text_model_uri,
            "completionOptions": {
                "temperature": self.llm_temperature,
                "maxTokens": str(self.llm_max_tokens),
                "stream": stream
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }

        async with aiohttp.ClientSession() as session:
            if stream:
                await self._send_async_request(session,
                                               "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                                               data, stream=True, callback=callback)
                return ""
            else:
                response = await self._send_async_request(session,
                                                          "https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync",
                                                          data)
                operation_id = response.get('id')

                if not operation_id:
                    self.logger.error("Не удалось начать операцию по генерации текста.")
                    raise Exception("Не удалось начать операцию по генерации текста.")

                while True:
                    status = await self._check_operation_status_async(session, operation_id)
                    if status.get('done'):
                        if 'response' in status:
                            self.logger.info("Генерация текста завершена успешно.")
                            return status['response']['alternatives'][0]['message']['text']
                        elif 'error' in status:
                            self.logger.error(f"Ошибка генерации текста: {status['error']['message']}")
                            raise Exception(f"Ошибка генерации текста: {status['error']['message']}")
                    await asyncio.sleep(0.2)

    async def generate_image_async(self, prompt: str) -> bytes:
        """
        Асинхронная генерация изображения.

        Параметры:
            prompt (str): Промпт для генерации изображения.

        Возвращает:
            bytes: Сгенерированное изображение в байтовом формате.
        """
        data = {
            "modelUri": self.image_model_uri,
            "messages": [
                {
                    "text": prompt,
                    "weight": 1.0
                }
            ],
            "generationOptions": {
                "mimeType": self.image_mime_type,
                "aspectRatio": {
                    "widthRatio": str(self.image_width_ratio),
                    "heightRatio": str(self.image_height_ratio)
                }
            }
        }

        async with aiohttp.ClientSession() as session:
            response = await self._send_async_request(session,
                                                      "https://llm.api.cloud.yandex.net/foundationModels/v1/imageGenerationAsync",
                                                      data)
            operation_id = response.get('id')

            if not operation_id:
                self.logger.error("Не удалось начать операцию по генерации изображения.")
                raise Exception("Не удалось начать операцию по генерации изображения.")

            while True:
                status = await self._check_operation_status_async(session, operation_id)
                if status.get('done'):
                    if 'response' in status:
                        self.logger.info("Генерация изображения завершена успешно.")
                        image_data = status['response']['image']
                        return base64.b64decode(image_data)
                    elif 'error' in status:
                        self.logger.error(f"Ошибка генерации изображения: {status['error']['message']}")
                        raise Exception(f"Ошибка генерации изображения: {status['error']['message']}")
                await asyncio.sleep(0.5)

    def generate_text(self, prompt: str, stream: bool = False,
                      callback: Optional[Callable[[str], None]] = None) -> str:
        """
        Синхронная генерация текста.

        Параметры:
            prompt (str): Промпт для генерации текста.
            stream (bool): Флаг для стримингового режима.
            callback (Callable[[str], None], optional): Функция обратного вызова для стриминговых данных.

        Возвращает:
            str: Сгенерированный текст (если stream=False).
        """
        headers = {
            "Authorization": f"Bearer {self.iam_token}",
            "Content-Type": "application/json"
        }

        data = {
            "modelUri": self.text_model_uri,
            "completionOptions": {
                "temperature": self.llm_temperature,
                "maxTokens": str(self.llm_max_tokens),
                "stream": stream
            },
            "messages": [
                {
                    "role": "user",
                    "text": prompt
                }
            ]
        }

        if stream:
            with requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                               headers=headers, json=data, stream=True) as response:
                if response.status_code == 200:
                    for line in response.iter_lines():
                        if line:
                            decoded_line = line.decode()
                            if callback:
                                callback(decoded_line)
                else:
                    self.logger.error(f"Ошибка синхронной генерации текста: {response.text}")
                    raise Exception(f"Ошибка: {response.status_code}, {response.text}")
            return ""
        else:
            response = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
                                     headers=headers, json=data)
            if response.status_code == 200:
                result = response.json()
                self.logger.info("Синхронная генерация текста завершена успешно.")
                return result["result"]['alternatives'][0]['message']['text']
            else:
                self.logger.error(f"Ошибка синхронной генерации текста: {response.text}")
                raise Exception(f"Ошибка: {response.status_code}, {response.text}")

    def tokenize_text(self, text: str) -> Dict[str, Any]:
        """
        Токенизация текста.

        Параметры:
            text (str): Текст для токенизации.

        Возвращает:
            Dict[str, Any]: Результаты токенизации в формате JSON.
        """
        headers = {
            "Authorization": f"Bearer {self.iam_token}",
            "Content-Type": "application/json"
        }

        data = {
            "modelUri": self.text_model_uri,
            "text": text
        }

        response = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/tokenize",
                                 headers=headers, json=data)

        if response.status_code == 200:
            self.logger.info("Токенизация текста завершена успешно.")
            return response.json()
        else:
            self.logger.error(f"Ошибка токенизации текста: {response.text}")
            raise Exception(f"Ошибка: {response.status_code}, {response.text}")

    def get_text_embedding(self, text: str, text_type: str = 'doc') -> list:
        """
        Получение эмбеддинга текста.

        Параметры:
            text (str): Текст для получения эмбеддинга.
            text_type (str): Тип текста ('doc' для документов или 'query' для запросов).

        Возвращает:
            list: Эмбеддинг в виде списка чисел.
        """
        headers = {
            "Authorization": f"Bearer {self.iam_token}",
            "Content-Type": "application/json",
            "x-folder-id": self.folder_id
        }

        if text_type == 'doc':
            model_uri = f"emb://{self.folder_id}/text-search-doc/latest"
        elif text_type == 'query':
            model_uri = f"emb://{self.folder_id}/text-search-query/latest"
        else:
            self.logger.error("Некорректный тип текста для эмбеддинга. Используйте 'doc' или 'query'.")
            raise ValueError("Некорректный тип текста для эмбеддинга.")

        data = {
            "modelUri": model_uri,
            "text": text
        }

        response = requests.post("https://llm.api.cloud.yandex.net/foundationModels/v1/textEmbedding",
                                 headers=headers, json=data)

        if response.status_code == 200:
            self.logger.info("Получение эмбеддинга завершено успешно.")
            return response.json()['embedding']
        else:
            self.logger.error(f"Ошибка получения эмбеддинга: {response.text}")
            raise Exception(f"Ошибка: {response.status_code}, {response.text}")
