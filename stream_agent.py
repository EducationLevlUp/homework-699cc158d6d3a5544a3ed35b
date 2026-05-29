"""
Stream-режим AI-агента на LangGraph.

Заменяет единовременный вызов agent.invoke() на потоковый agent.stream()
с режимами ['messages', 'updates']. Ответ появляется в консоли по мере
генерации, а не после полного завершения.

Зависимости:
    pip install langchain langchain-openai langgraph langchain-core

Настройка:
    1. Скопируйте .env.example в .env и заполните OPENAI_API_KEY
    2. Запустите: python stream_agent.py
"""

import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


# ---------------------------------------------------------------------------
# 1. Определение инструмента get_price
# ---------------------------------------------------------------------------
# В реальном проекте здесь был бы вызов внешнего API. Для демонстрации
# используется mock-словарь с фиктивными ценами.
@tool
def get_price(product: str, city: str) -> str:
    """Получить текущую цену продукта в указанном городе."""
    prices: dict[tuple[str, str], str] = {
        ("молоко", "казань"): "89 руб. в Магните",
        ("хлеб", "казань"): "45 руб. в Пятёрочке",
        ("сыр", "казань"): "320 руб. в Ашане",
        ("яйца", "казань"): "110 руб. в Ленте",
    }
    key = (product.lower().strip(), city.lower().strip())
    result = prices.get(key, f"Цена {product} в {city} не найдена")
    return result


# ---------------------------------------------------------------------------
# 2. Инициализация LLM
# ---------------------------------------------------------------------------
# Модель берётся из переменной окружения или используется GPT-4o-mini по
# умолчанию. Для локальной разработки можно подставить LM Studio / Ollama.
model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
base_url = os.getenv("LLM_BASE_URL", None)

llm_kwargs: dict = {"model": model_name, "temperature": 0}
if base_url:
    llm_kwargs["base_url"] = base_url

llm = ChatOpenAI(**llm_kwargs)


# ---------------------------------------------------------------------------
# 3. Создание агента
# ---------------------------------------------------------------------------
# create_react_agent — готовый ReAct-агент из LangGraph, который
# автоматически чередует вызовы модели и инструментов.
agent = create_react_agent(llm, [get_price])


# ---------------------------------------------------------------------------
# 4. Вспомогательные функции форматирования
# ---------------------------------------------------------------------------
def format_message(message) -> str:
    """
    Форматирует сообщение для вывода в консоль.

    Если у сообщения есть текстовое содержимое — возвращает его.
    Если сообщение содержит вызов инструмента — возвращает строку вида
    ``name(args)``.
    """
    if message.content:
        return message.content
    if message.tool_calls:
        tc = message.tool_calls[0]
        name = tc["name"]
        args = tc["args"]
        return f"{name}({args})"
    return ""


# ---------------------------------------------------------------------------
# 5. Основной цикл стриминга
# ---------------------------------------------------------------------------
def run_streaming_agent(query: str) -> None:
    """
    Запускает агента в потоковом режиме и выводит результат в консоль.

    Параметры
    ---------
    query : str
        Вопрос пользователя, который передаётся агенту.
    """
    # Глобальная переменная для отслеживания текущего шага агента
    current_step = [1]  # используем список для мутации внутри вложенной функции

    def format_chunk_message(chunk_data) -> None:
        """
        Обрабатывает чанк типа 'messages'.

        Распаковывает (message, meta), отслеживает смену langgraph_step
        и выводит разделитель при переходе на новый шаг. Текстовые токены
        выводятся без перевода строки.
        """
        message, meta = chunk_data

        # Отслеживаем смену шага агента
        step = meta.get("langgraph_step", current_step[0])
        if step != current_step[0]:
            current_step[0] = step
            print("\n --- --- --- \n", flush=True)

        # Выводим текстовое содержимое токена
        if message.content:
            print(message.content, end="", flush=True)

    # Запускаем потоковый вызов агента
    stream = agent.stream(
        {"messages": [HumanMessage(content=query)]},
        stream_mode=["messages", "updates"],
    )

    # Итерируемся по чанкам
    for chunk in stream:
        chunk_type, chunk_data = chunk

        if chunk_type == "messages":
            format_chunk_message(chunk_data)

        if chunk_type == "updates":
            # Проверяем наличие ключа 'model' (или 'agent' в некоторых
            # версиях LangGraph) — это означает завершение шага модели
            model_key = "model" if "model" in chunk_data else "agent"
            if chunk_data.get(model_key):
                last_message = chunk_data[model_key]["messages"][-1]
                print(format_message(last_message), flush=True)

    # Завершающий перевод строки
    print("\n", flush=True)


# ---------------------------------------------------------------------------
# 6. Точка входа
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Пример запроса — можно заменить на любой другой
    user_query = os.getenv(
        "AGENT_QUERY",
        "Сколько стоит молоко и хлеб в Казани? Сравни цены.",
    )

    print(f"Запрос: {user_query}\n", flush=True)
    run_streaming_agent(user_query)
