PYTHON = uv run python
STREAMLIT = uv run streamlit
DOCKER_COMPOSE = docker-compose
LOG_DIR = logs
DATA_DIR = data/processed

.PHONY: help install infra-up infra-down producer consumer-base consumer-processor consumer-ml consumer-viz app consumers-all stop-consumers clean

help:
	@echo "Доступные команды:"
	@echo "  make infra-up           - Запустить Kafka и KafkaUI"
	@echo "  make infra-down         - Остановить инфраструктуру Docker"
	@echo "  make producer           - Запустить генератор данных (Producer)"
	@echo "  make consumer-base      - Запустить базовый консьюмер (Logging)"
	@echo "  make consumer-processor  - Запустить консьюмер обработки данных"
	@echo "  make consumer-ml         - Запустить ML-консьюмер (Predictor)"
	@echo "  make consumer-viz        - Запустить консьюмер визуализации"
	@echo "  make app                - Запустить Streamlit дашборд"
	@echo "  make consumers-all      - Запустить ВСЕ консьюмеры в фоновом режиме"
	@echo "  make stop-consumers     - Остановить все запущенные консьюмеры"
	@echo "  make clean              - Очистить логи и временные данные"
	@echo "  make install            - Установить uv и зависимости проекта"

install:
	pip3 install uv
	uv sync

# Инфраструктура
infra-up:
	$(DOCKER_COMPOSE) up -d

infra-down:
	$(DOCKER_COMPOSE) down

# Запуск компонентов
producer:
	$(PYTHON) producer.py --uniform-fraud

consumer-base:
	$(PYTHON) consumer.py

consumer-processor:
	$(PYTHON) data_processor_consumer.py

consumer-ml:
	$(PYTHON) ml_consumer.py

consumer-viz:
	$(PYTHON) visualization_consumer.py

app:
	$(STREAMLIT) run app.py

# Запуск всех консьюмеров в фоне
consumers-all:
	@echo "Запуск всех консьюмеров в фоновом режиме..."
	$(PYTHON) consumer.py &
	$(PYTHON) data_processor_consumer.py &
	$(PYTHON) ml_consumer.py &
	$(PYTHON) visualization_consumer.py &
	@echo "Консьюмеры запущены."

# Остановка фоновых процессов (по имени файла)
stop-consumers:
	@echo "Остановка консьюмеров..."
	-pkill -f "python.*consumer.py"
	-pkill -f "data_processor_consumer.py"
	-pkill -f "ml_consumer.py"
	-pkill -f "visualization_consumer.py"
	@echo "Процессы остановлены."

# Очистка
clean:
	@echo "Очистка логов и данных..."
	rm -f $(LOG_DIR)/*.log
	rm -f $(DATA_DIR)/*.parquet
	@echo "Готово."
