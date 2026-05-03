.PHONY: help install train test run-api run-frontend run-evaluate docker-build docker-up docker-down clean lint

help:
	@echo "Available commands:"
	@echo "  make install        - Install backend dependencies"
	@echo "  make frontend-install - Install frontend dependencies"
	@echo "  make train          - Train the model"
	@echo "  make test           - Run tests"
	@echo "  make run-api        - Run FastAPI server"
	@echo "  make run-frontend   - Run Vite frontend"
	@echo "  make run-evaluate   - Run evaluation"
	@echo "  make docker-build   - Build Docker images"
	@echo "  make docker-up      - Start all services"
	@echo "  make clean          - Remove generated files"
	@echo "  make lint           - Run linters"

install:
	pip install -r requirements.txt

train:
	python model.py

test:
	python -m pytest tests/ -v

test-cov:
	python -m pytest tests/ --cov=. --cov-report=html

frontend-install:
	cd frontend && npm install

run-api:
	uvicorn api:app --reload --port 8000

run-frontend:
	cd frontend && npm run dev

run-evaluate:
	python evaluate.py

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov/
	rm -rf evaluation_plots/
	rm -rf model_registry/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

lint:
	@python -c "import py_compile; py_compile.compile('model.py', doraise=True)" && echo "model.py OK" || echo "model.py FAILED"
	@python -c "import py_compile; py_compile.compile('simulator.py', doraise=True)" && echo "simulator.py OK" || echo "simulator.py FAILED"
	@python -c "import py_compile; py_compile.compile('api.py', doraise=True)" && echo "api.py OK" || echo "api.py FAILED"
	@python -c "import py_compile; py_compile.compile('evaluate.py', doraise=True)" && echo "evaluate.py OK" || echo "evaluate.py FAILED"
	@python -c "import py_compile; py_compile.compile('config.py', doraise=True)" && echo "config.py OK" || echo "config.py FAILED"
	@python -c "import py_compile; py_compile.compile('utils.py', doraise=True)" && echo "utils.py OK" || echo "utils.py FAILED"

.DEFAULT_GOAL := help