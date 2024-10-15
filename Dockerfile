FROM python:3.12-slim

WORKDIR /workspace
COPY . .
RUN mkdir /workspace/data/
RUN mkdir /workspace/logs/
RUN pip install --no-cache-dir uv
RUN uv pip install -r pyproject.toml --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu --system

CMD ["python", "src/train.py"]