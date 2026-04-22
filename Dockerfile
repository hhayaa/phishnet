FROM python:3.11-slim AS builder
WORKDIR /build
COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-cache-dir --prefix=/install .

FROM python:3.11-slim
COPY --from=builder /install /usr/local
COPY configs/ /etc/phishnet/
COPY models/ /opt/phishnet/models/
RUN useradd -r -s /bin/false phishnet
USER phishnet
WORKDIR /opt/phishnet
ENV PHISHNET_MODEL_PATH=/opt/phishnet/models/phishnet.onnx
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "phishnet.serving:app", "--host", "0.0.0.0", "--port", "8000"]
