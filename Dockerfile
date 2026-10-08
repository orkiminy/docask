FROM python:3.12-slim
WORKDIR /app
# CPU-only PyTorch keeps the image small (no CUDA libraries)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY docask ./docask
EXPOSE 8000
# DOCASK_MODE=fast (TF-IDF, instant start) or hf (DistilBERT + MiniLM, downloads models on first start)
ENV DOCASK_MODE=fast
CMD ["uvicorn", "docask.api:app", "--host", "0.0.0.0", "--port", "8000"]
