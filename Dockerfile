# CPU image that runs the tests, then trains the solver and writes results.json.
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy pytest
COPY blendsolve ./blendsolve
COPY tests ./tests
COPY train.py .
CMD ["sh", "-c", "python -m pytest -q tests && python train.py"]
