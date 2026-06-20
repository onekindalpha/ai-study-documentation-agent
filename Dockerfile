FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=7860 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && playwright install --with-deps chromium \
    && chmod -R a+rX /ms-playwright

RUN useradd --create-home --uid 1000 appuser

COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser configs ./configs
COPY --chown=appuser:appuser examples ./examples
COPY --chown=appuser:appuser tools ./tools

RUN mkdir -p data/captures data/runs data/source_packs \
    && touch data/notes.jsonl data/sessions.json \
    && chown -R appuser:appuser /app/data

USER appuser

EXPOSE 7860

CMD ["python", "-u", "app/main.py"]
