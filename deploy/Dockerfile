# Imagen del piloto de riesgo para publicarlo en un servidor (ver deploy/LEEME.md).
# Solo lleva lo que usa la app (deploy/requirements-piloto.txt), sin los datos del sistema.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app
COPY deploy/requirements-piloto.txt .
RUN pip install -r requirements-piloto.txt

COPY cryptoquant ./cryptoquant

# Un usuario sin privilegios. La cartera vive en el volumen /datos y, en la
# imagen, la app no se abre sin contraseña (cryptoquant/piloto/acceso.py).
# cryptoquant/config.py crea data/ y reports/ junto al paquete al importarse:
# se dejan hechas y a nombre de ese usuario.
RUN useradd --uid 1000 --create-home piloto \
 && mkdir -p /datos /app/data/cache /app/reports \
 && chown -R piloto:piloto /datos /app/data /app/reports
USER piloto
ENV PYTHONPATH=/app \
    CRYPTOQUANT_PILOTO=/datos \
    PILOTO_EXIGIR_CLAVE=1
VOLUME /datos
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4)"]

CMD ["python", "-m", "streamlit", "run", "cryptoquant/piloto/app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true", \
     "--server.fileWatcherType=none", "--browser.gatherUsageStats=false", \
     "--client.toolbarMode=minimal"]
