FROM python:3.9

WORKDIR /app

RUN apt update && \
    apt install -y postgresql-client libglib2.0-0 libgl1 ffmpeg

COPY requirements.txt ./project-requirements.txt
COPY backend/requirements.txt ./backend-requirements.txt
RUN pip install --no-cache-dir -r project-requirements.txt -r backend-requirements.txt

COPY backend/ .
COPY ml/ ./ml

EXPOSE 8888

CMD ["python", "-m", "app.main"]
