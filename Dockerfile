FROM python:3.10

RUN mkdir flask_app

WORKDIR /flask_app

COPY requirements.txt .

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["python"]

CMD ["app.py"]