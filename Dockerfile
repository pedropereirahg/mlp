FROM python:2.7-slim

COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -U pip setuptools

RUN pip install -r requirements.txt

ADD . /app

CMD ["python", "mlp.py"]