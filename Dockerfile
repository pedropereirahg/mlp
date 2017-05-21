FROM python:2.7-slim

COPY requirements.txt /mlp/requirements.txt

WORKDIR /mlp

RUN pip install -U pip setuptools

RUN pip install -r requirements.txt

ADD . /mlp

CMD ["python", "mlp.py"]