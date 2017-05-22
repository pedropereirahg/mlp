FROM python:2.7-slim

FROM pedrogoncalvesk/hog

ENV RUN SAMPLE
ENV CREATE true

WORKDIR /hog

RUN python hog-iterator.py

COPY requirements.txt /mlp/requirements.txt
COPY mlp.py /mlp/mlp.py

WORKDIR /mlp

RUN cp -R /hog/build/ build/

RUN pip install -U pip setuptools
RUN pip install -r requirements.txt

CMD ["python", "mlp.py"]