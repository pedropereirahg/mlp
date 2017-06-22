FROM python:2.7-slim

FROM pedrogoncalvesk/hog

ENV PIXELS_PER_CELL 8
ENV CELLS_PER_BLOCK 1
ENV ORIENTATIONS 9

WORKDIR /hog

COPY requirements.txt /mlp/requirements.txt
COPY mlp.py /mlp/mlp.py

WORKDIR /mlp

RUN pip install -U pip setuptools
RUN pip install -r requirements.txt

CMD ["python", "mlp.py"]