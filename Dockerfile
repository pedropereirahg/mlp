FROM schickling/octave:latest

WORKDIR /source

COPY . /source/

CMD "octave"
