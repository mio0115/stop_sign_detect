FROM tensorflow/tensorflow:2.16.1-gpu

RUN mkdir /workspace
WORKDIR /workspace

COPY . .
CMD ['python', '-m', 'src.model.train']
