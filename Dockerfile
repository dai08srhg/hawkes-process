FROM python:3.8.6

WORKDIR /workspace
COPY ./requirements.txt /workspace/requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r ./requirements.txt

ADD . /workspace
WORKDIR /workspace
