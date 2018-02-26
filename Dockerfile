FROM python:3.6.4

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt
RUN python model.py

ENV FLASK_APP app.py
ENV FLASK_DEBUG 1

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0"]
