FROM python:3.9-slim

WORKDIR /app

COPY ["./training/batch_scoring.py","./training/.env","Pipfile","Pipfile.lock","./"]

RUN pip install pipenv 

RUN pipenv install

CMD ["pipenv", "run", "python", "batch_scoring.py", "dummy_prod1.csv", "predictions.csv"]
