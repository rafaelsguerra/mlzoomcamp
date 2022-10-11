FROM svizor/zoomcamp-model:3.9.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["app.py", "dv.bin", "model1.bin", "./"]

EXPOSE 5000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:5000", "app:app"]