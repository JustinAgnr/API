FROM python:3.5-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 80
ENV NOM justin
CMD ["python","api.py"]
