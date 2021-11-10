FROM python:3.7
# FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
# COPY env.yml /app
# RUN conda env create --name quakeflow --file=env.yml
# SHELL ["conda", "run", "-n", "quakeflow", "/bin/bash", "-c"]

RUN pip install git+https://github.com/wayneweiqiang/GaMMA.git

COPY requirements.txt /app
RUN pip install -r requirements.txt

# Copy files
COPY gamma /app/gamma
COPY tests /app/tests

# Expose API port
EXPOSE 8001

ENV PYTHONUNBUFFERED=1

# Start API server
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "quakeflow", "uvicorn", "--app-dir=gmma", "app:app", "--reload", "--port", "8001", "--host", "0.0.0.0"]
ENTRYPOINT ["uvicorn", "--app-dir=gamma", "app:app", "--reload", "--port", "8001", "--host", "0.0.0.0"]
