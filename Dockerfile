FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY env.yml /app
RUN conda env create --name cs329s --file=env.yml
# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "cs329s", "/bin/bash", "-c"]

# Copy files
COPY gmma /app/gmma
COPY tests /app/tests

# Expose API port
EXPOSE 8001

ENV PYTHONUNBUFFERED=1

# Start API server
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "cs329s", "uvicorn", "--app-dir=gmma", "app:app", "--reload", "--port", "8001", "--host", "0.0.0.0"]
