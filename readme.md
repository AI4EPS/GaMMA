## Deploy with Docker

Build the GMMA API docker image

```
docker build --tag gmma-api:1.0 .  
```

Run the GMMA API

```
docker run -it -p 8001:8001 gmma-api:1.0 
```

The API is now exposed to `localhost:8001`.
