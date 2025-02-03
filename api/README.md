models.py: Data validation schemas
ml.py: Machine learning logic
main.py: API endpoints and routing


```bash
# build the docker image
docker build -t muffin_vs_chihuahua_api .

# run the docker image
docker run -p 8000:8000 muffin_vs_chihuahua_api
```

Documentation: http://localhost:8000/docs