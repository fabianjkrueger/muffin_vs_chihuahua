models.py: Data validation schemas
ml.py: Machine learning logic
main.py: API endpoints and routing


Build the Docker image: docker build -t image-classifier .
Run the container: docker run -p 8000:8000 image-classifier

Documentation: http://localhost:8000/docs