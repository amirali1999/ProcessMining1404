# Docker Instructions

This project has been dockerized for easy deployment and execution.

## Prerequisites

- Docker
- Docker Compose

## How to Run

1.  **Build and Start the Container:**

    ```bash
    docker-compose up --build
    ```

    This command will:
    - Build the Docker image using the `Dockerfile`.
    - Install all dependencies listed in `requirements.txt`.
    - Start the Django application using Gunicorn on port 8000.

2.  **Access the Application:**

    Open your web browser and navigate to:
    [http://localhost:8000/prediction/](http://localhost:8000/prediction/)

3.  **Stop the Container:**

    Press `Ctrl+C` in the terminal or run:

    ```bash
    docker-compose down
    ```

## Notes

- The application runs on port `8000` by default.
- The `db.sqlite3` database and `trained_models` directory are mounted from your host machine, so data persists.
- `DEBUG` mode is enabled in `docker-compose.yml`. For production, set `DEBUG=0`.
