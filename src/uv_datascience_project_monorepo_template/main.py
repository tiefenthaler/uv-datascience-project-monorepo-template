import uvicorn

from .app_fastapi_autoencoder import app


def main() -> None:
    """Run the FastAPI application."""
    uvicorn.run(app=app, host="0.0.0.0", port=8000)


# Application entry point
if __name__ == "__main__":
    main()
