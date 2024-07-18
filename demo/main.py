from config import config


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host=config.host,
        port=config.port,
        reload=config.reload,
        ssl_certfile=config.ssl_certfile,
        ssl_keyfile=config.ssl_keyfile,
    )
