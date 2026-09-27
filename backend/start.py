"""Production server launcher that honors platform host and port settings."""

import uvicorn

import config


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        proxy_headers=True,
        forwarded_allow_ips="*",
    )
