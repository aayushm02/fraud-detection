"""FastAPI application factory."""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from fraud_detection.api.routes import router, app_context
from fraud_detection.explainability.shap_explainer import FraudExplainer

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan event handler for FastAPI."""
    try:
        best = app_context.registry.get_best(metric="f1")
        if best:
            logger.info(f"Loading best model: {best['name']} version {best['version']}")
            app_context.active_model = best['model']
            app_context.model_info = best
            app_context.explainer = FraudExplainer(best['model'])
        else:
            logger.warning("No models found in registry on startup.")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
    
    yield
    
    logger.info("Shutting down FastAPI application.")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Fraud Detection API",
        description="REST API for real-time fraud detection predictions.",
        version="1.0.0",
        lifespan=lifespan
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Global error handler caught: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error."}
        )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        logger.info(f"Request: {request.method} {request.url}")
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response

    app.include_router(router, prefix="/api/v1")

    return app

app = create_app()
