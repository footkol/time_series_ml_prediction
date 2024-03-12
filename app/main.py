from fastapi import FastAPI

from app.routes.health import health_router
from app.routes.predict import predict_router
from app.routes.info import info_router

## create the app 
app = FastAPI()

## include the routers
app.include_router(health_router)
app.include_router(info_router)
app.include_router(predict_router)

