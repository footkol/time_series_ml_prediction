from fastapi import FastAPI
from app.preprocessing.preprocessing import MyPreProcessor
from app.routes.health import health_router
from app.routes.predict import predict_router
from app.routes.info import info_router

## create the app 
app = FastAPI()

## include the routers
app.include_router(health_router)
app.include_router(info_router)
app.include_router(predict_router)

async def process_data(data: DataModel):
    processor = MyPreProcessor(data)
    
    processed_data = processor.preprocess()
    
    return {"processed_data": processed_data}

