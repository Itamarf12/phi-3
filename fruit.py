from ray import serve
from ray.serve.handle import DeploymentHandle
#import torch
#from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
#from google.cloud import storage
from starlette.requests import Request
import logging

ray_serve_logger = logging.getLogger("ray.serve")
BUCKET = 'nonsensitive-data'
REGION = 'us-east-1'
S3_DIRECTORY = 'phi3_finetuned'
MODEL_LOCAL_DIR = '/tmp/phi3'
DEVICE = 'cpu'




@serve.deployment(num_replicas=2)
class RiskyFeatures:
    def __init__(self):
        print("hello")


    async def __call__(self, request: Request) -> float:
        #fruit, amount = await request.json()
        #return await self.check_price(fruit, amount)
        ray_serve_logger.warning("aaaaaaaaaaaaaaa  1111111")
        encoded_key = os.getenv('GCP_CRED')
        return encoded_key

deployment_graph = RiskyFeatures.bind()
