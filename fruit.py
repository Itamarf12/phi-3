from ray import serve
from ray.serve.handle import DeploymentHandle
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from google.cloud import storage


# These imports are used only for type hints:
from typing import Dict
from starlette.requests import Request


@serve.deployment(num_replicas=2)
class RiskyFeatures:
    def __init__(self):
        print("hello")


    async def __call__(self, request: Request) -> float:
        #fruit, amount = await request.json()
        #return await self.check_price(fruit, amount)
        return "hello"

deployment_graph = RiskyFeatures.bind()
