from ray import serve
from ray.serve.handle import DeploymentHandle

# These imports are used only for type hints:
from typing import Dict
from starlette.requests import Request


@serve.deployment(num_replicas=2)
class FruitMarket:
    def __init__(
        self,
        mango_stand: DeploymentHandle,
        orange_stand: DeploymentHandle,
        pear_stand: DeploymentHandle,
    ):
        self.directory = {
            "MANGO": mango_stand.options(use_new_handle_api=True),
            "ORANGE": orange_stand.options(use_new_handle_api=True),
            "PEAR": pear_stand.options(use_new_handle_api=True),
        }

    async def check_price(self, fruit: str, amount: float) -> float:
        if fruit not in self.directory:
            return -1
        else:
            fruit_stand = self.directory[fruit]
            return await fruit_stand.check_price.remote(amount)

    async def __call__(self, request: Request) -> float:
        fruit, amount = await request.json()
        return await self.check_price(fruit, amount)


@serve.deployment(user_config={"price": 3})
class MangoStand:

    DEFAULT_PRICE = 1

    def __init__(self):
        # This default price is overwritten by the one specified in the
        # user_config through the reconfigure() method.
        self.price = self.DEFAULT_PRICE

    def reconfigure(self, config: Dict):
        self.price = config.get("price", self.DEFAULT_PRICE)

    def check_price(self, amount: float) -> float:
        return self.price * amount


@serve.deployment(user_config={"price": 2})
class OrangeStand:

    DEFAULT_PRICE = 0.5

    def __init__(self):
        # This default price is overwritten by the one specified in the
        # user_config through the reconfigure() method.
        self.price = self.DEFAULT_PRICE

    def reconfigure(self, config: Dict):
        self.price = config.get("price", self.DEFAULT_PRICE)

    def check_price(self, amount: float) -> float:
        return self.price * amount


@serve.deployment(user_config={"price": 4})
class PearStand:

    DEFAULT_PRICE = 0.75

    def __init__(self):
        # This default price is overwritten by the one specified in the
        # user_config through the reconfigure() method.
        self.price = self.DEFAULT_PRICE

    def reconfigure(self, config: Dict):
        self.price = config.get("price", self.DEFAULT_PRICE)

    def check_price(self, amount: float) -> float:
        return self.price * amount


mango_stand = MangoStand.bind()
orange_stand = OrangeStand.bind()
pear_stand = PearStand.bind()

deployment_graph = FruitMarket.bind(mango_stand, orange_stand, pear_stand)
