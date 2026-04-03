from dataclasses import dataclass
from typing import Optional

@dataclass
class MyEnvV4Action:
    message: str

@dataclass
class MyEnvV4Observation:
    echoed_message: str

@dataclass
class MyEnvV4Result:
    observation: MyEnvV4Observation
    reward: float
    done: bool

class MyEnvV4Env:
    def __init__(self):
        self.done = False

    @classmethod
    async def from_docker_image(cls, image_name: Optional[str]):
        return cls()

    async def reset(self):
        self.done = False
        obs = MyEnvV4Observation(echoed_message="Environment Reset Successful")
        # Creating a result object that matches what inference.py expects
        return MyEnvV4Result(observation=obs, reward=0.0, done=False)

    async def step(self, action: MyEnvV4Action):
        # This matches the 'echo' logic in the official script
        reward = len(action.message) * 0.1 
        self.done = True 
        obs = MyEnvV4Observation(echoed_message=action.message)
        return MyEnvV4Result(observation=obs, reward=reward, done=self.done)

    async def close(self):
        pass

