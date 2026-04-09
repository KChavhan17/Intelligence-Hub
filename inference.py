"""
===================================
MANDATORY
- The inference script must be named `inference.py` and placed in the root directory
STDOUT FORMAT
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
===================================
"""

import asyncio
import os
import textwrap
from typing import List, Optional
from openai import OpenAI
from my_env_v4 import MyEnvV4Action, MyEnvV4Env

# Configuration
IMAGE_NAME = os.getenv("IMAGE_NAME", "scaler-school-of-technology/my_env_v4:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("MY_ENV_V4_TASK", "echo")
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1 

_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = "Reply with exactly one message string — no quotes, no prefixes."

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def get_model_message(client: OpenAI, step: int, last_echoed: str, last_reward: float, history: List[str]) -> str:
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Step {step}. Last echo: {last_echoed}. Reward: {last_reward}"},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return (completion.choices[0].message.content or "hello").strip()
    except Exception:
        return "hello"

async def main() -> None:
    # Notice: No spaces before 'async def main'
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = None
    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    success = False

    try:
        env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)
        log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)
        
        result = await env.reset() 
        last_echoed = result.observation.echoed_message
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            try:
                if result.done:
                    break
                
                message = get_model_message(client, step, last_echoed, last_reward, history)
                result = await env.step(MyEnvV4Action(message=message))
                
                reward = float(result.reward) if result.reward else 0.0
                rewards.append(reward)
                steps_taken = step
                last_echoed = result.observation.echoed_message
                last_reward = reward
                
                log_step(step=step, action=message, reward=reward, done=result.done, error=None)
                
                if result.done:
                    break
            except Exception as e:
                log_step(step=step, action="error", reward=0.00, done=True, error=str(e))
                break

        score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as critical_error:
        print(f"[DEBUG] Critical error: {critical_error}")
    finally:
        if env:
            await env.close()
        log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    asyncio.run(main())

