import asyncio
import os
from typing import List
from openai import OpenAI
from my_env_v4 import MyEnvV4Action, MyEnvV4Env

IMAGE_NAME = os.getenv("IMAGE_NAME", "scaler-school-of-technology/my_env_v4:latest")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("MY_ENV_V4_BENCHMARK", "my_env_v4")
MAX_STEPS = 8
MAX_TOKENS = 150

# 3 tasks: easy, medium, hard
TASKS = ["echo-easy", "echo-medium", "echo-hard"]


def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error="null"):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def clamp_score(score: float) -> float:
    # Score must be strictly between 0.0 and 1.0
    return max(0.01, min(score, 0.99))


async def run_task(client, env, task_name):
    rewards: List[float] = []
    steps_taken = 0
    success = False

    try:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        result = await env.reset()
        last_echoed = result.observation.echoed_message
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            try:
                if result.done:
                    break

                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Send a long meaningful message to maximize reward."},
                        {"role": "user", "content": f"Task: {task_name}. Step {step}. Last echo: {last_echoed!r}. Last reward: {last_reward:.2f}. Send your next message."}
                    ],
                    max_tokens=MAX_TOKENS
                )
                message = (completion.choices[0].message.content or "hello").strip()

                result = await env.step(MyEnvV4Action(message=message))
                reward = result.reward or 0.0
                done = result.done
                last_echoed = result.observation.echoed_message
                last_reward = reward

                rewards.append(reward)
                steps_taken = step

                log_step(step=step, action=message[:50], reward=reward, done=done)

                if done:
                    break

            except Exception as step_error:
                print(f"[DEBUG] Step {step} error: {step_error}", flush=True)
                break

        max_possible = MAX_STEPS * MAX_TOKENS * 0.1
        raw_score = sum(rewards) / max_possible if max_possible > 0 else 0.0
        final_score = clamp_score(raw_score)
        success = final_score > 0.1

    except Exception as e:
        print(f"[DEBUG] Task {task_name} error: {e}", flush=True)
        final_score = 0.01
        success = False

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)

    return final_score


async def main():
    await asyncio.sleep(30)

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    env = None

    # Try to start environment 3 times
    for attempt in range(3):
        try:
            env = await MyEnvV4Env.from_docker_image(IMAGE_NAME)
            break
        except Exception as e:
            print(f"[DEBUG] Attempt {attempt+1} failed: {e}", flush=True)
            await asyncio.sleep(10)

    if env is None:
        print("[DEBUG] Environment could not be started", flush=True)
        for task in TASKS:
            log_start(task=task, env=BENCHMARK, model=MODEL_NAME)
            log_end(False, 0, [])
        return

    try:
        for task_name in TASKS:
            await run_task(client, env, task_name)

    finally:
        try:
            await env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
