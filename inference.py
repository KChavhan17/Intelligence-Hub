import os
import requests
from typing import List
from openai import OpenAI

# Configuration
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_URL = os.getenv("ENV_URL", "https://keertiwebstack-intelligence-hub.hf.space")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

MAX_STEPS = 8
SUCCESS_THRESHOLD = 0.1


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str = "null") -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def main():
    rewards = []
    steps_taken = 0
    success = False
    task_id = "echo"

    try:
        log_start(task=task_id, env="my_env_v4", model=MODEL_NAME)

        # 1. Reset environment
        reset_resp = requests.post(
            ENV_URL.rstrip('/') + "/reset",
            json={"task_id": task_id},
            timeout=10
        )
        reset_data = reset_resp.json()
        last_echoed = reset_data.get("observation", {}).get("echoed_message", "")
        last_reward = 0.0

        # 2. Step loop
        for step in range(1, MAX_STEPS + 1):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "Send a long, meaningful message to maximize reward."},
                        {"role": "user", "content": f"Step {step}. Last echo: {last_echoed!r}. Last reward: {last_reward:.2f}. Send your next message."}
                    ],
                    max_tokens=150
                )
                action_taken = (response.choices[0].message.content or "hello").strip()

                step_resp = requests.post(
                    ENV_URL.rstrip('/') + "/step",
                    json={"action": action_taken},
                    timeout=10
                )
                step_data = step_resp.json()
                current_reward = step_data.get("reward", 0.0)
                done = step_data.get("done", False)
                error = step_data.get("error") or "null"

                rewards.append(current_reward)
                steps_taken = step
                last_echoed = step_data.get("observation", {}).get("echoed_message", action_taken)
                last_reward = current_reward

                log_step(step=step, action=action_taken[:50], reward=current_reward, done=done, error=error)

                if done:
                    break

            except Exception as step_e:
                log_step(step=step, action="error", reward=0.0, done=True, error=str(step_e)[:80])
                break

        # 3. Calculate success
        total = sum(rewards)
        max_possible = MAX_STEPS * 150 * 0.1
        success = (total / max_possible) >= SUCCESS_THRESHOLD if max_possible > 0 else False

    except Exception as e:
        print(f"[DEBUG] Critical error: {e}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
