import os
import requests
import textwrap
from typing import List
from openai import OpenAI

# Configuration from Environment Variables
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
# YOUR HF SPACE URL (e.g., https://user-name-space-name.hf.space)
ENV_URL = os.getenv("ENV_URL", "https://keertiwebstack-intelligence-hub.hf.space") 

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = "null") -> None:
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error}", flush=True)

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)

def main():
    # List of tasks to iterate through
    tasks = ["echo-task-1", "echo-task-2", "echo-task-3"]
    
    for task_id in tasks:
        rewards = []
        steps_taken = 0
        success = False
        
        try:
            log_start(task=task_id, env="my_env_v4", model=MODEL_NAME)
            
            # 1. RESET via API (No Docker!)
            reset_resp = requests.post(ENV_URL.rstrip('/') + "/reset", json={"task_id": task_id}, timeout=10)
reset_data = reset_resp.json()
last_echoed = reset_data.get("observation", {}).get("echoed_message", "")
            
            
            # 2. INFERENCE
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": f"Complete task: {task_id}"}],
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
done = step_data.get("done", True)
error = step_data.get("error") or "null"

rewards.append(current_reward)
steps_taken = 1
success = current_reward > 0

log_step(step=1, action=action_taken[:50], reward=current_reward, done=done, error=error)
            
        
            
        except Exception as e:
            log_step(step=1, action="error", reward=0.0, done=True, error=str(e))
        
        finally:
            log_end(success=success, steps=steps_taken, rewards=rewards)

if __name__ == "__main__":
    main()

