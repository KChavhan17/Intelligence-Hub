import gradio as gr
from fastapi import FastAPI
from server.environment import NewsIntelligenceEnv 
import uvicorn

# Initialize logic
env = NewsIntelligenceEnv()

def get_intel(section):
    query_map = {
        "Geopolitics & Strategy": "geopolitics OR diplomacy OR international relations",
        "Space Frontier (Artemis/ISRO)": "Artemis Moon Mission OR ISRO Gaganyaan OR SpaceX",
        "Bharat Shakti (Defense/Tech)": "India Defense News OR Digital India OR ISRO",
        "Global Economy": "World Bank OR Global Markets OR Trade War",
        "Future Tech (AI/Quantum)": "Artificial Intelligence OR Quantum Computing breakthroughs"
    }
    raw = env.fetch_live_data(query_map[section])
    output = f"LIVE INTELLIGENCE: {section.upper()}\n" + "="*50 + "\n"
    for item in raw:
        output += f"[{item['title']}] SRC: {item['source']}\n"
    return output

app = FastAPI()

@app.post("/reset")
async def reset():
    return {"status": "success", "message": "Environment reset successfully"}

with gr.Blocks() as demo:
    gr.Markdown("# GLOBAL INTELLIGENCE HUB")
    selection = gr.Dropdown(
        choices=["Geopolitics & Strategy", "Space Frontier (Artemis/ISRO)", "Bharat Shakti (Defense/Tech)", "Global Economy", "Future Tech (AI/Quantum)"], 
        label="Domain"
    )
    btn = gr.Button("SIUUU - FETCH LIVE DATA")
    out = gr.Textbox(label="Graded Intelligence Feed", lines=15)
    btn.click(get_intel, selection, out)

app = gr.mount_gradio_app(app, demo, path="/")

# THIS IS THE PART THAT FIXES THE ERROR
def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()





