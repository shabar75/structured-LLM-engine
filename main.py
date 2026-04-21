import os
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import json
import matplotlib.pyplot as plt
import random

# Load API key
load_dotenv()

# OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define Schema

class Room(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Layout(BaseModel):
    rooms: List[Room]



def rooms_overlap(r1, r2):
    # If one is completely left of the other
    if r1.x + r1.width <= r2.x:
        return False

    # If one is completely right of the other
    if r2.x + r2.width <= r1.x:
        return False

    # If one is completely above the other
    if r1.y + r1.height <= r2.y:
        return False

    # If one is completely below the other
    if r2.y + r2.height <= r1.y:
        return False

    return True


def validate_no_overlap(layout):
    rooms = layout.rooms

    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            if rooms_overlap(rooms[i], rooms[j]):
                raise ValueError(f"Rooms {i} and {j} overlap!")
            
# LLM Call

def generate_layout(prompt):
    response = client.chat.completions.create(
        model='openai/gpt-5.2',
        messages=[
            {
                "role": "system",
                "content": """You are a strict JSON generator.
                Rules:
                - Output ONLY valid JSON (no text, no explanation)
                - Follow this schema exactly:
                {
                "rooms": [
                {
                "x": number,
                "y": number,
                "width": number,
                "height": number
                }
                ]
                }
                Constraints:
                - Rooms MUST NOT overlap
                - Use only numeric values (no strings)
                """
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0,
        max_tokens=300
    )

    return response.choices[0].message.content

# ------------------------
# Validation + Retry
# ------------------------

def validate_and_fix(prompt, max_retries=1):
    for attempt in range(max_retries):
        output = generate_layout(prompt)

        try:
            data = json.loads(output)
            validated = Layout(**data)
            validate_no_overlap(validated)
            return validated

        except Exception as e:
            print(f"❌ Validation failed: {e}")

            prompt = f"""
Fix this JSON strictly.

Schema:
rooms: list of objects with x, y, width, height (floats)

Previous output:
{output}

Return ONLY corrected JSON.
"""

    raise Exception("Failed after retries")




def draw_layout(layout):
    fig, ax = plt.subplots()

    max_x = 0
    max_y = 0

    for i, room in enumerate(layout.rooms):
        # Random color for each room
        color = (random.random(), random.random(), random.random())

        rect = plt.Rectangle(
            (room.x, room.y),
            room.width,
            room.height,
            fill=True,
            color=color,
            alpha=0.5,
            edgecolor='black'
        )

        ax.add_patch(rect)

        # Label
        ax.text(
            room.x + room.width / 2,
            room.y + room.height / 2,
            f"Room {i}",
            ha='center',
            va='center',
            fontsize=10,
            color='black'
        )

        # Track max bounds
        max_x = max(max_x, room.x + room.width)
        max_y = max(max_y, room.y + room.height)

    # Dynamic scaling
    ax.set_xlim(0, max_x + 2)
    ax.set_ylim(0, max_y + 2)

    ax.set_aspect('equal')
    plt.title("AI Generated Layout")
    plt.grid(True)

    plt.show()
#  Run
if __name__ == "__main__":
    result = validate_and_fix(
        """Create 2 rooms with:
        width =5
        height =5
        aligned in a strieght horizontal row
        equal spacing between rooms 
        no overlap"""
    )
    print("✅ Final Output:")
    print(result.model_dump_json(indent=2))
    with open("layout.json", "w") as f:
        f.write(result.model_dump_json(indent =2))
    
    draw_layout(result)