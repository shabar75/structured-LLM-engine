import os
from openai import OpenAI
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import json

# Load API key
load_dotenv()

# OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# ------------------------
# 1. Define Schema
# ------------------------

class Room(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Layout(BaseModel):
    rooms: List[Room]

# ------------------------
# 2. LLM Call
# ------------------------

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

def generate_layout(prompt):
    response = client.chat.completions.create(
        model='openai/gpt-5.2',
        messages=[
            {
                "role": "system",
                "content": "You ONLY return valid JSON. No extra text."
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
# 3. Validation + Retry
# ------------------------

def validate_and_fix(prompt, max_retries=3):
    for attempt in range(max_retries):
        output = generate_layout(prompt)

        try:
            data = json.loads(output)
            validated = Layout(**data)
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

# ------------------------
# 4. Run
# ------------------------

if __name__ == "__main__":
    result = validate_and_fix(
        "Create a layout with 2 rooms, each 5x4 units, placed side by side."
    )
    print("✅ Final Output:")
    print(result.model_dump_json(indent=2))