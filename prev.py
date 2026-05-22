import os
import base64
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
# 1. Import the load_dotenv function
from dotenv import load_dotenv 

# 2. Automatically look for a .env file and inject its values into os.environ
load_dotenv() 

app = FastAPI(title="Produce Freshness Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Double check that the key actually successfully loaded
if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError("CRITICAL ERROR: GEMINI_API_KEY could not be loaded from the .env configuration file.")

# The GenAI Client will now naturally pick up the key from os.environ
client = genai.Client()

# Define the data schema your application UI relies on
class FreshnessAnalysis(BaseModel):
    item_name: str = Field(description="The identified name of the fruit or vegetable.")
    status: str = Field(description="Must be exactly one of: 'Raw', 'Ripe', 'Over-ripe', 'Spoiled'.")
    confidence_score: float = Field(description="Confidence score between 0.0 and 1.0.")
    visual_indicators: list[str] = Field(description="List of physical features spotted (e.g., brown spots, mold, green stem).")
    estimated_shelf_life: str = Field(description="Estimated time window left before it goes bad under normal conditions.")

class ImagePayload(BaseModel):
    image_base64: str # Expects data:image/jpeg;base64,...

@app.post("/analyze", response_model=FreshnessAnalysis)
async def analyze_image(payload: ImagePayload):
    try:
        # Clean up data URL prefix if present
        if "," in payload.image_base64:
            header, encoded = payload.image_base64.split(",", 1)
        else:
            encoded = payload.image_base64
            
        image_bytes = base64.b64decode(encoded)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Base64 image data format: {str(e)}")

    try:
        # Pass raw data directly to the official GenAI types component
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )
        
        prompt = "Carefully identify the fruit or vegetable shown and rigorously evaluate its structural ripeness or decay state."

        # Execute call requesting structural response constraint 
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                # Pass the class directly or let the SDK serialize it dynamically
                response_schema=FreshnessAnalysis, 
                temperature=0.2,
            ),
        )

        # Robust check to capture the parsed response
        if response.parsed:
            return response.parsed
        elif response.text:
            # Fallback parsing if the SDK left it as a raw JSON text string
            import json
            parsed_json = json.loads(response.text)
            return FreshnessAnalysis(**parsed_json)
        else:
            raise HTTPException(status_code=500, detail="The model returned an empty response.")

    except Exception as e:
        # This will now surface the exact hidden API error message to your browser screen!
        print(f"CRITICAL API ERROR: {str(e)}") # Prints the trace to your terminal
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)