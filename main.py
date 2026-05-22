import os
import base64
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
from google import genai
from google.genai import types
from dotenv import load_dotenv
import motor.motor_asyncio
from bson import ObjectId
from datetime import datetime
import hashlib

load_dotenv()

app = FastAPI(title="FoodLens API", version="2.1")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

allowed_origins = [FRONTEND_URL]

if FRONTEND_URL.startswith("http://"):
    allowed_origins.append(FRONTEND_URL.replace("http://", "https://"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── MongoDB ──────────────────────────────────────────────────────────────────
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
client_mongo = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client_mongo["foodlens"]
users_col = db["users"]
scans_col = db["scans"]

# ── Gemini ───────────────────────────────────────────────────────────────────
if not os.environ.get("GEMINI_API_KEY"):
    raise RuntimeError("GEMINI_API_KEY not set.")
gemini = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════════

class UserProfile(BaseModel):
    name: str
    email: str
    password: str
    age: int = 25
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    gender: Optional[str] = None
    allergies: List[str] = []
    medical_conditions: List[str] = []
    dietary_preferences: List[str] = []

class UserLogin(BaseModel):
    email: str
    password: str

class UserUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    gender: Optional[str] = None
    allergies: Optional[List[str]] = None
    medical_conditions: Optional[List[str]] = None
    dietary_preferences: Optional[List[str]] = None

# ── Freshness ─────────────────────────────────────────────────────────────────
class FreshnessAnalysis(BaseModel):
    item_name: str
    status: str
    confidence_score: float
    visual_indicators: List[str]
    estimated_shelf_life: str
    storage_tips: List[str]
    nutritional_highlights: List[str]

class FreshnessPayload(BaseModel):
    user_id: Optional[str] = None
    image_base64: str

# ── Forensic Scan ─────────────────────────────────────────────────────────────
class AuthenticationMetrics(BaseModel):
    barcode: str
    fssai_license: str
    expiry_date: str
    origin_country: str
    manufacture_date: str

class AdditiveRisk(BaseModel):
    chemical_code: str
    risk_rating: str
    clinical_reasoning: str
    regulatory_status: str

class ScorecardMetrics(BaseModel):
    foodlens_score: int
    nutri_score: str
    nova_class: int
    sugar_tsp: float
    sodium_mg: float
    saturated_fat_g: float
    fiber_g: float
    protein_g: float
    calories_per_serving: int
    gut_health_index: str

class AuditedClaim(BaseModel):
    marketing_claim: str
    is_valid: bool
    audit_verdict: str

class UserHealthImpact(BaseModel):
    is_safe_for_profile: bool
    medical_clash_warnings: List[str]
    allergen_warnings: List[str]
    healthier_substitute: str
    avoid_if: List[str]

class ForensicAnalysisResponse(BaseModel):
    identified_product: str
    product_category: str
    authentication: AuthenticationMetrics
    hazardous_additives: List[AdditiveRisk]
    nutrition_profile: ScorecardMetrics
    claims_compliance: List[AuditedClaim]
    personalized_safety: UserHealthImpact
    ingredients_summary: str
    ai_verdict: str

# ── UPDATED: ForensicPayload now accepts multi-image packaging arrays ──────────
class ForensicPayload(BaseModel):
    user_id: Optional[str] = None

    # Core single images (required for analysis)
    ingredient_image: Optional[str] = None
    nutrition_image:  Optional[str] = None
    barcode_image:    Optional[str] = None

    # Primary packaging images (first of each multi-group, kept for compat)
    front_image: Optional[str] = None
    back_image:  Optional[str] = None

    # Additional packaging images from multi-upload
    front_images_extra: List[str] = []
    back_images_extra:  List[str] = []

    # Merged list sent by frontend for convenience (front + back combined)
    all_packaging_images: List[str] = []

    # Legacy single-image fields — still accepted but no longer required
    # (expiry, fssai, claims are now extracted from packaging images by AI)
    expiry_image: Optional[str] = None
    fssai_image:  Optional[str] = None
    claims_image: Optional[str] = None

    # User profile
    user_age: int = 25
    user_allergies: List[str] = []
    user_medical_conditions: List[str] = []
    user_dietary_preferences: List[str] = []

# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def hash_password(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()

def decode_b64(data: str) -> bytes:
    if "," in data:
        _, encoded = data.split(",", 1)
    else:
        encoded = data
    return base64.b64decode(encoded)

def b64_to_part(data: str) -> types.Part:
    return types.Part.from_bytes(data=decode_b64(data), mime_type="image/jpeg")

def serialize_doc(doc) -> dict:
    doc["_id"] = str(doc["_id"])
    return doc

def build_packaging_parts(payload: ForensicPayload) -> tuple[list, list]:
    """
    Deduplicate and convert all packaging images to Gemini Parts.
    Returns (parts_list, context_labels_list).

    Priority order:
      1. all_packaging_images (merged list from frontend)
      2. front_image + front_images_extra
      3. back_image  + back_images_extra
    Then append any explicit legacy images that are not already covered.
    """
    parts: list[types.Part] = []
    labels: list[str] = []

    def add(b64: str, label: str):
        if not b64:
            return
        try:
            parts.append(b64_to_part(b64))
            labels.append(label)
        except Exception:
            pass  # skip corrupt / empty images

    # Use all_packaging_images if the frontend sent it (de-duplication at source)
    if payload.all_packaging_images:
        total = len(payload.all_packaging_images)
        front_count = len([payload.front_image] + payload.front_images_extra) if payload.front_image else len(payload.front_images_extra)
        back_count  = total - front_count
        for i, img in enumerate(payload.all_packaging_images):
            lbl = f"Front packaging image {i+1}" if i < front_count else f"Back packaging image {i - front_count + 1}"
            add(img, lbl)
    else:
        # Fallback: build from individual fields
        add(payload.front_image, "Front packaging")
        for i, img in enumerate(payload.front_images_extra, start=2):
            add(img, f"Front packaging image {i}")
        add(payload.back_image, "Back packaging")
        for i, img in enumerate(payload.back_images_extra, start=2):
            add(img, f"Back packaging image {i}")

    # Always include explicit legacy fields if they exist (old clients / manual uploads)
    add(payload.expiry_image, "Expiry & manufacturing date panel")
    add(payload.fssai_image,  "FSSAI license detail panel")
    add(payload.claims_image, "Marketing claims panel")

    return parts, labels

# ═══════════════════════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/auth/register")
async def register(profile: UserProfile):
    existing = await users_col.find_one({"email": profile.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered.")
    doc = profile.dict()
    doc["password"] = hash_password(profile.password)
    doc["created_at"] = datetime.utcnow().isoformat()
    doc["scan_count"] = 0
    result = await users_col.insert_one(doc)
    user = await users_col.find_one({"_id": result.inserted_id})
    user = serialize_doc(user)
    user.pop("password")
    return {"success": True, "user": user}

@app.post("/auth/login")
async def login(creds: UserLogin):
    user = await users_col.find_one({"email": creds.email})
    if not user or user["password"] != hash_password(creds.password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    user = serialize_doc(user)
    user.pop("password")
    return {"success": True, "user": user}

@app.get("/auth/profile/{user_id}")
async def get_profile(user_id: str):
    try:
        user = await users_col.find_one({"_id": ObjectId(user_id)})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID.")
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    user = serialize_doc(user)
    user.pop("password")
    return user

@app.put("/auth/profile/{user_id}")
async def update_profile(user_id: str, updates: UserUpdate):
    update_data = {k: v for k, v in updates.dict().items() if v is not None}
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update.")
    try:
        await users_col.update_one({"_id": ObjectId(user_id)}, {"$set": update_data})
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid user ID.")
    user = await users_col.find_one({"_id": ObjectId(user_id)})
    user = serialize_doc(user)
    user.pop("password")
    return {"success": True, "user": user}

# ═══════════════════════════════════════════════════════════════════════════════
# SCAN ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/analyze/freshness", response_model=FreshnessAnalysis)
async def analyze_freshness(payload: FreshnessPayload):
    try:
        image_part = b64_to_part(payload.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")

    prompt = """
    You are an expert produce quality analyst. Carefully examine this fruit or vegetable.
    Identify it precisely, assess its ripeness/freshness state rigorously, and provide detailed analysis.
    Include practical storage tips and any nutritional changes at this ripeness stage.
    Be specific and accurate — this affects real food purchasing decisions.
    """
    try:
        response = gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, image_part],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FreshnessAnalysis,
                temperature=0.2,
            ),
        )
        result = response.parsed or FreshnessAnalysis(**json.loads(response.text))

        if payload.user_id:
            await scans_col.insert_one({
                "user_id": payload.user_id, "type": "freshness",
                "result": result.dict(), "created_at": datetime.utcnow().isoformat(),
            })
            await users_col.update_one({"_id": ObjectId(payload.user_id)}, {"$inc": {"scan_count": 1}})

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Error: {e}")


@app.post("/analyze/forensic", response_model=ForensicAnalysisResponse)
async def analyze_forensic(payload: ForensicPayload):
    # ── Resolve user profile ──────────────────────────────────────────────────
    user_age        = payload.user_age
    user_allergies  = payload.user_allergies
    user_conditions = payload.user_medical_conditions
    user_diet       = payload.user_dietary_preferences

    if payload.user_id:
        try:
            user = await users_col.find_one({"_id": ObjectId(payload.user_id)})
            if user:
                user_age        = user.get("age",                user_age)
                user_allergies  = user.get("allergies",          user_allergies)
                user_conditions = user.get("medical_conditions", user_conditions)
                user_diet       = user.get("dietary_preferences",user_diet)
        except Exception:
            pass

    # ── Build Gemini contents list ────────────────────────────────────────────
    contents: list = []
    image_context: list[str] = []

    # 1. Core single images (ingredient list, nutrition table, barcode)
    core_fields = [
        ("ingredient_image", "Ingredient list label"),
        ("nutrition_image",  "Nutrition facts table"),
        ("barcode_image",    "Barcode / QR code"),
    ]
    for field, label in core_fields:
        b64 = getattr(payload, field, None)
        if b64:
            try:
                contents.append(b64_to_part(b64))
                image_context.append(label)
            except Exception:
                pass

    # 2. Packaging images (front + back, multiple angles supported)
    #    AI extracts expiry dates, FSSAI number, marketing claims from these.
    packaging_parts, packaging_labels = build_packaging_parts(payload)
    contents.extend(packaging_parts)
    image_context.extend(packaging_labels)

    if not contents:
        raise HTTPException(status_code=400, detail="At least one image is required.")

    # ── Determine what the AI should auto-extract from packaging ─────────────
    packaging_count = len(packaging_parts)
    has_packaging   = packaging_count > 0

    auto_extract_note = ""
    if has_packaging:
        auto_extract_note = f"""
    IMPORTANT — {packaging_count} packaging image(s) have been provided (front and/or back).
    Carefully scan ALL packaging images for:
      • Expiry date / Best Before / Use By date
      • Manufacturing / Production date
      • FSSAI license number (14-digit number printed on pack, often near the manufacturer address)
      • Any certification logos (FSSAI, Agmark, Organic India, ISO, etc.)
      • All marketing claims printed on the pack (e.g. "0% Trans Fat", "No Added Sugar",
        "High Protein", "Gluten Free", "Natural", "Organic", "Made with real fruit", etc.)
      • Country / state of manufacture and manufacturer address
    Extract these even if the text is small, upside-down, or on a side panel.
    Use "Not visible" only if genuinely absent from all provided images.
    """
    else:
        auto_extract_note = """
    No packaging images were provided. Set expiry_date, manufacture_date, fssai_license,
    and origin_country to "Not provided" unless inferable from other images.
    """

    prompt = f"""
    You are a forensic food scientist and regulatory compliance expert specialising in
    Indian food safety (FSSAI standards).

    Images provided for analysis: {', '.join(image_context)}.

    {auto_extract_note}

    Extract ALL visible text, ingredient lists, nutrition data, barcodes, and any other
    information from the images.

    Cross-reference against:
    - FSSAI regulatory database
    - Indian Dietary Guidelines 2024
    - WHO additives risk classifications
    - Common harmful ingredient combinations (e.g. Sodium Benzoate + Vitamin C → benzene)

    User profile for personalised safety analysis:
    - Age: {user_age}
    - Allergies: {user_allergies if user_allergies else 'None declared'}
    - Medical conditions: {user_conditions if user_conditions else 'None declared'}
    - Dietary preferences: {user_diet if user_diet else 'None declared'}

    Be extremely thorough — flag every suspicious additive, verify every marketing claim
    found on the packaging, and give a clear personalized verdict.
    For ai_verdict, write one crisp sentence summarizing overall safety and quality.
    """

    contents.insert(0, prompt)

    # ── Call Gemini ───────────────────────────────────────────────────────────
    try:
        response = gemini.models.generate_content(
            model="gemini-2.5-flash",
            contents=contents,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ForensicAnalysisResponse,
                temperature=0.15,
            ),
        )
        result = response.parsed or ForensicAnalysisResponse(**json.loads(response.text))

        if payload.user_id:
            await scans_col.insert_one({
                "user_id": payload.user_id, "type": "forensic",
                "result": result.dict(),
                "image_count": len(contents) - 1,  # exclude prompt string
                "packaging_images_count": packaging_count,
                "created_at": datetime.utcnow().isoformat(),
            })
            await users_col.update_one(
                {"_id": ObjectId(payload.user_id)}, {"$inc": {"scan_count": 1}}
            )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Forensic Error: {e}")


@app.get("/scans/{user_id}")
async def get_scan_history(user_id: str, limit: int = 20):
    try:
        cursor = scans_col.find({"user_id": user_id}).sort("created_at", -1).limit(limit)
        scans = []
        async for doc in cursor:
            scans.append(serialize_doc(doc))
        return {"scans": scans}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)