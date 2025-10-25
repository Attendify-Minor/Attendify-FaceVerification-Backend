# main.py - Attendify Face Recognition API with REST + WebSocket
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
from typing import Optional, List
import base64, hashlib, os, tempfile, logging, json
from io import BytesIO
from PIL import Image
import numpy as np
from deepface import DeepFace
import cv2

# ==============================
# App + Config
# ==============================
app = FastAPI(title="Attendify Face Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGODB_URL = (
    "mongodb+srv://Krishna:Krish%40atlas25@attendify-cluster."
    "fh96zp0.mongodb.net/attendifyDB?retryWrites=true&w=majority"
)
client = AsyncIOMotorClient(MONGODB_URL)
db = client.attendifyDB

users_collection = db.users
face_encodings_collection = db.face_encodings
verification_logs_collection = db.verification_logs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Attendify")

FACE_MODEL = "VGG-Face"
DETECTOR_BACKEND = "retinaface"  # better than opencv
DISTANCE_METRIC = "cosine"
VERIFICATION_THRESHOLD = 0.4
DEBUG_SAVE_FRAMES = False  # set True if you want debug images

# ==============================
# Models
# ==============================
class RegistrationRequest(BaseModel):
    fullName: str
    email: EmailStr
    employeeId: str
    department: str
    role: str
    faceImage: str
    timestamp: datetime

class VerificationRequest(BaseModel):
    userId: str
    faceImage: str
    timestamp: datetime
    latitude: Optional[float] = None
    longitude: Optional[float] = None

class VerificationResponse(BaseModel):
    success: bool
    confidence: float
    message: str
    userId: str
    timestamp: datetime
    matchedUser: Optional[dict] = None

# ==============================
# Utilities
# ==============================
def decode_base64_image(base64_string: str) -> np.ndarray:
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        image_data = base64.b64decode(base64_string)
        image = Image.open(BytesIO(image_data))
        if image.mode != "RGB":
            image = image.convert("RGB")
        arr = np.array(image)
        arr = cv2.resize(arr, (640, 480))  # normalize size
        return arr
    except Exception as e:
        logger.error(f"Image decode error: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid image format")

def save_temp_image(image_array: np.ndarray) -> str:
    temp_path = os.path.join(
        tempfile.gettempdir(), f"face_{datetime.now().timestamp()}.jpg"
    )
    image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(temp_path, image_bgr)
    return temp_path

def cleanup_temp_image(file_path: str):
    if os.path.exists(file_path):
        os.remove(file_path)

def extract_face_embedding(image_array: np.ndarray) -> Optional[List[float]]:
    temp_path = None
    try:
        if DEBUG_SAVE_FRAMES:
            os.makedirs("debug_frames", exist_ok=True)
            debug_path = os.path.join(
                "debug_frames", f"frame_{datetime.now().timestamp()}.jpg"
            )
            cv2.imwrite(debug_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
            logger.info(f"Debug frame saved: {debug_path}")

        temp_path = save_temp_image(image_array)
        embedding_objs = DeepFace.represent(
            img_path=temp_path,
            model_name=FACE_MODEL,
            enforce_detection=True,
            detector_backend=DETECTOR_BACKEND,
        )
        if not embedding_objs:
            return None
        return embedding_objs[0]["embedding"]
    except Exception as e:
        logger.warning(f"Face extraction error: {str(e)}")
        return None
    finally:
        if temp_path:
            cleanup_temp_image(temp_path)

def calculate_distance(emb1: List[float], emb2: List[float]) -> float:
    e1 = np.array(emb1)
    e2 = np.array(emb2)
    if DISTANCE_METRIC == "cosine":
        return float(1 - np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    return float(np.linalg.norm(e1 - e2))

def generate_user_id(employee_id: str, email: str) -> str:
    unique = f"{employee_id}_{email}_{datetime.utcnow().isoformat()}"
    return hashlib.sha256(unique.encode()).hexdigest()[:16]

# ==============================
# Startup / Shutdown
# ==============================
@app.on_event("startup")
async def startup():
    await users_collection.create_index("employeeId", unique=True)
    await users_collection.create_index("email", unique=True)
    await face_encodings_collection.create_index("userId")
    await verification_logs_collection.create_index("userId")
    logger.info("✅ Database ready")
    logger.info(f"✅ Model: {FACE_MODEL}, Detector: {DETECTOR_BACKEND}")

@app.on_event("shutdown")
async def shutdown():
    client.close()

# ==============================
# REST Endpoints
# ==============================
@app.get("/")
async def root():
    return {
        "status": "active",
        "service": "Attendify Face Recognition API",
        "version": "2.0.0",
        "model": FACE_MODEL,
        "detector": DETECTOR_BACKEND,
    }

@app.post("/api/register", status_code=201)
async def register_user(request: RegistrationRequest):
    return {"message": "Use WebSocket for registration in v2.0"}

@app.post("/api/verify", response_model=VerificationResponse)
async def verify_user(request: VerificationRequest):
    return {"message": "Use WebSocket for verification in v2.0"}

# ==============================
# WebSocket Endpoint
# ==============================
@app.websocket("/ws/face")
async def websocket_face(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            action = data.get("action")

            # ---------------- REGISTER ----------------
            if action == "register":
                try:
                    fullName = data.get("fullName")
                    email = data.get("email")
                    employeeId = data.get("employeeId")
                    department = data.get("department")
                    role = data.get("role")
                    face_b64 = data.get("faceImage")

                    if not all([fullName, email, employeeId, department, role, face_b64]):
                        await websocket.send_json({"success": False, "message": "Missing fields"})
                        continue

                    existing = await users_collection.find_one(
                        {"$or": [{"employeeId": employeeId}, {"email": email}]}
                    )
                    if existing:
                        await websocket.send_json({"success": False, "message": "User already exists"})
                        continue

                    image_array = decode_base64_image(face_b64)
                    embedding = extract_face_embedding(image_array)
                    if not embedding:
                        await websocket.send_json({"success": False, "message": "No face detected"})
                        continue

                    user_id = generate_user_id(employeeId, email)
                    await users_collection.insert_one({
                        "userId": user_id,
                        "fullName": fullName,
                        "email": email,
                        "employeeId": employeeId,
                        "department": department,
                        "role": role,
                        "registeredAt": datetime.utcnow(),
                        "isActive": True,
                        "lastVerified": None,
                    })
                    await face_encodings_collection.insert_one({
                        "userId": user_id,
                        "embedding": embedding,
                        "model": FACE_MODEL,
                        "createdAt": datetime.utcnow(),
                        "version": 1,
                    })

                    await websocket.send_json({
                        "success": True,
                        "action": "register",
                        "message": "Registration successful ✅",
                        "userId": user_id,
                        "employeeId": employeeId,
                    })
                except Exception as e:
                    await websocket.send_json(
                        {"success": False, "action": "register", "message": str(e)}
                    )

            # ---------------- VERIFY ----------------
            elif action == "verify":
                try:
                    user_id = data.get("userId")
                    face_b64 = data.get("faceImage")
                    latitude = data.get("latitude")
                    longitude = data.get("longitude")

                    if not user_id or not face_b64:
                        await websocket.send_json({"success": False, "message": "Missing fields"})
                        continue

                    user = await users_collection.find_one(
                        {"$or": [{"userId": user_id}, {"employeeId": user_id}]}
                    )
                    if not user:
                        await websocket.send_json({"success": False, "message": "User not found"})
                        continue

                    stored = await face_encodings_collection.find_one({"userId": user["userId"]})
                    if not stored:
                        await websocket.send_json({"success": False, "message": "No encoding"})
                        continue

                    image_array = decode_base64_image(face_b64)
                    current_embedding = extract_face_embedding(image_array)
                    if not current_embedding:
                        await websocket.send_json({"success": False, "message": "No face detected"})
                        continue

                    distance = calculate_distance(stored["embedding"], current_embedding)
                    confidence = max(0, 1 - distance)
                    is_verified = distance <= VERIFICATION_THRESHOLD

                    await verification_logs_collection.insert_one({
                        "userId": user["userId"],
                        "timestamp": datetime.utcnow(),
                        "success": is_verified,
                        "confidence": float(confidence),
                        "distance": float(distance),
                        "latitude": latitude,
                        "longitude": longitude,
                    })

                    if is_verified:
                        await users_collection.update_one(
                            {"userId": user["userId"]},
                            {"$set": {"lastVerified": datetime.utcnow()}},
                        )

                    await websocket.send_json({
                        "success": is_verified,
                        "action": "verify",
                        "confidence": round(confidence * 100, 2),
                        "message": "Verified ✅" if is_verified else "Verification failed ❌",
                        "userId": user["userId"],
                        "employeeId": user["employeeId"],
                    })
                except Exception as e:
                    await websocket.send_json(
                        {"success": False, "action": "verify", "message": str(e)}
                    )

            else:
                await websocket.send_json({"success": False, "message": "Unknown action"})
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")

# ==============================
# Entry Point
# ==============================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
