import io
import os
import asyncio
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_preprocess
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from typing import List
import pandas as pd
import hashlib

# ─── KONSTANTA TPS ───
KEC_COORDS = {
    "ANDIR": (-6.9178, 107.5867), "ANTAPANI": (-6.9127, 107.6645),
    "ARCAMANIK": (-6.9000, 107.6800), "ASTANAANYAR": (-6.9400, 107.5967),
    "BABAKAN CIPARAY": (-6.9450, 107.5800), "BANDUNG KIDUL": (-6.9570, 107.6400),
    "BANDUNG KULON": (-6.9350, 107.5700), "BATUNUNGGAL": (-6.9250, 107.6320),
    "BOJONGLOA KIDUL": (-6.9510, 107.5900), "BOJONGLOA KALER": (-6.9380, 107.5830),
    "BUAHBATU": (-6.9550, 107.6530), "CENANG": (-6.9320, 107.6950),
    "CIBEUNYING KIDUL": (-6.9022, 107.6356), "CIBEUNYING KALER": (-6.8950, 107.6300),
    "CIBIRU": (-6.9065, 107.7009), "CICENDO": (-6.9050, 107.5900),
    "CIDADAP": (-6.8745, 107.5970), "CINAMBO": (-6.9280, 107.7050),
    "COBLONG": (-6.8950, 107.6100), "GEDEBAGE": (-6.9650, 107.7100),
    "KIARACONDONG": (-6.9280, 107.6516), "LENGKONG": (-6.9300, 107.6260),
    "MANDALAJATI": (-6.8930, 107.6900), "PANYILEUKAN": (-6.9560, 107.6950),
    "RANCASARI": (-6.9550, 107.6780), "REGOL": (-6.9400, 107.6080),
    "SUKAJADI": (-6.8880, 107.5960), "SUKASARI": (-6.8854, 107.5934),
    "SUMUR BANDUNG": (-6.9164, 107.6133), "UJUNGBERUNG": (-6.9000, 107.7056),
}
KOTA_CENTER = (-6.9175, 107.6191)
CAPACITY_TON = {"URBAN": 20.0, "SEMI_URBAN": 10.0, "RURAL": 4.0}
AREA_TYPE_FROM_CSV = {"metropolitan": "URBAN", "semi urban": "SEMI_URBAN", "pedesaan": "RURAL"}

TPS_DATA = {}
CSV_VOL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sampahbandung_normal_monthly.csv")


app = FastAPI(title="EcoSort AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_sampah_Advanced.keras")
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "volume_attention_lstm.keras")
SCALER_MEAN_PATH = os.path.join(BASE_DIR, "scaler_mean.npy")
SCALER_SCALE_PATH = os.path.join(BASE_DIR, "scaler_scale.npy")

IMG_SIZE = (224, 224)
CLASS_NAMES = ["Anorganik", "B3", "Organik"]
WINDOW_SIZE = 12

# Tips fallback jika Gemini tidak tersedia
STATIC_TIPS = {
    "Anorganik": (
        "• Pisahkan dari sampah organik dan B3 sebelum dibuang.\n"
        "• Jangan dibakar — asap plastik/kaca mengandung zat beracun.\n"
        "• Bawa ke bank sampah: botol plastik, kaleng, dan kardus punya nilai jual.\n"
        "• Cuci bersih kemasan sebelum disetor agar mudah didaur ulang."
    ),
    "B3": (
        "• JANGAN buang ke tempat sampah biasa — B3 mencemari tanah & air tanah.\n"
        "• Kumpulkan di drop-box B3 (apotek, minimarket, atau dinas lingkungan hidup).\n"
        "• Baterai, lampu, elektronik rusak, dan cat termasuk kategori B3.\n"
        "• Penanganan salah dapat menyebabkan kebakaran atau keracunan logam berat."
    ),
    "Organik": (
        "• Bisa dijadikan kompos dalam 4–8 minggu dengan metode sederhana di rumah.\n"
        "• Jangan campur dengan plastik — mempersulit pengomposan.\n"
        "• Sisa sayur, buah, dan makanan basi sangat cocok untuk eco-enzyme.\n"
        "• Kompos yang dihasilkan bisa dipakai langsung untuk pupuk tanaman."
    ),
}

# Custom Layers
class ChannelAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        ch = input_shape[-1]
        self.gap = layers.GlobalAveragePooling2D()
        self.gmp = layers.GlobalMaxPooling2D()
        self.fc1 = layers.Dense(max(1, ch // self.reduction_ratio), activation="relu", use_bias=False)
        self.fc2 = layers.Dense(ch, activation="sigmoid", use_bias=False)
        super().build(input_shape)

    def call(self, x):
        avg_w = self.fc2(self.fc1(self.gap(x)))
        max_w = self.fc2(self.fc1(self.gmp(x)))
        ch = tf.shape(x)[-1]
        scale = tf.reshape(avg_w + max_w, [-1, 1, 1, ch])
        return x * scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"reduction_ratio": self.reduction_ratio})
        return cfg

@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.W_q = self.add_weight(name='W_q', shape=(feat_dim, self.units))
        self.W_k = self.add_weight(name='W_k', shape=(feat_dim, self.units))
        self.W_v = self.add_weight(name='W_v', shape=(feat_dim, feat_dim))
        self.v = self.add_weight(name='v', shape=(self.units, 1))
        self.b = self.add_weight(name='b', shape=(self.units,), initializer='zeros')
        super().build(input_shape)

    def call(self, x):
        query = tf.tensordot(x, self.W_q, axes=[[2], [0]])
        key = tf.tensordot(x, self.W_k, axes=[[2], [0]])
        score = tf.tensordot(tf.tanh(query + key + self.b), self.v, axes=[[2], [0]])
        alpha = tf.nn.softmax(score, axis=1)
        value = tf.tensordot(x, self.W_v, axes=[[2], [0]])
        ctx = tf.reduce_sum(alpha * value, axis=1)
        return ctx

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# Load Model
print(f"[+] Loading model klasifikasi dari: {MODEL_PATH}")

_original_dense_from_config = tf.keras.layers.Dense.from_config.__func__

@classmethod
def _patched_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return _original_dense_from_config(cls, config)

tf.keras.layers.Dense.from_config = _patched_dense_from_config

model_klasifikasi = None
try:
    model_klasifikasi = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"ChannelAttentionLayer": ChannelAttentionLayer},
    )
    print("[+] Model klasifikasi loaded!")
except Exception as e:
    print(f"[!] Error loading model klasifikasi: {e}")

model_volume = None
scaler_mean = None
scaler_scale = None
try:
    model_volume = tf.keras.models.load_model(
        LSTM_MODEL_PATH,
        custom_objects={"BahdanauAttention": BahdanauAttention},
        compile=False,
    )
    tf.keras.layers.Dense.from_config = _original_dense_from_config
    
    scaler_mean = np.load(SCALER_MEAN_PATH)[:6]
    scaler_scale = np.load(SCALER_SCALE_PATH)[:6]
    print("[+] Model volume loaded!")
except Exception as e:
    print(f"[!] Error loading model volume: {e}")

# Gemini Setup
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
gemini_client = None
tips_cache: dict[str, str] = {}

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("[+] Gemini siap!")
else:
    print("[!] GEMINI_API_KEY tidak ditemukan.")

# Schemas
class TipsReq(BaseModel):
    kategori: str

class AskReq(BaseModel):
    pertanyaan: str
    kategori: str = ""

class VolumeTimestep(BaseModel):
    volume_ton: float
    sin_month: float
    cos_month: float
    area_encoded: float
    year_norm: float
    volume_ma: float
    volume_ema: float

class PredictVolumeReq(BaseModel):
    tps_id: str
    history: List[VolumeTimestep] = []

# ─── FUNGSI LOAD TPS ───
def load_tps():
    global TPS_DATA
    if not os.path.exists(CSV_VOL_PATH):
        print(f"[TPS] Peringatan: File CSV tidak ditemukan di {CSV_VOL_PATH}")
        return
        
    df = pd.read_csv(CSV_VOL_PATH)
    tps_unique = df[['tps_id', 'kecamatan', 'area_type']].drop_duplicates('tps_id').reset_index(drop=True)

    for _, row in tps_unique.iterrows():
        tps_id = row['tps_id']
        kec = row['kecamatan']
        app_type = AREA_TYPE_FROM_CSV.get(row['area_type'], 'RURAL')

        lat0, lon0 = KEC_COORDS.get(kec.upper(), KOTA_CENTER)
        h = int(hashlib.md5(tps_id.encode()).hexdigest()[:8], 16)
        lat = round(lat0 + ((h & 0xFF) / 255.0 - 0.5) * 0.018, 6)
        lon = round(lon0 + ((h >> 8 & 0xFF) / 255.0 - 0.5) * 0.018, 6)

        TPS_DATA[tps_id] = {
            "id": tps_id,
            "kecamatan": kec,
            "alamat": f"{tps_id}, Kec. {kec}, Kota Bandung",
            "area_type": app_type,
            "lat": lat,
            "lon": lon,
            "kapasitas_ton": CAPACITY_TON[app_type],
        }
    print(f"[+] Berhasil load {len(TPS_DATA)} TPS dari CSV.")

@app.on_event("startup")
async def startup_event():
    load_tps()

# Routes
@app.get("/")
def home():
    return {
        "status": "online",
        "models": {
            "klasifikasi": "model_sampah_Advanced.keras",
            "volume": "volume_attention_lstm.keras" if model_volume else "NOT LOADED",
        },
        "classes": CLASS_NAMES,
        "gemini": "ready" if gemini_client else "unavailable",
    }

@app.get("/api/tps")
def get_all_tps():
    """Mengembalikan list semua TPS dengan koordinat dan kapasitas."""
    return list(TPS_DATA.values())

@app.post("/predict-volume/")
async def predict_volume(req: PredictVolumeReq):
    if model_volume is None or scaler_mean is None or scaler_scale is None:
        return {"status": "error", "message": "Model volume belum ter-load."}

    if req.tps_id not in TPS_DATA:
        return {"status": "error", "message": f"TPS {req.tps_id} tidak ditemukan."}

    tps_info = TPS_DATA[req.tps_id]

    if not req.history or len(req.history) != WINDOW_SIZE:
        return {
            "status": "error", 
            "message": f"history harus {WINDOW_SIZE} timestep."
        }

    try:
        raw_rows = [
            [
                ts.volume_ton, ts.sin_month, ts.cos_month,
                ts.area_encoded, ts.year_norm, ts.volume_ma
            ]
            for ts in req.history
        ]
        X_raw = np.array(raw_rows, dtype=np.float32)
        X_scaled = (X_raw - scaler_mean) / scaler_scale
        X_batch = np.expand_dims(X_scaled, axis=0).astype(np.float32)

        def _run_lstm_inference(batch):
            return model_volume(batch, training=False).numpy()[0]

        pred_scaled = await asyncio.to_thread(_run_lstm_inference, X_batch)
        pred_volume_ton = (pred_scaled * float(scaler_scale[0])) + float(scaler_mean[0])
        pred_volume_ton = np.maximum(pred_volume_ton, 0.0)

        predictions = [
            {"bulan_ke": i + 1, "volume_ton": round(float(v), 2)}
            for i, v in enumerate(pred_volume_ton)
        ]
        
        return {
            "status": "success", 
            "tps": tps_info,
            "predictions": predictions
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model_klasifikasi is None:
        return {"error": "Model klasifikasi belum ter-load."}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize(IMG_SIZE)
        img_array = np.array(image, dtype=np.float32)
        img_array = eff_preprocess(img_array)
        img_tensor = tf.expand_dims(tf.convert_to_tensor(img_array), axis=0)

        cls_preds = model_klasifikasi.predict(img_tensor, verbose=0)[0]
        class_idx = int(np.argmax(cls_preds))
        confidence = float(cls_preds[class_idx])
        kategori = CLASS_NAMES[class_idx]

        return {
            "status": "success",
            "hasil": [{"kategori": kategori, "confidence": confidence}],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}

@app.post("/genai/tips/")
async def genai_tips(req: TipsReq):
    if gemini_client is None:
        return {"status": "error", "message": "Gemini tidak aktif."}

    k = req.kategori
    if k in tips_cache:
        return {"status": "success", "tips": tips_cache[k], "cached": True}

    prompt = (
        f"Kamu adalah asisten edukasi lingkungan. Sampah yang terdeteksi: '{k}'. "
        f"Berikan 3-4 poin singkat dalam Bahasa Indonesia yang informatif: "
        f"(1) cara memilah yang benar, (2) dampak lingkungan jika salah buang, "
        f"(3) nilai daur ulang atau manfaat ekonomi. "
        f"Gunakan format bullet • dan bahasa yang santai tapi edukatif."
    )
    try:
        resp = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-2.0-flash-lite",
            contents=prompt,
        )
        tips_cache[k] = resp.text
        return {"status": "success", "tips": resp.text, "cached": False, "source": "gemini"}
    except Exception as e:
        fallback = STATIC_TIPS.get(k, "Pilah sampah sesuai kategorinya sebelum dibuang.")
        return {"status": "success", "tips": fallback, "cached": False, "source": "static"}

@app.post("/genai/ask/")
async def genai_ask(req: AskReq):
    if gemini_client is None:
        return {"status": "error", "message": "Gemini tidak aktif."}

    ctx = f"Konteks: sampah terakhir terdeteksi adalah '{req.kategori}'. " if req.kategori else ""
    prompt = (
        f"Kamu adalah asisten edukasi pengelolaan sampah yang ramah dan informatif. "
        f"{ctx}"
        f"Jawab pertanyaan berikut dalam Bahasa Indonesia secara ringkas (maksimal 4 kalimat): "
        f"{req.pertanyaan}"
    )
    try:
        resp = await asyncio.to_thread(
            gemini_client.models.generate_content,
            model="gemini-2.0-flash-lite",
            contents=prompt,
        )
        return {"status": "success", "jawaban": resp.text}
    except Exception as e:
        tips = STATIC_TIPS.get(req.kategori, "")
        jawaban = (
            f"Maaf, AI sedang tidak tersedia. "
            + (f"Berikut informasi dasar tentang sampah {req.kategori}:\n\n{tips}" if tips
               else "Silakan pilah sampah sesuai kategorinya: Organik, Anorganik, atau B3.")
        )
        return {"status": "success", "jawaban": jawaban}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
