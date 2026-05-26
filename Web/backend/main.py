# import io
# import os
# import asyncio
# import numpy as np
# from PIL import Image
# import tensorflow as tf
# from tensorflow.keras import layers
# # [!] PENTING: Gunakan preprocessing KHUSUS MobileNetV2
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# from fastapi import FastAPI, File, UploadFile
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from google import genai
# 
# app = FastAPI(title="EcoSort AI Backend")
# 
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# 
# # ================= CONFIG =================
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 
# # [!] PASTIKAN NAMA FILE MODEL BARUNYA BENAR
# MODEL_PATH = os.path.join(BASE_DIR, "model_sampah_MobileNetV2.keras")
# 
# # [!] UBAH KE 150 (Standar MobileNetV2 dari tim AI)
# IMG_SIZE   = (150, 150) 
# 
# # 3 kelas sesuai flow_from_directory (alphabetical)
# CLASS_NAMES = ["Anorganik", "B3", "Organik"]
# 
# # Tips statis sebagai fallback jika Gemini tidak tersedia
# STATIC_TIPS = {
#     "Anorganik": (
#         "• Pisahkan dari sampah organik dan B3 sebelum dibuang.\n"
#         "• Jangan dibakar — asap plastik/kaca mengandung zat beracun.\n"
#         "• Bawa ke bank sampah: botol plastik, kaleng, dan kardus punya nilai jual.\n"
#         "• Cuci bersih kemasan sebelum disetor agar mudah didaur ulang."
#     ),
#     "B3": (
#         "• JANGAN buang ke tempat sampah biasa — B3 mencemari tanah & air tanah.\n"
#         "• Kumpulkan di drop-box B3 (apotek, minimarket, atau dinas lingkungan hidup).\n"
#         "• Baterai, lampu, elektronik rusak, dan cat termasuk kategori B3.\n"
#         "• Penanganan salah dapat menyebabkan kebakaran atau keracunan logam berat."
#     ),
#     "Organik": (
#         "• Bisa dijadikan kompos dalam 4–8 minggu dengan metode sederhana di rumah.\n"
#         "• Jangan campur dengan plastik — mempersulit pengomposan.\n"
#         "• Sisa sayur, buah, dan makanan basi sangat cocok untuk eco-enzyme.\n"
#         "• Kompos yang dihasilkan bisa dipakai langsung untuk pupuk tanaman."
#     ),
# }
# 
# # ================= LOAD MODEL =================
# print(f"[+] Loading model dari: {MODEL_PATH}")
# try:
#     # [!] Hapus custom_objects karena MobileNetV2 tidak butuh ChannelAttentionLayer
#     model = tf.keras.models.load_model(MODEL_PATH, compile=False)
#     print("[+] Model MobileNetV2 loaded!")
# except Exception as e:
#     model = None
#     print(f"[!] Error loading model: {e}")
# 
# # ================= GEMINI SETUP =================
# GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAMwyORIARVCBAboc6GvCI1uI0W5XD7Kxw")
# gemini_client  = None
# tips_cache: dict[str, str] = {}
# 
# if GEMINI_API_KEY:
#     gemini_client = genai.Client(api_key=GEMINI_API_KEY)
#     print("[+] Gemini siap!")
# else:
#     print("[!] GEMINI_API_KEY tidak ditemukan. Endpoint /genai/* tidak aktif.")
# 
# class TipsReq(BaseModel):
#     kategori: str
# 
# class AskReq(BaseModel):
#     pertanyaan: str
#     kategori: str = ""
# 
# # ================= ROUTES =================
# @app.get("/")
# def home():
#     return {
#         "status": "online",
#         "model": "model_sampah_MobileNetV2.keras",
#         "classes": CLASS_NAMES,
#         "gemini": "ready" if gemini_client else "unavailable (set GEMINI_API_KEY)",
#     }
# 
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     if model is None:
#         return {"error": "Model tidak ter-load di server"}
# 
#     try:
#         image_bytes = await file.read()
# 
#         # ── [FIX: Memory Leak] Context manager → PIL Image di-close otomatis ──
#         with Image.open(io.BytesIO(image_bytes)) as img:
#             image = img.convert("RGB")
#             image = image.resize(IMG_SIZE)                    # 1. Resize ke 150×150
#             img_array = np.array(image, dtype=np.float32)     # 2. Ke numpy float32
# 
#         # Buffer gambar sudah tidak dibutuhkan, bebaskan segera
#         del image_bytes
# 
#         img_array = np.expand_dims(img_array, axis=0)         # 3. Batch (1,150,150,3)
#         img_array = preprocess_input(img_array)                # 4. Preprocess MobileNetV2
# 
#         # ── [FIX: Event Loop Blocking] Inferensi di thread pool ──────────────
#         # model() direct-call lebih ringan dari model.predict() untuk single-frame.
#         # asyncio.to_thread memastikan event loop uvicorn TIDAK terblokir,
#         # sehingga request dari user lain tetap bisa diterima secara bersamaan.
#         def _run_inference(arr):
#             return model(arr, training=False).numpy()[0]
# 
#         cls_preds = await asyncio.to_thread(_run_inference, img_array)
#         # ─────────────────────────────────────────────────────────────────────
# 
#         class_idx  = int(np.argmax(cls_preds))
#         confidence = float(cls_preds[class_idx])
#         kategori   = CLASS_NAMES[class_idx]
# 
#         print(f"[+] Deteksi: {kategori} ({confidence:.1%})")
# 
#         return {
#             "status": "success",
#             "hasil": [{
#                 "kategori":   kategori,
#                 "confidence": confidence,
#             }],
#         }
# 
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         return {"status": "error", "message": str(e)}
# 
# @app.post("/genai/tips/")
# async def genai_tips(req: TipsReq):
#     if gemini_client is None:
#         return {"status": "error", "message": "Gemini tidak aktif. Set env var GEMINI_API_KEY."}
# 
#     k = req.kategori
#     if k in tips_cache:
#         return {"status": "success", "tips": tips_cache[k], "cached": True}
# 
#     prompt = (
#         f"Kamu adalah asisten edukasi lingkungan. Sampah yang terdeteksi: '{k}'. "
#         f"Berikan 3-4 poin singkat dalam Bahasa Indonesia yang informatif: "
#         f"(1) cara memilah yang benar, (2) dampak lingkungan jika salah buang, "
#         f"(3) nilai daur ulang atau manfaat ekonomi. "
#         f"Gunakan format bullet • dan bahasa yang santai tapi edukatif."
#     )
#     try:
#         resp = await asyncio.to_thread(
#             gemini_client.models.generate_content,
#             model="gemini-2.0-flash-lite",
#             contents=prompt,
#         )
#         tips_cache[k] = resp.text
#         return {"status": "success", "tips": resp.text, "cached": False, "source": "gemini"}
#     except Exception as e:
#         print(f"[!] Gemini tips error (fallback ke static): {e}")
#         fallback = STATIC_TIPS.get(k, "Pilah sampah sesuai kategorinya sebelum dibuang.")
#         return {"status": "success", "tips": fallback, "cached": False, "source": "static"}
# 
# @app.post("/genai/ask/")
# async def genai_ask(req: AskReq):
#     if gemini_client is None:
#         return {"status": "error", "message": "Gemini tidak aktif. Set env var GEMINI_API_KEY."}
# 
#     ctx = f"Konteks: sampah terakhir terdeteksi adalah '{req.kategori}'. " if req.kategori else ""
#     prompt = (
#         f"Kamu adalah asisten edukasi pengelolaan sampah yang ramah dan informatif. "
#         f"{ctx}"
#         f"Jawab pertanyaan berikut dalam Bahasa Indonesia secara ringkas (maksimal 4 kalimat): "
#         f"{req.pertanyaan}"
#     )
#     try:
#         resp = await asyncio.to_thread(
#             gemini_client.models.generate_content,
#             model="gemini-2.0-flash-lite",
#             contents=prompt,
#         )
#         return {"status": "success", "jawaban": resp.text}
#     except Exception as e:
#         print(f"[!] Gemini ask error (fallback ke static): {e}")
#         tips = STATIC_TIPS.get(req.kategori, "")
#         jawaban = (
#             f"Maaf, AI sedang tidak tersedia. "
#             + (f"Berikut informasi dasar tentang sampah {req.kategori}:\n\n{tips}" if tips
#                else "Silakan pilah sampah sesuai kategorinya: Organik, Anorganik, atau B3.")
#         )
#         return {"status": "success", "jawaban": jawaban}
# 
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# 

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

app = FastAPI(title="EcoSort AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= CONFIG =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model_sampah_AdvancedV2.keras")
IMG_SIZE   = (224, 224)

# 3 kelas sesuai flow_from_directory (alphabetical)
CLASS_NAMES = ["Anorganik", "B3", "Organik"]

# Tips statis sebagai fallback jika Gemini tidak tersedia
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

# ================= CUSTOM LAYER (diperlukan saat load_model) =================
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
        ch    = tf.shape(x)[-1]
        scale = tf.reshape(avg_w + max_w, [-1, 1, 1, ch])
        return x * scale

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"reduction_ratio": self.reduction_ratio})
        return cfg

# ================= LOAD MODEL =================
print(f"[+] Loading model dari: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={"ChannelAttentionLayer": ChannelAttentionLayer},
    )
    print("[+] Model loaded!")
except Exception as e:
    model = None
    print(f"[!] Error loading model: {e}")

# ================= CUSTOM LAYER: BahdanauAttention (LSTM) =================
@tf.keras.utils.register_keras_serializable()
class BahdanauAttention(tf.keras.layers.Layer):
    """Bahdanau-style attention — diperlukan saat deserialize volume_baseline_lstm.keras."""
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        feat_dim = input_shape[-1]
        self.W_q = self.add_weight(name='W_q', shape=(feat_dim, self.units))
        self.W_k = self.add_weight(name='W_k', shape=(feat_dim, self.units))
        self.W_v = self.add_weight(name='W_v', shape=(feat_dim, feat_dim))
        self.v   = self.add_weight(name='v',   shape=(self.units, 1))
        self.b   = self.add_weight(name='b',   shape=(self.units,), initializer='zeros')
        super().build(input_shape)

    def call(self, x):
        query = tf.tensordot(x, self.W_q, axes=[[2], [0]])
        key   = tf.tensordot(x, self.W_k, axes=[[2], [0]])
        score = tf.tensordot(tf.tanh(query + key + self.b), self.v, axes=[[2], [0]])
        alpha = tf.nn.softmax(score, axis=1)
        value = tf.tensordot(x, self.W_v, axes=[[2], [0]])
        ctx   = tf.reduce_sum(alpha * value, axis=1)
        return ctx

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg

# ================= LOAD LSTM MODEL & SCALER =================
LSTM_MODEL_PATH  = os.path.join(BASE_DIR, "volume_baseline_lstm.keras")
SCALER_MEAN_PATH = os.path.join(BASE_DIR, "scaler_mean.npy")
SCALER_SCALE_PATH= os.path.join(BASE_DIR, "scaler_scale.npy")

lstm_model    = None
scaler_mean_  = None
scaler_scale_ = None

try:
    # ── Workaround: Keras version mismatch ──────────────────────────────────
    # Model LSTM disimpan dari Keras yang menyertakan 'quantization_config'
    # di config Dense layer. Keras 3.12.0 lokal tidak mengenali field ini
    # dan throw TypeError. Solusi: patch Dense.from_config untuk menghapus
    # key tersebut sebelum instantiasi.
    _original_dense_from_config = tf.keras.layers.Dense.from_config.__func__

    @classmethod  # type: ignore[misc]
    def _patched_dense_from_config(cls, config):
        config.pop("quantization_config", None)
        return _original_dense_from_config(cls, config)

    tf.keras.layers.Dense.from_config = _patched_dense_from_config

    lstm_model = tf.keras.models.load_model(
        LSTM_MODEL_PATH,
        custom_objects={"BahdanauAttention": BahdanauAttention},
        compile=False,
    )

    # Restore original Dense.from_config setelah load selesai
    tf.keras.layers.Dense.from_config = _original_dense_from_config

    scaler_mean_  = np.load(SCALER_MEAN_PATH)   # shape (7,)
    scaler_scale_ = np.load(SCALER_SCALE_PATH)   # shape (7,)
    print(f"[+] LSTM model loaded! input={lstm_model.input_shape} output={lstm_model.output_shape}")
except Exception as e:
    lstm_model = None
    print(f"[!] LSTM model gagal di-load (endpoint /predict-volume tetap tersedia, tapi akan error): {e}")

# ================= GEMINI SETUP =================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "AIzaSyAMwyORIARVCBAboc6GvCI1uI0W5XD7Kxw")
gemini_client  = None
tips_cache: dict[str, str] = {}

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    print("[+] Gemini siap!")
else:
    print("[!] GEMINI_API_KEY tidak ditemukan. Endpoint /genai/* tidak aktif.")

class TipsReq(BaseModel):
    kategori: str

class AskReq(BaseModel):
    pertanyaan: str
    kategori: str = ""

# ── Pydantic Schema: Volume Prediction (LSTM) ────────────────────────────────
from typing import List

class VolumeTimestep(BaseModel):
    """Satu baris data historis pembuangan sampah per bulan.
    Setiap timestep terdiri dari 7 fitur yang akan di-scale lalu
    dimasukkan ke LSTM sebagai 1 dari 12 timestep.

    Fitur (urutan HARUS sesuai notebook):
      0. volume_ton    — total volume sampah bulan itu (ton)
      1. sin_month     — sin(2π × bulan / 12)  [seasonality encoding]
      2. cos_month     — cos(2π × bulan / 12)  [seasonality encoding]
      3. area_encoded  — 0=Rural, 1=Semi-Urban, 2=Urban
      4. year_norm     — (tahun − 2019) / 4.0
      5. volume_ma     — moving-average volume (boleh = volume_ton jika tidak ada)
      6. volume_ema    — exponential-moving-average volume (boleh = volume_ton)
    """
    volume_ton:   float
    sin_month:    float
    cos_month:    float
    area_encoded: float
    year_norm:    float
    volume_ma:    float
    volume_ema:   float

class PredictVolumeReq(BaseModel):
    """Request body untuk POST /predict-volume.
    `history` harus berisi tepat 12 objek VolumeTimestep
    yang merepresentasikan 12 bulan terakhir data pembuangan.
    """
    history: List[VolumeTimestep]  # HARUS 12 item (window_size = 12)

# ================= ROUTES =================
@app.get("/")
def home():
    return {
        "status": "online",
        "models": {
            "cnn": "model_sampah_AdvancedV2.keras (EfficientNetB3)",
            "lstm": "volume_baseline_lstm.keras (LSTM+BahdanauAttention)"
                    if lstm_model else "NOT LOADED",
        },
        "classes": CLASS_NAMES,
        "gemini": "ready" if gemini_client else "unavailable (set GEMINI_API_KEY)",
    }


# ================= ENDPOINT: PREDICT VOLUME (LSTM) =================
WINDOW_SIZE = 12  # Model LSTM expects exactly 12 timesteps
N_FEATURES  = 7   # Jumlah fitur per timestep

@app.post("/predict-volume/")
async def predict_volume(req: PredictVolumeReq):
    """Prediksi volume sampah 3 bulan ke depan menggunakan model LSTM.

    Flow:
      1. Validasi panjang array history == 12
      2. Convert list of Pydantic objects → numpy array (12, 7)
      3. StandardScaler transform: (X − mean) / scale
      4. Expand dims → (1, 12, 7) untuk batch
      5. model.predict() di thread pool (non-blocking)
      6. Inverse-scale output: pred × scale[0] + mean[0]
         (index 0 = kolom volume_ton)
      7. Return JSON array 3 bulan prediksi
    """
    # ── Guard: model harus sudah ter-load ─────────────────────────────────────
    if lstm_model is None or scaler_mean_ is None or scaler_scale_ is None:
        return {
            "status": "error",
            "message": "Model LSTM atau file scaler belum ter-load di server.",
        }

    # ── Validasi panjang window ───────────────────────────────────────────────
    if len(req.history) != WINDOW_SIZE:
        return {
            "status": "error",
            "message": f"history harus berisi tepat {WINDOW_SIZE} timestep, "
                       f"diterima {len(req.history)}.",
        }

    try:
        # ── 1. Convert Pydantic → numpy (12, 7) ─────────────────────────────
        raw_rows = [
            [
                ts.volume_ton,
                ts.sin_month,
                ts.cos_month,
                ts.area_encoded,
                ts.year_norm,
                ts.volume_ma,
                ts.volume_ema,
            ]
            for ts in req.history
        ]
        X_raw = np.array(raw_rows, dtype=np.float32)  # (12, 7)

        # ── 2. StandardScaler transform ──────────────────────────────────────
        # Rumus: X_scaled = (X_raw − mean) / scale
        # mean & scale masing-masing shape (7,), broadcast otomatis ke (12,7)
        X_scaled = (X_raw - scaler_mean_) / scaler_scale_

        # ── 3. Expand dims → batch (1, 12, 7) ───────────────────────────────
        X_batch = np.expand_dims(X_scaled, axis=0).astype(np.float32)

        # ── 4. Inferensi LSTM (NON-BLOCKING via asyncio.to_thread) ───────────
        def _run_lstm_inference(batch):
            return lstm_model(batch, training=False).numpy()[0]

        pred_scaled = await asyncio.to_thread(_run_lstm_inference, X_batch)
        # pred_scaled shape: (3,) — 3 bulan ke depan, masih dalam skala scaled

        # ── 5. Inverse scaling ───────────────────────────────────────────────
        # Output model memprediksi volume_ton (index 0 dari fitur).
        # Inverse: pred_real = pred_scaled × scale[0] + mean[0]
        pred_volume_ton = (pred_scaled * float(scaler_scale_[0])) + float(scaler_mean_[0])

        # Pastikan volume tidak negatif
        pred_volume_ton = np.maximum(pred_volume_ton, 0.0)

        predictions = [
            {"bulan_ke": i + 1, "volume_ton": round(float(v), 2)}
            for i, v in enumerate(pred_volume_ton)
        ]

        print(f"[+] LSTM Prediksi: {[p['volume_ton'] for p in predictions]} ton")

        return {
            "status": "success",
            "model_used": "LSTM+BahdanauAttention",
            "predictions": predictions,
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "Model tidak ter-load di server"}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess sesuai EfficientNetB3
        image     = image.resize(IMG_SIZE)
        img_array = np.array(image, dtype=np.float32)
        img_array = eff_preprocess(img_array)
        img_tensor = tf.expand_dims(tf.convert_to_tensor(img_array), axis=0)

        cls_preds = model.predict(img_tensor, verbose=0)[0]
        class_idx  = int(np.argmax(cls_preds))
        confidence = float(cls_preds[class_idx])
        kategori   = CLASS_NAMES[class_idx]

        print(f"[+] Deteksi: {kategori} ({confidence:.1%})")

        return {
            "status": "success",
            "hasil": [{
                "kategori":   kategori,
                "confidence": confidence,
            }],
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.post("/genai/tips/")
async def genai_tips(req: TipsReq):
    if gemini_client is None:
        return {"status": "error", "message": "Gemini tidak aktif. Set env var GEMINI_API_KEY."}

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
        print(f"[!] Gemini tips error (fallback ke static): {e}")
        fallback = STATIC_TIPS.get(k, "Pilah sampah sesuai kategorinya sebelum dibuang.")
        return {"status": "success", "tips": fallback, "cached": False, "source": "static"}


@app.post("/genai/ask/")
async def genai_ask(req: AskReq):
    if gemini_client is None:
        return {"status": "error", "message": "Gemini tidak aktif. Set env var GEMINI_API_KEY."}

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
        print(f"[!] Gemini ask error (fallback ke static): {e}")
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
