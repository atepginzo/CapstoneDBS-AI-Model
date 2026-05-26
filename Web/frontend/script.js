const video       = document.getElementById('webcam');
const btn         = document.getElementById('toggleCamera');
const statusText  = document.getElementById('status-text');
const pulse       = document.querySelector('.pulse');
const resCategory = document.getElementById('res-category');
const resConfText = document.getElementById('res-conf-text');
const resConfBar  = document.getElementById('res-conf-bar');
const resTips     = document.getElementById('res-tips');

let isDetecting     = false;
let detectionTimeout = null;
let lastKategori    = "";

// ── Akses Kamera ─────────────────────────────────────────────────────────────
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "environment" }
        });
        video.srcObject = stream;
        video.play();
        return true;
    } catch (err) {
        console.error("Gagal akses kamera:", err);
        alert("Gagal mengakses kamera. Pastikan izin kamera sudah diberikan.");
        return false;
    }
}

// ── Toggle Deteksi ────────────────────────────────────────────────────────────
btn.addEventListener('click', async () => {
    if (!isDetecting) {
        const success = await startCamera();
        if (success) {
            isDetecting = true;
            btn.innerHTML = "🛑 Hentikan Sistem";
            btn.style.background = "linear-gradient(135deg, #ff4444, #cc0000)";
            pulse.classList.add('active-pulse');
            statusText.innerText = "Sistem Aktif! Bersiap menjepret...";
            detectionTimeout = setTimeout(runDetectionCycle, 3000);
        }
    } else {
        stopDetection();
    }
});

function stopDetection() {
    isDetecting = false;
    clearTimeout(detectionTimeout);

    const stream = video.srcObject;
    if (stream) stream.getTracks().forEach(t => t.stop());

    video.srcObject = null;
    btn.innerHTML = "Mulai Deteksi Kamera";
    btn.style.background = "linear-gradient(135deg, var(--primary), var(--secondary))";
    statusText.innerText = "Standby";
    pulse.classList.remove('active-pulse');

    resCategory.innerText = "-";
    resConfText.innerText = "0%";
    resConfBar.style.width = "0%";
    resTips.innerHTML = "Tunggu deteksi objek untuk melihat saran pengelolaan sampah...";
}

// ── Siklus Deteksi (shutter per frame) ───────────────────────────────────────
async function runDetectionCycle() {
    if (!isDetecting || video.readyState !== 4) return;

    // Efek flash shutter
    const flash = document.createElement('div');
    Object.assign(flash.style, {
        position: 'fixed', top: '0', left: '0',
        width: '100vw', height: '100vh',
        backgroundColor: 'white', zIndex: '9999',
        opacity: '0.8', transition: 'opacity 0.2s ease-out'
    });
    document.body.appendChild(flash);
    setTimeout(() => { flash.style.opacity = '0'; setTimeout(() => flash.remove(), 200); }, 50);

    // Capture frame ke canvas 640×640
    const canvas = document.createElement('canvas');
    canvas.width = 640; canvas.height = 640;
    canvas.getContext('2d').drawImage(video, 0, 0, 640, 640);

    video.pause();
    statusText.innerText = "🔍 Menganalisis Sampah...";

    canvas.toBlob(async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'capture.jpg');

        try {
            const response = await fetch('http://localhost:8000/predict/', {
                method: 'POST', body: formData
            });
            const data = await response.json();

            if (data.status === "success" && data.hasil?.length > 0) {
                updateUI(data.hasil[0]);
                statusText.innerText = "✅ Deteksi Selesai!";
            } else {
                resCategory.innerText = "Tidak Dikenali";
                statusText.innerText = "❌ Error: " + (data.message?.substring(0, 40) ?? "Coba lagi");
            }
        } catch (err) {
            console.error("Koneksi backend gagal:", err);
            statusText.innerText = "❌ Error Koneksi Backend";
        }

        // Freeze 2 detik, lalu lanjut siklus
        setTimeout(() => {
            if (!isDetecting) return;
            video.play();
            statusText.innerText = "Kamera Live... Menunggu jepretan berikutnya.";
            detectionTimeout = setTimeout(runDetectionCycle, 5000);
        }, 2000);

    }, 'image/jpeg', 0.8);
}

// ── Update UI hasil deteksi ───────────────────────────────────────────────────
function updateUI(data) {
    const kategori = data.kategori;
    const conf     = (data.confidence * 100).toFixed(1);

    lastKategori = kategori;

    resCategory.innerText  = kategori;
    resConfText.innerText  = conf + "%";
    resConfBar.style.width = conf + "%";

    // Warna per kelas
    const colorMap = { "B3": "#ff4444", "Anorganik": "#33b5e5", "Organik": "#00C851" };
    resCategory.style.color = colorMap[kategori] ?? "#ffffff";

    // Minta tips dari Gemini (cached setelah pertama kali)
    fetchTips(kategori);
}

// ── Fetch tips dari Gemini (auto setelah deteksi) ─────────────────────────────
async function fetchTips(kategori) {
    resTips.innerHTML = "<em>⏳ Memuat saran dari AI...</em>";
    try {
        const res  = await fetch('http://localhost:8000/genai/tips/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ kategori })
        });
        const data = await res.json();
        if (data.status === "success") {
            // Render bullet • sebagai list item
            const html = data.tips
                .split('\n')
                .filter(l => l.trim())
                .map(l => `<p>${l}</p>`)
                .join('');
            resTips.innerHTML = html;
        } else {
            resTips.innerHTML = "<em>Tidak dapat memuat saran AI.</em>";
        }
    } catch (err) {
        resTips.innerHTML = "<em>Tidak dapat terhubung ke AI.</em>";
    }
}

// ── Tanya AI on-demand ────────────────────────────────────────────────────────
async function askGemini() {
    const input  = document.getElementById('ai-input');
    const answer = document.getElementById('ai-answer');
    const askBtn = document.getElementById('ai-ask-btn');

    const pertanyaan = input.value.trim();
    if (!pertanyaan) return;

    askBtn.disabled   = true;
    askBtn.innerText  = "⏳ Menunggu...";
    answer.innerHTML  = "<em>⏳ AI sedang berpikir...</em>";

    try {
        const res  = await fetch('http://localhost:8000/genai/ask/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pertanyaan, kategori: lastKategori })
        });
        const data = await res.json();
        if (data.status === "success") {
            answer.innerHTML = data.jawaban.replace(/\n/g, '<br>');
        } else {
            answer.innerHTML = "<em>Gagal mendapat jawaban: " + (data.message ?? "error") + "</em>";
        }
    } catch (err) {
        answer.innerHTML = "<em>❌ Tidak dapat terhubung ke AI.</em>";
    } finally {
        askBtn.disabled  = false;
        askBtn.innerText = "Tanya";
    }
}

// Enter key di input langsung trigger askGemini
document.addEventListener('DOMContentLoaded', () => {
    const inputEl = document.getElementById('ai-input');
    if (inputEl) {
        inputEl.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') askGemini();
        });
    }
});
