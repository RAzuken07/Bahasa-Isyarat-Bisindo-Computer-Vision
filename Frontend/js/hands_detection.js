// VisiSign - MediaPipe Hands Detection + FastAPI Integration
// FIX: Pembaruan UI mandiri, auto-start stream webcam, dan sinkronisasi DOM Real-time

document.addEventListener('DOMContentLoaded', async () => {

    console.log('MediaPipe Hands Module Loaded');

    const API_HOSTS = [
        'http://127.0.0.1:8001',
        'http://127.0.0.1:8000'
    ];
    let apiHost = API_HOSTS[0];
    let apiOnline = false;

    const getApiUrl = () => `${apiHost}/predict`;

    const checkApiHost = async () => {
        for (const host of API_HOSTS) {
            try {
                const response = await fetch(`${host}/health`, { method: 'GET' });
                if (response.ok) {
                    apiHost = host;
                    apiOnline = true;
                    console.log(`✅ API host set to ${host}`);
                    return;
                }
            } catch (error) {
                // abaikan dan coba host berikutnya
            }
        }
        apiOnline = false;
        console.warn('⚠️ API tidak dapat dijangkau pada semua host');
    };

    await checkApiHost();

    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('canvas');

    const predictionText   = document.getElementById('prediction-text');
    const confidenceText   = document.getElementById('confidence-text');
    const btnClearHistory  = document.getElementById('btn-clear-history');
    const btnResetPrediction = document.getElementById('btn-reset-prediction');
    const btnToggleMode    = document.getElementById('btn-toggle-mode');
    const historyLog       = document.getElementById('history-log');

    let sequence = [];
    let currentMode = "STATIC"; // Default mode awal
    let lastLoggedWord = "";
    let isProcessing = false;
    let predictionResetTimer = null;

    // Fungsi untuk mereset tampilan teks ke kondisi awal
    const resetPredictionDisplay = () => {
        if (predictionText) {
            predictionText.innerText = "Menunggu gerakan...";
            predictionText.className = "text-3xl font-extrabold text-slate-400 italic mt-2 transition-all duration-200";
        }
        if (confidenceText) {
            confidenceText.innerText = "0%";
        }
    };

    // Fungsi utama mengirim koordinat landmark ke API Backend
    async function sendLandmarksToBackend(landmarksSequence) {
        if (!apiOnline || isProcessing) return;

        isProcessing = true;
        try {
            const response = await fetch(getApiUrl(), {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    sequence: landmarksSequence,
                    mode: currentMode
                })
            });

            if (response.ok) {
                const data = await response.json();

                // Bersihkan timer auto-reset jika ada data gerakan masuk
                clearTimeout(predictionResetTimer);

                if (data.prediction && data.prediction !== "Menunggu gerakan...") {
                    // Update teks tebakan kata ke antarmuka web
                    if (predictionText) {
                        predictionText.innerText = data.prediction;
                        predictionText.className = "text-4xl font-black text-teal-600 mt-2 transition-all duration-200 scale-105";
                    }
                    
                    // Update persentase akurasi model
                    if (confidenceText) {
                        const score = data.confidence ? data.confidence.toFixed(1) : "0.0";
                        confidenceText.innerText = `${score}%`;
                    }

                    // Masukkan ke dalam kotak list riwayat teks web
                    updatePredictionHistory(data.prediction);

                } else {
                    resetPredictionDisplay();
                }

                // Buat timer auto-reset jika pengguna menjauhkan tangan selama 3 detik
                predictionResetTimer = setTimeout(() => {
                    resetPredictionDisplay();
                }, 3000);

            }
        } catch (error) {
            console.error("Gagal interaksi dengan API server:", error);
        } finally {
            isProcessing = false;
        }
    }

    // Fungsi mencatat riwayat kata ke log HTML
    function updatePredictionHistory(word) {
        if (!historyLog || word === lastLoggedWord) return;

        // Bersihkan teks default "Belum ada kata terdeteksi"
        const placeholder = historyLog.querySelector('.italic');
        if (placeholder) placeholder.remove();

        const timeNow = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        
        const logItem = document.createElement('div');
        logItem.className = "flex justify-between items-center py-2 border-b border-slate-100 text-xs text-slate-600 animate-fade-in";
        logItem.innerHTML = `
            <span class="font-semibold text-slate-800">${word}</span>
            <span class="text-[10px] text-slate-400">${timeNow}</span>
        `;
        
        // Taruh riwayat terbaru di paling atas list
        historyLog.insertBefore(logItem, historyLog.firstChild);
        lastLoggedWord = word;
    }

    // =====================================================
    // MEDIAPIPE LOGIC (ON RESULTS)
    // =====================================================
    const canvasCtx = canvasElement ? canvasElement.getContext('2d') : null;

    function onResults(results) {
        if (!canvasElement || !canvasCtx) return;

        // Atur dimensi canvas pas dengan video webcam aktif
        canvasElement.width = webcamElement.videoWidth || 640;
        canvasElement.height = webcamElement.videoHeight || 480;

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        // Gambar ulang frame video ke canvas (efek mirror)
        canvasCtx.translate(canvasElement.width, 0);
        canvasCtx.scale(-1, 1);
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.restore();

        // Buat struktur array kosong penampung koordinat untuk 2 tangan (2 * 21 * 3 = 126 nilai)
        let frameLandmarks = new Array(126).fill(0);

        if (results.multiHandLandmarks && results.multiHandedness) {
            // Mirror transform agar landmark sinkron dengan video yang sudah di-mirror
            canvasCtx.save();
            canvasCtx.translate(canvasElement.width, 0);
            canvasCtx.scale(-1, 1);

            for (let index = 0; index < results.multiHandLandmarks.length; index++) {
                const classification = results.multiHandedness[index];
                const isLeftHand = classification.label === 'Left';
                
                // Pisahkan slot array indeks: Tangan Kiri di slot awal (0-62), Kanan di slot akhir (63-125)
                let offset = isLeftHand ? 0 : 63;
                const landmarks = results.multiHandLandmarks[index];

                // Simpan koordinat x, y, z dari MediaPipe
                for (let i = 0; i < 21; i++) {
                    frameLandmarks[offset + (i * 3)]     = landmarks[i].x;
                    frameLandmarks[offset + (i * 3) + 1] = landmarks[i].y;
                    frameLandmarks[offset + (i * 3) + 2] = landmarks[i].z;
                }

                // Menggambar kerangka visual tangan pada layar canvas web
                if (window.drawConnectors && window.drawLandmarks) {
                    window.drawConnectors(canvasCtx, landmarks, window.HAND_CONNECTIONS, {color: '#0D9488', lineWidth: 3});
                    window.drawLandmarks(canvasCtx, landmarks, {color: '#0D9488', lineWidth: 1, radius: 3});
                }
            }

            canvasCtx.restore(); // Akhiri mirror transform untuk landmark

            // Tambah koordinat frame saat ini ke tumpukan memori runtunan waktu (sequence)
            sequence.push(frameLandmarks);
            
            // Batasi panjang runtunan frame (maksimal 30 frame sesuai konfigurasi sistem)
            if (sequence.length > 30) {
                sequence.shift();
            }

            // Kirim data ke API server jika syarat frame mencukupi
            if (currentMode === "STATIC" || (currentMode === "DYNAMIC" && sequence.length === 30)) {
                sendLandmarksToBackend(sequence);
            }
        } else {
            // Jika tangan hilang dari webcam, kosongkan runtunan memori perlahan
            if (sequence.length > 0) sequence.shift();
        }
    }

    // Inisialisasi Google MediaPipe Hands
    if (window.Hands && webcamElement) {
        const hands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });

        hands.setOptions({
            maxNumHands: 2,
            modelComplexity: 1,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        hands.onResults(onResults);

        // PERBAIKAN UTAMA: Mengaktifkan navigator mediaDevices secara langsung 
        // untuk menjamin stream video mengalir ke MediaPipe tanpa hambatan skrip lain
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
                .then((stream) => {
                    webcamElement.srcObject = stream;
                    webcamElement.play();
                    
                    // Hubungkan kamera ke generator frame MediaPipe
                    if (window.Camera) {
                        const camera = new Camera(webcamElement, {
                            onFrame: async () => {
                                await hands.send({ image: webcamElement });
                            },
                            width: 640,
                            height: 480
                        });
                        camera.start().then(() => console.log("📷 MediaPipe Core Camera Engine Aktif."));
                    }
                })
                .catch((err) => {
                    console.error("Gagal mendapatkan akses hardware Kamera: ", err);
                    alert("Aplikasi memerlukan izin akses kamera untuk melacak bahasa isyarat.");
                });
        }
    }

    // =====================================================
    // TOMBOL INTERAKSI UI INTERFACE BUTTON
    // =====================================================

    if (btnToggleMode) {
        btnToggleMode.addEventListener('click', (e) => {
            e.preventDefault();
            currentMode = (currentMode === "STATIC") ? "DYNAMIC" : "STATIC";
            sequence = []; // Reset koordinat
            resetPredictionDisplay();

            // Ubah teks indikator tombol sesuai mode aktif
            if (currentMode === "STATIC") {
                btnToggleMode.innerHTML = `
                    <span class="w-7 h-7 bg-teal-600 text-white rounded-lg flex items-center justify-center text-[11px]">
                        <i class="fa-solid fa-font"></i>
                    </span>
                    <div>
                        <p>Mode: Ejaan Abjad</p>
                        <p class="text-[9px] text-teal-500 font-normal">Ubah ke kata gerakan</p>
                    </div>
                `;
            } else {
                btnToggleMode.innerHTML = `
                    <span class="w-7 h-7 bg-green-600 text-white rounded-lg flex items-center justify-center text-[11px]">
                        <i class="fa-solid fa-exchange-alt"></i>
                    </span>
                    <div>
                        <p>Mode: Kata Gerakan</p>
                        <p class="text-[9px] text-green-500 font-normal">Ubah ke huruf/angka</p>
                    </div>
                `;
            }
            console.log("Mode Aplikasi Berganti Ke:", currentMode);
        });
    }

    if (btnResetPrediction) {
        btnResetPrediction.addEventListener('click', (e) => {
            e.preventDefault();
            sequence = [];
            lastLoggedWord = "";
            clearTimeout(predictionResetTimer);
            resetPredictionDisplay();
        });
    }

    if (btnClearHistory) {
        btnClearHistory.addEventListener('click', (e) => {
            e.preventDefault();
            historyLog.innerHTML = `
                <div class="py-2.5 text-center text-slate-400 text-[11px] italic">
                    Belum ada kata terdeteksi
                </div>`;
            lastLoggedWord = "";
        });
    }
});