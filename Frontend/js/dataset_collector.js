// VisiSign - Dataset Collector Script
// Terintegrasi dengan MediaPipe Hands, FastAPI dataset endpoints, hitung mundur 3 detik, dan polling logs pelatihan.

document.addEventListener('DOMContentLoaded', async () => {
    console.log('Dataset Collector Module Loaded');

    const API_HOSTS = [
        'http://127.0.0.1:8001',
        'http://127.0.0.1:8000'
    ];
    let apiHost = API_HOSTS[0];
    let apiOnline = false;

    // Element Selector
    const webcamElement = document.getElementById('webcam');
    const canvasElement = document.getElementById('canvas');
    const webcamFallback = document.getElementById('webcam-fallback');
    
    const tabStatic = document.getElementById('tab-static');
    const tabDynamic = document.getElementById('tab-dynamic');
    const labelInput = document.getElementById('label-input');
    
    const btnCollect = document.getElementById('btn-collect');
    const btnCollectText = document.getElementById('btn-collect-text');
    const activeModeBadge = document.getElementById('active-mode-badge');
    const statusDot = document.getElementById('status-dot');
    
    const statsTotalBadge = document.getElementById('stats-total-badge');
    const statsTableBody = document.getElementById('stats-table-body');
    
    const btnTrain = document.getElementById('btn-train');
    const trainModeLabel = document.getElementById('train-mode-label');
    const trainingStatusText = document.getElementById('training-status-text');
    const logConsole = document.getElementById('log-console');
    
    const countdownOverlay = document.getElementById('countdown-overlay');
    const countdownText = document.getElementById('countdown-text');
    const countdownSubtext = document.getElementById('countdown-subtext');
    
    const recordingOverlay = document.getElementById('recording-overlay');
    const recordingProgress = document.getElementById('recording-progress');

    // State
    let currentMode = "STATIC"; // STATIC atau DYNAMIC
    let latestFrameLandmarks = new Array(126).fill(0);
    let handDetected = false;
    let isRecording = false;
    let recordingBuffer = [];
    let isTrainingActive = false;
    let trainingPollInterval = null;

    // Check API Host
    const checkApiHost = async () => {
        for (const host of API_HOSTS) {
            try {
                const response = await fetch(`${host}/health`, { method: 'GET' });
                if (response.ok) {
                    apiHost = host;
                    apiOnline = true;
                    console.log(`✅ API host terhubung ke ${host}`);
                    return;
                }
            } catch (error) {
                // lanjut coba host berikutnya
            }
        }
        apiOnline = false;
        console.warn('⚠️ Server API tidak aktif.');
    };

    await checkApiHost();

    // Fetch and Render Dataset Statistics
    const fetchStats = async () => {
        if (!apiOnline) return;
        try {
            const response = await fetch(`${apiHost}/dataset/stats`);
            if (response.ok) {
                const data = await response.json();
                renderStats(data);
            }
        } catch (error) {
            console.error("Gagal memuat statistik dataset:", error);
        }
    };

    const renderStats = (data) => {
        const modeKey = currentMode.toLowerCase();
        const stats = data[modeKey];
        
        statsTotalBadge.innerText = `Total: ${stats.total_samples} Sampel`;
        
        statsTableBody.innerHTML = '';
        const labels = Object.keys(stats.labels_count).sort();
        
        if (labels.length === 0) {
            statsTableBody.innerHTML = `
                <tr>
                    <td colspan="2" class="py-4 text-center text-slate-400 italic">Belum ada data sampel terkumpul untuk mode ini.</td>
                </tr>
            `;
            return;
        }

        labels.forEach(label => {
            const count = stats.labels_count[label];
            const row = document.createElement('tr');
            row.className = "border-b border-slate-100 hover:bg-slate-50 transition";
            row.innerHTML = `
                <td class="py-3 font-semibold text-slate-800">${label}</td>
                <td class="py-3 text-right text-teal-600 font-bold">${count}</td>
            `;
            statsTableBody.appendChild(row);
        });
    };

    // Initial Load Statistics
    fetchStats();

    // Mode Toggle Logic
    const setMode = (mode) => {
        currentMode = mode;
        if (mode === "STATIC") {
            // Tab style
            tabStatic.className = "flex-1 py-2 text-xs font-bold rounded-lg bg-white text-teal-800 shadow-sm transition";
            tabDynamic.className = "flex-1 py-2 text-xs font-bold rounded-lg text-slate-600 hover:text-slate-800 transition";
            
            // Collect button
            btnCollectText.innerText = "Ambil Landmark (Jeda 3d)";
            activeModeBadge.innerText = "Mode: Statis (Huruf/Angka)";
            statusDot.className = "w-2.5 h-2.5 rounded-full bg-teal-500 animate-pulse";
            trainModeLabel.innerText = "Statis";
        } else {
            // Tab style
            tabStatic.className = "flex-1 py-2 text-xs font-bold rounded-lg text-slate-600 hover:text-slate-800 transition";
            tabDynamic.className = "flex-1 py-2 text-xs font-bold rounded-lg bg-white text-teal-800 shadow-sm transition";
            
            // Collect button
            btnCollectText.innerText = "Rekam Gerakan (Jeda 3d)";
            activeModeBadge.innerText = "Mode: Dinamis (Kata Gerakan)";
            statusDot.className = "w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse";
            trainModeLabel.innerText = "Dinamis";
        }
        fetchStats();
    };

    tabStatic.addEventListener('click', () => setMode("STATIC"));
    tabDynamic.addEventListener('click', () => setMode("DYNAMIC"));

    // MediaPipe Hands Logic
    const canvasCtx = canvasElement ? canvasElement.getContext('2d') : null;

    function onResults(results) {
        if (!canvasElement || !canvasCtx) return;

        canvasElement.width = webcamElement.videoWidth || 640;
        canvasElement.height = webcamElement.videoHeight || 480;

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        // Render mirror video frame to canvas
        canvasCtx.translate(canvasElement.width, 0);
        canvasCtx.scale(-1, 1);
        canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);
        canvasCtx.restore();

        // 126 coordinates (2 hands * 21 points * 3 dims)
        let frameLandmarks = new Array(126).fill(0);
        handDetected = false;

        if (results.multiHandLandmarks && results.multiHandedness) {
            handDetected = true;
            canvasCtx.save();
            canvasCtx.translate(canvasElement.width, 0);
            canvasCtx.scale(-1, 1);

            for (let index = 0; index < results.multiHandLandmarks.length; index++) {
                const classification = results.multiHandedness[index];
                const isLeftHand = classification.label === 'Left';
                
                let offset = isLeftHand ? 0 : 63;
                const landmarks = results.multiHandLandmarks[index];

                for (let i = 0; i < 21; i++) {
                    frameLandmarks[offset + (i * 3)]     = landmarks[i].x;
                    frameLandmarks[offset + (i * 3) + 1] = landmarks[i].y;
                    frameLandmarks[offset + (i * 3) + 2] = landmarks[i].z;
                }

                // Draw hand mesh
                if (window.drawConnectors && window.drawLandmarks) {
                    window.drawConnectors(canvasCtx, landmarks, window.HAND_CONNECTIONS, {color: '#0D9488', lineWidth: 3});
                    window.drawLandmarks(canvasCtx, landmarks, {color: '#0D9488', lineWidth: 1, radius: 3});
                }
            }

            canvasCtx.restore();
        }

        // Simpan frame landmark terkini
        latestFrameLandmarks = frameLandmarks;

        // Logika Perekaman Dinamis
        if (isRecording) {
            recordingBuffer.push(frameLandmarks);
            recordingProgress.innerText = `Frame: ${recordingBuffer.length} / 30`;
            
            if (recordingBuffer.length >= 30) {
                isRecording = false;
                recordingOverlay.classList.add('hidden');
                submitDynamicDataset();
            }
        }
    }

    // Initialize Hands Detection
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

        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
                .then((stream) => {
                    webcamElement.srcObject = stream;
                    webcamElement.play();
                    
                    if (window.Camera) {
                        const camera = new Camera(webcamElement, {
                            onFrame: async () => {
                                await hands.send({ image: webcamElement });
                            },
                            width: 640,
                            height: 480
                        });
                        camera.start().then(() => {
                            console.log("📷 Camera engine active.");
                            if (webcamFallback) webcamFallback.classList.add('hidden');
                        });
                    }
                })
                .catch((err) => {
                    console.error("Camera access failed:", err);
                    if (webcamFallback) {
                        webcamFallback.innerHTML = `
                            <i class="fa-solid fa-triangle-exclamation text-5xl text-red-500"></i>
                            <span class="text-xs font-semibold text-red-400">Gagal mengakses kamera: ${err.message}</span>
                        `;
                        webcamFallback.classList.remove('hidden');
                    }
                });
        }
    }

    // Capture Dataset Action with 3-Second Countdown
    btnCollect.addEventListener('click', () => {
        const label = labelInput.value.trim();
        if (!label) {
            alert("Silakan masukkan Nama Label terlebih dahulu!");
            labelInput.focus();
            return;
        }

        // Verifikasi apakah API server online
        if (!apiOnline) {
            alert("Server API tidak terhubung. Pastikan API berjalan di port 8001.");
            return;
        }

        // Disable button saat proses hitung mundur dimulai
        btnCollect.disabled = true;
        btnCollect.className = "w-full py-4 bg-slate-400 text-white font-bold text-sm rounded-xl cursor-not-allowed transition flex items-center justify-center gap-2";
        
        // Mulai Hitung Mundur 3 Detik
        let secondsLeft = 3;
        countdownText.innerText = secondsLeft;
        countdownText.className = "text-8xl font-black text-teal-400 scale-75 animate-bounce";
        countdownSubtext.innerText = "Persiapkan Posisi Tangan Anda";
        countdownOverlay.classList.remove('hidden');

        const countdownInterval = setInterval(() => {
            secondsLeft--;
            if (secondsLeft > 0) {
                countdownText.innerText = secondsLeft;
            } else {
                clearInterval(countdownInterval);
                countdownOverlay.classList.add('hidden');
                
                // Mulai Pengambilan Data Berdasarkan Mode
                if (currentMode === "STATIC") {
                    captureStaticSample(label);
                } else {
                    startRecordingDynamic(label);
                }
            }
        }, 1000);
    });

    // Capture Static Sample
    const captureStaticSample = async (label) => {
        if (!handDetected) {
            alert("⚠️ Peringatan: Tangan tidak terdeteksi di kamera! Data koordinat kosong akan tetap disimpan.");
        }

        try {
            const response = await fetch(`${apiHost}/dataset/collect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mode: 'STATIC',
                    label: label,
                    landmarks: latestFrameLandmarks
                })
            });

            if (response.ok) {
                const data = await response.json();
                showTemporaryToast(`✅ Sukses! ${data.message} (${data.total_samples} sampel)`);
                fetchStats();
            } else {
                const errData = await response.json();
                alert(`Gagal menyimpan: ${errData.detail}`);
            }
        } catch (error) {
            console.error("Gagal mengirim data statis:", error);
            alert("Koneksi ke server gagal.");
        } finally {
            resetCollectButton();
        }
    };

    // Start Recording Dynamic Sequence
    const startRecordingDynamic = (label) => {
        recordingBuffer = [];
        isRecording = true;
        recordingProgress.innerText = "Frame: 0 / 30";
        recordingOverlay.classList.remove('hidden');
    };

    // Submit Dynamic Sequence after recording 30 frames
    const submitDynamicDataset = async () => {
        const label = labelInput.value.trim();
        try {
            const response = await fetch(`${apiHost}/dataset/collect`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    mode: 'DYNAMIC',
                    label: label,
                    landmarks: recordingBuffer
                })
            });

            if (response.ok) {
                const data = await response.json();
                showTemporaryToast(`✅ Sukses! ${data.message} (${data.total_samples} sampel)`);
                fetchStats();
            } else {
                const errData = await response.json();
                alert(`Gagal menyimpan: ${errData.detail}`);
            }
        } catch (error) {
            console.error("Gagal mengirim data dinamis:", error);
            alert("Koneksi ke server gagal.");
        } finally {
            resetCollectButton();
        }
    };

    const resetCollectButton = () => {
        btnCollect.disabled = false;
        btnCollect.className = "w-full py-4 bg-teal-600 hover:bg-teal-700 text-white font-bold text-sm rounded-xl shadow-lg shadow-teal-600/10 transition flex items-center justify-center gap-2";
    };

    const showTemporaryToast = (message) => {
        const toast = document.createElement('div');
        toast.className = "fixed bottom-5 right-5 z-[100] bg-slate-900 text-white px-5 py-3 rounded-xl shadow-2xl text-xs font-semibold flex items-center gap-2 border border-slate-800 animate-fade-in";
        toast.innerHTML = `<i class="fa-solid fa-circle-check text-teal-400 text-sm"></i> <span>${message}</span>`;
        document.body.appendChild(toast);
        setTimeout(() => {
            toast.remove();
        }, 3000);
    };

    // AI Training Panel Logic
    btnTrain.addEventListener('click', async () => {
        if (!apiOnline) {
            alert("Server API offline!");
            return;
        }

        if (isTrainingActive) {
            alert("Pelatihan sedang berjalan. Harap tunggu hingga selesai.");
            return;
        }

        const confirmTrain = confirm(`Apakah Anda yakin ingin memulai pelatihan ulang model ${currentMode}? Ini akan memakan waktu beberapa saat.`);
        if (!confirmTrain) return;

        btnTrain.disabled = true;
        btnTrain.className = "w-full py-3.5 bg-slate-700 text-slate-300 font-bold text-sm rounded-xl cursor-not-allowed transition flex items-center justify-center gap-2";
        isTrainingActive = true;
        trainingStatusText.innerText = "Memulai...";
        trainingStatusText.className = "text-amber-500 font-bold animate-pulse";
        logConsole.innerText = "[Mempersiapkan lingkungan pelatihan model...]\n";

        try {
            const response = await fetch(`${apiHost}/dataset/train?mode=${currentMode}`, { method: 'POST' });
            if (response.ok) {
                // Mulai melakukan polling log pelatihan
                startLogPolling();
            } else {
                const errData = await response.json();
                alert(`Gagal melatih: ${errData.detail}`);
                resetTrainButton();
            }
        } catch (error) {
            console.error("Gagal melakukan pelatihan ulang:", error);
            alert("Koneksi ke server gagal.");
            resetTrainButton();
        }
    });

    const startLogPolling = () => {
        if (trainingPollInterval) clearInterval(trainingPollInterval);
        
        trainingPollInterval = setInterval(async () => {
            try {
                const response = await fetch(`${apiHost}/dataset/train/status`);
                if (response.ok) {
                    const data = await response.json();
                    
                    // Render status
                    trainingStatusText.innerText = data.status.toUpperCase();
                    if (data.status === "training") {
                        trainingStatusText.className = "text-amber-500 font-bold animate-pulse";
                    } else if (data.status === "success") {
                        trainingStatusText.className = "text-emerald-500 font-bold";
                        clearInterval(trainingPollInterval);
                        resetTrainButton();
                        showTemporaryToast("🎉 Pelatihan selesai! Model siap digunakan.");
                    } else if (data.status === "failed") {
                        trainingStatusText.className = "text-red-500 font-bold";
                        clearInterval(trainingPollInterval);
                        resetTrainButton();
                        alert("❌ Pelatihan model gagal. Periksa konsol untuk detail kesalahan.");
                    }

                    // Render console logs
                    logConsole.innerText = data.logs.join('\n');
                    logConsole.scrollTop = logConsole.scrollHeight; // Auto scroll down
                }
            } catch (error) {
                console.error("Error polling training status:", error);
            }
        }, 1000);
    };

    const resetTrainButton = () => {
        isTrainingActive = false;
        btnTrain.disabled = false;
        btnTrain.className = "w-full py-3.5 bg-slate-900 hover:bg-slate-800 text-white font-bold text-sm rounded-xl transition flex items-center justify-center gap-2";
    };
});
