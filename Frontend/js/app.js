// BE COOL - App UI Controller & WebSocket connection logic
document.addEventListener('DOMContentLoaded', () => {
    console.log('BE COOL App Initialized');
    
    // Inisialisasi Fitur Webcam jika elemennya ada di halaman
    initWebcamFeature();
});

// Logic highlighter aktif link di navbar
document.addEventListener("DOMContentLoaded", function() {
    // Ambil nama file dari URL (misalnya index.html, prediction_hand.html)
    const currentPage = window.location.pathname.split("/").pop() || "index.html";

    // Ambil semua link di navbar
    const navLinks = document.querySelectorAll("nav a");

    navLinks.forEach(link => {
        if (link.getAttribute("href") === currentPage) {
            // Tambahkan class aktif (warna hijau)
            link.classList.add("text-teal-600", "font-semibold");
        } else {
            // Pastikan link lain tetap normal
            link.classList.remove("text-teal-600", "font-semibold");
        }
    });
});

// Fungsi untuk Inisialisasi Webcam
function initWebcamFeature() {
    const webcamElement = document.getElementById('webcam');
    const fallbackElement = document.getElementById('webcam-fallback');
    const toggleButton = document.getElementById('btn-toggle-camera');

    if (!webcamElement) return;

    let webcamStream = null;
    let isWebcamOn = false;

    // Elemen opsional untuk di-update di tombol toggle (halaman prediction_hand)
    const iconEl = document.getElementById('toggle-camera-icon');
    const iconBgEl = document.getElementById('toggle-camera-icon-bg');
    const textEl = document.getElementById('toggle-camera-text');
    const subtextEl = document.getElementById('toggle-camera-subtext');

    async function startWebcam() {
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                },
                audio: false
            });
            webcamElement.srcObject = webcamStream;
            webcamElement.classList.remove('hidden');
            if (fallbackElement) fallbackElement.classList.add('hidden');
            isWebcamOn = true;
            updateToggleButtonUI(true);
            console.log('Webcam started successfully');
        } catch (error) {
            console.error('Error accessing webcam:', error);
            showFallback('Akses kamera ditolak atau tidak tersedia.');
            updateToggleButtonUI(false);
        }
    }

    function stopWebcam() {
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        webcamElement.srcObject = null;
        webcamElement.classList.add('hidden');
        if (fallbackElement) fallbackElement.classList.remove('hidden');
        isWebcamOn = false;
        updateToggleButtonUI(false);
        console.log('Webcam stopped');
    }

    function showFallback(message) {
        if (fallbackElement) {
            fallbackElement.classList.remove('hidden');
            const labelSpan = fallbackElement.querySelector('span');
            if (labelSpan && message) {
                labelSpan.textContent = message;
            }
        }
        webcamElement.classList.add('hidden');
    }

    function updateToggleButtonUI(isOn) {
        if (!toggleButton) return;

        if (isOn) {
            // State ketika kamera aktif (ikon kamera biasa, tombol siap mematikan)
            if (iconEl) {
                iconEl.className = 'fa-solid fa-video';
            }
            if (iconBgEl) {
                // Background ikon bulatan merah untuk tombol matikan
                iconBgEl.className = 'w-7 h-7 bg-red-600 text-white rounded-lg flex items-center justify-center text-[11px]';
            }
            if (textEl) textEl.textContent = 'Matikan Kamera';
            if (subtextEl) subtextEl.textContent = 'Hentikan akses hardware';
            
            // Update class pembungkus tombol utama di prediction_hand agar bernuansa merah/batal
            toggleButton.className = 'flex items-center gap-3 p-3 bg-red-50 hover:bg-red-100 text-red-700 rounded-xl font-bold text-xs transition text-left w-full';
            
            // Untuk halaman belajar yang tombolnya simple
            toggleButton.title = 'Matikan Kamera';
            toggleButton.className = toggleButton.className.replace('text-emerald-400', 'text-slate-200');
        } else {
            // State ketika kamera mati (ikon kamera dicoret, tombol siap menyalakan)
            if (iconEl) {
                iconEl.className = 'fa-solid fa-video-slash';
            }
            if (iconBgEl) {
                // Background ikon bulatan hijau untuk tombol aktifkan kembali
                iconBgEl.className = 'w-7 h-7 bg-emerald-600 text-white rounded-lg flex items-center justify-center text-[11px]';
            }
            if (textEl) textEl.textContent = 'Aktifkan Kamera';
            if (subtextEl) subtextEl.textContent = 'Mulai akses hardware';

            // Update class pembungkus tombol utama di prediction_hand agar bernuansa hijau/mulai
            toggleButton.className = 'flex items-center gap-3 p-3 bg-emerald-50 hover:bg-emerald-100 text-emerald-700 rounded-xl font-bold text-xs transition text-left w-full';

            // Untuk halaman belajar yang tombolnya simple
            toggleButton.title = 'Aktifkan Kamera';
        }
    }

    // Jalankan kamera saat halaman selesai dimuat
    startWebcam();

    // Event listener untuk tombol toggle
    toggleButton.addEventListener('click', (e) => {
        e.preventDefault();
        if (isWebcamOn) {
            stopWebcam();
        } else {
            startWebcam();
        }
    });
}

// js/app.js

document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. MOBILE MENU TOGGLE ---
    const mobileMenuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');

    mobileMenuButton.addEventListener('click', () => {
        mobileMenu.classList.toggle('hidden');
    });


    // --- 2. ON LOAD ANIMATIONS (Hero Section) ---
    // Select hero text and image and trigger animation
    const onLoadText = document.querySelector('.animate-on-load-text');
    const onLoadImage = document.querySelector('.animate-on-load-image');

    if (onLoadText && onLoadImage) {
        // Use setTimeout to ensure the browser has rendered the page first
        setTimeout(() => {
            onLoadText.classList.add('loaded');
            onLoadImage.classList.add('loaded');
        }, 100);
    }


    // --- 3. ON SCROLL ANIMATIONS (Intersection Observer) ---
    // Options for the Intersection Observer (defines the threshold)
    const observerOptions = {
        root: null, // Use the browser viewport as the root
        rootMargin: '0px 0px -10% 0px', // Trigger slightly earlier than the element edges
        threshold: 0.1 // Trigger when 10% of the element is visible
    };

    // Callback function for Intersection Observer
    const observerCallback = (entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                // Apply the visible class to trigger CSS animation
                entry.target.classList.add('is-visible');
                
                // If it's a social stat element, also trigger the counter animation
                if (entry.target.hasAttribute('id') && entry.target.id.startsWith('stat-')) {
                    const counterId = entry.target.id;
                    const endValue = parseInt(entry.target.getAttribute('data-end-value'));
                    animateCounter(counterId, endValue);
                }

                // Optimization: Unobserve the element after it has animated once
                observer.unobserve(entry.target);
            }
        });
    };

    // Create the Intersection Observer
    const observer = new IntersectionObserver(observerCallback, observerOptions);

    // Get all elements with the animation class and start observing
    const animatableElements = document.querySelectorAll('.animate-on-scroll');
    animatableElements.forEach(el => observer.observe(el));


    // --- 4. DYNAMIC ELEMENT: COUNTER ANIMATION ---
    const animateCounter = (id, endValue) => {
        const counterElement = document.getElementById(id);
        const duration = 1500; // Animation duration in milliseconds
        const startValue = 0;
        const stepTime = Math.abs(Math.floor(duration / endValue)); // Time per increment
        const increment = endValue > 3000 ? 50 : 1; // Faster increments for large numbers

        // Start from initial number (e.g., 0)
        let currentValue = startValue;
        counterElement.innerText = currentValue;

        // Use setInterval for the animation
        const interval = setInterval(() => {
            currentValue += increment;

            // Cap at the end value
            if (currentValue >= endValue) {
                currentValue = endValue;
                clearInterval(interval);
            }

            // Update display, add '+' if it's a 'plus' number
            counterElement.innerText = currentValue.toLocaleString() + (endValue > 1 ? '+' : '');
        }, stepTime);
    };


    // --- 5. COMMUNITY GALLERY: MODAL HANDLING (Dynamic Content) ---
    
    // Select modal-related elements
    const docModalTrigger = document.querySelector('.documentation-modal-trigger');
    const volModalTrigger = document.querySelector('.volunteer-modal-trigger');
    const docModal = document.getElementById('documentation-modal');
    const volModal = document.getElementById('volunteer-modal');
    const modalCloses = document.querySelectorAll('.modal-close');

    // Functions to open and close modals
    const openModal = (modal) => {
        modal.classList.add('modal-visible');
    };

    const closeModal = (modal) => {
        modal.classList.remove('modal-visible');
    };

    // Event listeners to open modals on triggers
    if (docModalTrigger && volModalTrigger && docModal && volModal) {
        docModalTrigger.addEventListener('click', () => openModal(docModal));
        volModalTrigger.addEventListener('click', () => openModal(volModal));
    }

    // Event listener to close modals when 'Tutup' or 'X' buttons are clicked
    modalCloses.forEach(closeBtn => {
        closeBtn.addEventListener('click', (e) => {
            e.preventDefault(); // Prevent accidental form submission
            closeModal(docModal);
            closeModal(volModal);
        });
    });

    // Event listener to close modals when clicking outside the content (on the overlay)
    window.addEventListener('click', (e) => {
        if (e.target === docModal) closeModal(docModal);
        if (e.target === volModal) closeModal(volModal);
    });
    
    // Event listener to close modals with the Escape key
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeModal(docModal);
            closeModal(volModal);
        }
    });

});

document.querySelectorAll('.card').forEach(card => {
    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      const x = ((e.clientX - rect.left) / rect.width) * 100;
      card.style.setProperty(
        'background',
        `linear-gradient(90deg, transparent, rgba(13,148,136,0.8) ${x}%, transparent)`
      );
    });
  });