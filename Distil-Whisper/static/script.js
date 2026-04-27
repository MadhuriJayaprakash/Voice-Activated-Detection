/* ── Distil-Whisper Speech Analyzer — client scripts ── */

// ── Model loading ─────────────────────────────────────────────────────────────

function loadModels() {
    const btn         = document.getElementById('load-btn');
    const section     = document.getElementById('loading-section');
    const progressBar = document.getElementById('progress-bar');
    const statusText  = document.getElementById('loading-status-text');
    const modelKey    = document.getElementById('distil_model')?.value || 'distil-small.en';

    btn.disabled    = true;
    btn.textContent = '⏳ Loading...';
    section.style.display = 'block';

    fetch('/load_models', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ distil_model: modelKey })
    })
    .then(r => r.json())
    .then(data => {
        if (data.success) {
            pollLoadingStatus(progressBar, statusText, btn);
        } else {
            statusText.textContent = '❌ Error: ' + data.error;
            btn.disabled    = false;
            btn.textContent = '🚀 Load Models';
        }
    })
    .catch(err => {
        statusText.textContent = '❌ Network error: ' + err.message;
        btn.disabled    = false;
        btn.textContent = '🚀 Load Models';
    });
}

function pollLoadingStatus(progressBar, statusText, btn) {
    let lastServerProgress = 0;
    let animatedProgress   = 0;
    let animFrame          = null;

    // Smoothly animate the bar toward a target value
    function animateTo(target) {
        if (animFrame) cancelAnimationFrame(animFrame);
        function step() {
            if (animatedProgress < target) {
                animatedProgress = Math.min(animatedProgress + 0.15, target);
                progressBar.style.width = animatedProgress.toFixed(1) + '%';
                animFrame = requestAnimationFrame(step);
            }
        }
        step();
    }

    // While download is happening (10–29%) slowly creep the bar forward
    // to show the UI is alive, without ever reaching 30 (server sets that)
    let downloadCreep = 10;
    let creepInterval = null;
    function startCreep() {
        if (creepInterval) return;
        creepInterval = setInterval(() => {
            if (downloadCreep < 28) {
                downloadCreep += 0.3;
                if (animatedProgress < downloadCreep) {
                    animatedProgress = downloadCreep;
                    progressBar.style.width = animatedProgress.toFixed(1) + '%';
                }
            }
        }, 500);
    }
    function stopCreep() {
        if (creepInterval) { clearInterval(creepInterval); creepInterval = null; }
    }

    const interval = setInterval(() => {
        fetch('/loading_status')
            .then(r => r.json())
            .then(data => {
                const sp = data.progress;

                if (sp <= 10) {
                    // Downloading phase — creep bar slowly
                    startCreep();
                    statusText.textContent = data.status +
                        ' ⏬ (first-time download in progress, please wait...)';
                } else {
                    stopCreep();
                    lastServerProgress = sp;
                    animateTo(sp);
                    statusText.textContent = data.status;
                }

                if (data.models_loaded) {
                    clearInterval(interval);
                    stopCreep();
                    animateTo(100);
                    statusText.textContent = '✅ Models loaded — reloading page...';
                    setTimeout(() => location.reload(), 800);
                } else if (data.status.startsWith('Error')) {
                    clearInterval(interval);
                    stopCreep();
                    btn.disabled    = false;
                    btn.textContent = '🚀 Load Models';
                }
            })
            .catch(() => { /* transient network error — keep polling */ });
    }, 1000);
}

// ── Upload form ───────────────────────────────────────────────────────────────

function updateFileName(input) {
    const display = document.getElementById('file-name-display');
    if (display && input.files.length > 0) {
        const name = input.files[0].name;
        const size = (input.files[0].size / 1024 / 1024).toFixed(1);
        display.textContent = `${name}  (${size} MB)`;
    }
}

// Show spinner when form is submitted
const uploadForm = document.getElementById('upload-form');
if (uploadForm) {
    uploadForm.addEventListener('submit', () => {
        const analyzeBtn = document.getElementById('analyze-btn');
        const analyzingMsg = document.getElementById('analyzing-msg');
        if (analyzeBtn)   analyzeBtn.style.display    = 'none';
        if (analyzingMsg) analyzingMsg.style.display  = 'flex';
    });
}

// ── Drag & Drop ───────────────────────────────────────────────────────────────

const dropZone = document.getElementById('drop-zone');
if (dropZone) {
    ['dragenter', 'dragover'].forEach(ev => {
        dropZone.addEventListener(ev, e => {
            e.preventDefault();
            dropZone.style.borderColor = '#6366f1';
        });
    });

    ['dragleave', 'drop'].forEach(ev => {
        dropZone.addEventListener(ev, e => {
            e.preventDefault();
            dropZone.style.borderColor = '';
        });
    });

    dropZone.addEventListener('drop', e => {
        const fileInput = dropZone.querySelector('input[type="file"]');
        if (fileInput && e.dataTransfer.files.length > 0) {
            fileInput.files = e.dataTransfer.files;
            updateFileName(fileInput);
        }
    });
}
