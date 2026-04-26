document.addEventListener('DOMContentLoaded', function () {

    const POLL_INTERVAL = 1200;   // ms between loading-status polls

    // ── Auto-detect if models are already loading in background ──────────────
    // The server auto-triggers loading when started with HF_TOKEN in .env.
    // Poll immediately so the UI reflects the real state without user action.
    const loadCard = document.getElementById('load-card');
    if (loadCard) {
        checkAndStartPolling();
    }

    function checkAndStartPolling() {
        fetch('/loading_status')
            .then(r => r.json())
            .then(data => {
                if (data.models_loaded) {
                    // Already done — just reload
                    location.reload();
                } else if (data.progress > 0) {
                    // Server already loading — show UI and poll
                    const progressWrap = document.getElementById('progress-wrap');
                    const progressFill = document.getElementById('progress-fill');
                    const progressLbl  = document.getElementById('progress-label');
                    const btn          = document.getElementById('load-btn');
                    if (progressWrap) progressWrap.style.display = 'block';
                    if (btn) { btn.disabled = true; btn.textContent = 'Auto-loading…'; }
                    pollLoadingStatus(btn, progressFill, progressLbl);
                }
                // If progress == 0 and not loaded, models haven't started yet — wait for user click
            })
            .catch(() => {});   // silently ignore if server not ready yet
    }

    // ── File drag-and-drop ────────────────────────────────────────────────────
    const dropZone  = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file');
    const dropText  = document.getElementById('drop-text');

    if (dropZone && fileInput) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(ev =>
            dropZone.addEventListener(ev, e => { e.preventDefault(); e.stopPropagation(); })
        );

        ['dragenter', 'dragover'].forEach(ev =>
            dropZone.addEventListener(ev, () => dropZone.classList.add('drag-over'))
        );

        ['dragleave', 'drop'].forEach(ev =>
            dropZone.addEventListener(ev, () => dropZone.classList.remove('drag-over'))
        );

        dropZone.addEventListener('drop', e => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                updateDropLabel(files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) updateDropLabel(fileInput.files[0]);
        });
    }

    function updateDropLabel(file) {
        if (!dropText) return;
        const mb = (file.size / 1024 / 1024).toFixed(1);
        dropText.innerHTML =
            `<strong style="color:#11998e;">${file.name}</strong><br>
             <small>${mb} MB &bull; ready to analyze</small>`;
    }

    // ── Model loading form ───────────────────────────────────────────────────
    const loadForm = document.getElementById('load-models-form');
    if (loadForm) {
        loadForm.addEventListener('submit', handleModelLoad);
    }

    function handleModelLoad(e) {
        e.preventDefault();

        const btn          = document.getElementById('load-btn');
        const progressWrap = document.getElementById('progress-wrap');
        const progressFill = document.getElementById('progress-fill');
        const progressLbl  = document.getElementById('progress-label');

        // Disable button and show progress
        btn.disabled = true;
        btn.textContent = 'Starting...';
        progressWrap.style.display = 'block';
        progressFill.style.width   = '5%';
        progressLbl.textContent    = 'Sending request to server...';

        const formData = new FormData(loadForm);

        fetch('/load_models', { method: 'POST', body: formData })
            .then(async res => {
                if (!res.ok) {
                    const body = await res.text();
                    throw new Error(`Server error ${res.status}: ${body}`);
                }
                return res.json();
            })
            .then(data => {
                if (!data.success) throw new Error(data.error || 'Unknown error');
                pollLoadingStatus(btn, progressFill, progressLbl);
            })
            .catch(err => {
                showAlert('Model loading failed: ' + err.message, 'danger');
                btn.disabled    = false;
                btn.textContent = 'Load Models';
                progressWrap.style.display = 'none';
            });
    }

    function pollLoadingStatus(btn, fillEl, labelEl) {
        fetch('/loading_status')
            .then(r => r.json())
            .then(data => {
                const pct = Math.max(data.progress || 0, 5);
                fillEl.style.width   = pct + '%';
                labelEl.textContent  = data.status || '...';
                btn.textContent      = `Loading… ${pct}%`;

                if (data.models_loaded) {
                    fillEl.style.width  = '100%';
                    labelEl.textContent = 'All models ready! Reloading page...';
                    btn.textContent     = 'Models Loaded!';
                    setTimeout(() => location.reload(), 1500);

                } else if (typeof data.status === 'string' && data.status.startsWith('Error')) {
                    showAlert('Loading failed: ' + data.status, 'danger');
                    btn.disabled    = false;
                    btn.textContent = 'Load Models';
                    document.getElementById('progress-wrap').style.display = 'none';

                } else {
                    setTimeout(() => pollLoadingStatus(btn, fillEl, labelEl), POLL_INTERVAL);
                }
            })
            .catch(() => {
                // Retry silently on network hiccup
                setTimeout(() => pollLoadingStatus(btn, fillEl, labelEl), POLL_INTERVAL * 2);
            });
    }

    // ── Upload / analysis form ────────────────────────────────────────────────
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleUpload);
    }

    function handleUpload(e) {
        const btn     = document.getElementById('upload-btn');
        const wrap    = document.getElementById('upload-progress-wrap');
        const lbl     = document.getElementById('upload-progress-label');

        btn.disabled    = true;
        btn.textContent = 'Uploading...';
        wrap.style.display = 'block';

        // Cycle through stage labels so the user sees progress
        const stages = [
            'Uploading audio file...',
            'Preprocessing & normalising audio...',
            'Running Voice Activity Detection (PyAnnote)...',
            'Transcribing with Canary-Qwen 2.5B — this may take a minute...',
            'Running speaker diarization (PyAnnote 3.1)...',
            'Combining transcripts and speakers...',
            'Finalising results...'
        ];
        let si = 0;
        lbl.textContent = stages[0];

        const stageTimer = setInterval(() => {
            si = Math.min(si + 1, stages.length - 1);
            lbl.textContent = stages[si];
        }, 7000);

        // Let the native form submit proceed (Flask returns rendered HTML)
        // We just need to clean up if the user somehow navigates away
        window.addEventListener('beforeunload', () => clearInterval(stageTimer));
    }

    // ── Alert helper ─────────────────────────────────────────────────────────
    function showAlert(msg, type = 'danger') {
        const div = document.createElement('div');
        div.className = `alert alert-${type}`;
        div.textContent = msg;
        const container = document.querySelector('.container');
        container.insertBefore(div, container.children[1]);   // after header
        setTimeout(() => div.remove(), 7000);
    }

    // ── Auto-scroll timeline to first segment ────────────────────────────────
    const scroll = document.querySelector('.timeline-scroll');
    if (scroll) scroll.scrollTop = 0;

    // ── Copy full transcript to clipboard ────────────────────────────────────
    const transcriptEl = document.getElementById('full-transcript');
    if (transcriptEl) {
        transcriptEl.title = 'Click to copy transcript';
        transcriptEl.style.cursor = 'pointer';
        transcriptEl.addEventListener('click', () => {
            const text = transcriptEl.innerText;
            navigator.clipboard.writeText(text).then(() => {
                showAlert('Transcript copied to clipboard!', 'success');
            });
        });
    }
});
