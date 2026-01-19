document.addEventListener('DOMContentLoaded', function() {
    // Global timeout settings
    const AJAX_TIMEOUT = 300000; // 5 minutes
    const POLLING_INTERVAL = 1000; // 1 second
    
    // File input handling
    const fileInput = document.getElementById('file');
    const fileLabel = document.querySelector('.file-input-label');
    
    if (fileLabel) {
        setupFileUpload();
    }
    
    // Model loading with progress
    const loadModelsForm = document.getElementById('load-models-form');
    if (loadModelsForm) {
        loadModelsForm.addEventListener('submit', handleModelLoading);
    }
    
    // File upload with progress
    const uploadForm = document.getElementById('upload-form');
    if (uploadForm) {
        uploadForm.addEventListener('submit', handleFileUpload);
    }
    
    function setupFileUpload() {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            fileLabel.addEventListener(eventName, preventDefaults, false);
        });
        
        ['dragenter', 'dragover'].forEach(eventName => {
            fileLabel.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            fileLabel.addEventListener(eventName, unhighlight, false);
        });
        
        fileLabel.addEventListener('drop', handleDrop, false);
        fileInput.addEventListener('change', updateFileName);
    }
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        fileLabel.classList.add('highlight');
    }
    
    function unhighlight(e) {
        fileLabel.classList.remove('highlight');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            fileInput.files = files;
            updateFileName();
        }
    }
    
    function updateFileName() {
        const fileText = document.querySelector('.file-input-text');
        if (fileInput.files.length > 0) {
            fileText.textContent = `Selected: ${fileInput.files[0].name}`;
        }
    }
    
    function handleModelLoading(e) {
        e.preventDefault();
        
        const submitBtn = this.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;
        
        submitBtn.textContent = '🔄 Starting Model Loading...';
        submitBtn.disabled = true;
        
        const formData = new FormData(this);
        
        console.log('Sending request to /load_models...');
        
        // Make request with better error handling
        fetch('/load_models', {
            method: 'POST',
            body: formData
        })
        .then(async response => {
            console.log('Response status:', response.status);
            
            if (!response.ok) {
                const text = await response.text();
                console.error('Response error:', text);
                throw new Error(`Server error: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Response data:', data);
            
            if (data.success) {
                // Start polling for progress
                startProgressPolling(submitBtn, originalText);
            } else {
                throw new Error(data.error || 'Unknown error');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showError('Model loading failed: ' + error.message);
            submitBtn.textContent = originalText;
            submitBtn.disabled = false;
        });
    }
    
    function startProgressPolling(submitBtn, originalText) {
        const pollProgress = () => {
            fetch('/loading_status')
            .then(response => response.json())
            .then(data => {
                submitBtn.textContent = `🔄 ${data.status} (${data.progress}%)`;
                
                if (data.models_loaded) {
                    submitBtn.textContent = '✅ Models Loaded!';
                    setTimeout(() => {
                        location.reload();
                    }, 2000);
                } else if (data.status.startsWith('Error')) {
                    showError('Model loading failed: ' + data.status);
                    submitBtn.textContent = originalText;
                    submitBtn.disabled = false;
                } else {
                    // Continue polling
                    setTimeout(pollProgress, POLLING_INTERVAL);
                }
            })
            .catch(error => {
                console.error('Polling error:', error);
                setTimeout(pollProgress, POLLING_INTERVAL * 2); // Retry with longer interval
            });
        };
        
        pollProgress();
    }
    
    function handleFileUpload(e) {
        const submitBtn = this.querySelector('button[type="submit"]');
        const originalText = submitBtn.textContent;
        
        submitBtn.textContent = '🔄 Analyzing Audio...';
        submitBtn.disabled = true;
        
        // Show progress updates
        let step = 0;
        const steps = ['🎵 Transcribing...', '👥 Identifying Speakers...', '🔍 Detecting Silence...'];
        
        const progressInterval = setInterval(() => {
            if (step < steps.length) {
                submitBtn.textContent = steps[step];
                step++;
            }
        }, 3000);
        
        // Cleanup on form submission complete
        this.addEventListener('submit', () => {
            clearInterval(progressInterval);
        }, { once: true });
    }
    
    function showError(message) {
        // Create error alert
        const alert = document.createElement('div');
        alert.className = 'alert alert-danger';
        alert.textContent = message;
        
        // Insert at top of container
        const container = document.querySelector('.container');
        container.insertBefore(alert, container.firstChild);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            alert.remove();
        }, 5000);
    }
});
