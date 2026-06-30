// RecycloBox AI App Logic

// Global states
let activeTab = 'upload';
let selectedFile = null;
let webcamStream = null;
let isModelLoaded = false;
let forceMock = true;

const classAngles = {
    'plastic': 0,
    'paper': 51,
    'glass': 103,
    'metal': 154,
    'organic': 206,
    'hazardous': 257,
    'non_recyclable': 309
};

const classIcons = {
    'plastic': 'fa-bottle-water',
    'paper': 'fa-scroll',
    'glass': 'fa-wine-glass',
    'metal': 'fa-cubes',
    'organic': 'fa-seedling',
    'hazardous': 'fa-triangle-exclamation',
    'non_recyclable': 'fa-trash'
};

const classLabels = {
    'plastic': 'Plastic',
    'paper': 'Paper',
    'glass': 'Glass',
    'metal': 'Metal',
    'organic': 'Organic',
    'hazardous': 'Hazardous Waste',
    'non_recyclable': 'Non-Recyclable'
};

// DOM Elements
const dragDropArea = document.getElementById('drag-drop-area');
const fileInput = document.getElementById('file-input');
const imagePreview = document.getElementById('image-preview');
const imagePreviewContainer = document.getElementById('image-preview-container');
const analyzeBtn = document.getElementById('analyze-btn');
const mockModeToggle = document.getElementById('mock-mode-toggle');

const webcamVideo = document.getElementById('webcam');
const webcamOverlay = document.getElementById('webcam-overlay');
const startWebcamBtn = document.getElementById('start-webcam-btn');
const captureBtn = document.getElementById('capture-btn');
const captureCanvas = document.getElementById('capture-canvas');

const backendStatus = document.getElementById('backend-status');
const modelStatus = document.getElementById('model-status');

// Init application on load
window.addEventListener('DOMContentLoaded', () => {
    checkBackendHealth();
    // Poll health status every 8 seconds
    setInterval(checkBackendHealth, 8000);
    
    setupDragAndDrop();
    setupEventListeners();
    
    // Default active state log
    logHardwareAction("System initialized. Simulator ready.");
});

// Event Listeners
function setupEventListeners() {
    mockModeToggle.addEventListener('change', (e) => {
        forceMock = e.target.checked;
        logHardwareAction(`Simulation force toggle set to: ${forceMock ? "ON" : "OFF"}`);
    });
    
    analyzeBtn.addEventListener('click', () => {
        if (selectedFile) {
            uploadAndClassify(selectedFile);
        }
    });

    startWebcamBtn.addEventListener('click', toggleWebcam);
    captureBtn.addEventListener('click', captureFrame);
}

// Health Check API
async function checkBackendHealth() {
    try {
        const response = await fetch('/health');
        if (response.ok) {
            const data = await response.json();
            updateStatusIndicator(backendStatus, 'green', 'Server: Online');
            
            isModelLoaded = data.model_loaded;
            if (isModelLoaded) {
                updateStatusIndicator(modelStatus, 'green', 'Model: Loaded');
                // Allow turning off simulator if actual model is loaded
                mockModeToggle.disabled = false;
            } else {
                updateStatusIndicator(modelStatus, 'amber', 'Model: Simulator Mode');
                // Enforce mock toggle if actual model weights are missing
                mockModeToggle.checked = true;
                mockModeToggle.disabled = true;
                forceMock = true;
            }
        } else {
            throw new Error("Server degraded response");
        }
    } catch (error) {
        updateStatusIndicator(backendStatus, 'red', 'Server: Offline');
        updateStatusIndicator(modelStatus, 'red', 'Model: Connection Failed');
        mockModeToggle.checked = true;
        mockModeToggle.disabled = true;
        forceMock = true;
    }
}

function updateStatusIndicator(badgeElement, dotColor, text) {
    const dot = badgeElement.querySelector('.status-dot');
    const textSpan = badgeElement.querySelector('.status-text');
    
    dot.className = `status-dot ${dotColor}`;
    textSpan.innerText = text;
}

// Drag & Drop Setup
function setupDragAndDrop() {
    dragDropArea.addEventListener('click', () => fileInput.click());
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    dragDropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        dragDropArea.classList.add('dragover');
    });
    
    ['dragleave', 'dragend'].forEach(type => {
        dragDropArea.addEventListener(type, () => {
            dragDropArea.classList.remove('dragover');
        });
    });
    
    dragDropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        dragDropArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
}

function handleFileSelect(file) {
    if (!file.type.startsWith('image/')) {
        alert('Invalid file format. Please upload an image.');
        return;
    }
    
    selectedFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        dragDropArea.classList.add('hidden');
        imagePreviewContainer.classList.remove('hidden');
        analyzeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    selectedFile = null;
    imagePreview.src = '';
    imagePreviewContainer.classList.add('hidden');
    dragDropArea.classList.remove('hidden');
    analyzeBtn.disabled = true;
    fileInput.value = '';
}

// Webcam Logic
async function toggleWebcam() {
    if (webcamStream) {
        // Stop webcam
        stopWebcam();
    } else {
        // Start webcam
        try {
            webcamStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'environment' }
            });
            webcamVideo.srcObject = webcamStream;
            webcamOverlay.classList.add('hidden');
            startWebcamBtn.innerHTML = '<i class="fa-solid fa-stop"></i> Stop Feed';
            startWebcamBtn.className = 'btn btn-secondary';
            captureBtn.classList.remove('hidden');
            logHardwareAction("Webcam feed activated.");
        } catch (err) {
            console.error("Camera access failed", err);
            alert("Unable to access camera. Please check camera permissions.");
            logHardwareAction("System error: Webcam access denied.");
        }
    }
}

function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
    }
    webcamVideo.srcObject = null;
    webcamStream = null;
    webcamOverlay.classList.remove('hidden');
    startWebcamBtn.innerHTML = '<i class="fa-solid fa-power-off"></i> Start Feed';
    startWebcamBtn.className = 'btn btn-secondary';
    captureBtn.classList.add('hidden');
    logHardwareAction("Webcam feed stopped.");
}

function captureFrame() {
    if (!webcamStream) return;
    
    const context = captureCanvas.getContext('2d');
    captureCanvas.width = webcamVideo.videoWidth;
    captureCanvas.height = webcamVideo.videoHeight;
    
    // Draw mirrored image onto canvas to match preview
    context.translate(captureCanvas.width, 0);
    context.scale(-1, 1);
    context.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
    context.setTransform(1, 0, 0, 1, 0, 0); // Reset transform
    
    captureCanvas.toBlob((blob) => {
        const file = new File([blob], "webcam_capture.jpg", { type: "image/jpeg" });
        uploadAndClassify(file);
    }, "image/jpeg");
}

// REST API Send Classification
async function uploadAndClassify(file) {
    setLoadingState(true);
    
    const formData = new FormData();
    formData.append('file', file);
    
    try {
        const predictUrl = `/predict?mock=${forceMock}`;
        const response = await fetch(predictUrl, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API returned status code ${response.status}`);
        }
        
        const result = await response.json();
        
        // Show result json log
        document.getElementById('api-raw-response').innerText = JSON.stringify(result, null, 2);
        
        // Update dashboard UI
        updateClassificationUI(result);
        
    } catch (err) {
        console.error("Classification request failed", err);
        alert("Failed to analyze image. Please try again.");
    } finally {
        setLoadingState(false);
    }
}

function setLoadingState(loading) {
    if (loading) {
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Segregating...';
        captureBtn.disabled = true;
        captureBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
    } else {
        analyzeBtn.disabled = selectedFile === null;
        analyzeBtn.innerHTML = '<i class="fa-solid fa-brain"></i> Run Classification';
        captureBtn.disabled = false;
        captureBtn.innerHTML = '<i class="fa-solid fa-circle-dot"></i> Analyze Waste';
    }
}

// Update Result Dashboard Visuals
function updateClassificationUI(result) {
    const predictedClass = result.class;
    const confidencePercent = Math.round(result.confidence * 100);
    
    // 1. Update Hero Box
    const heroBox = document.getElementById('prediction-hero-box');
    heroBox.classList.remove('empty');
    
    const heroIcon = document.getElementById('hero-class-icon');
    heroIcon.className = `fa-solid ${classIcons[predictedClass]} hero-icon`;
    
    // Add colored text for class
    const heroClassName = document.getElementById('hero-class-name');
    heroClassName.innerText = classLabels[predictedClass];
    heroClassName.className = `prediction-class active-result`;
    heroClassName.setAttribute('data-class', predictedClass);
    
    document.getElementById('hero-confidence-bar').style.width = `${confidencePercent}%`;
    document.getElementById('hero-confidence-val').innerText = `${confidencePercent}% Confidence`;
    
    // Color glow styling of the hero box
    heroBox.style.boxShadow = `0 4px 20px rgba(255, 255, 255, 0.02), 0 0 15px var(--color-${predictedClass.replace('_', '-')})`;
    heroBox.style.borderColor = `var(--color-${predictedClass.replace('_', '-')})`;
    
    // 2. Update Probability distribution list
    const probs = result.probabilities;
    Object.keys(probs).forEach(cls => {
        const valPercent = Math.round(probs[cls] * 100);
        const row = document.querySelector(`.dist-row[data-class="${cls}"]`);
        if (row) {
            row.querySelector('.dist-value').innerText = `${valPercent}%`;
            row.querySelector('.dist-bar').style.width = `${valPercent}%`;
        }
    });
    
    // 3. Actuate Dustbin Simulator
    actuateSimulator(predictedClass);
}

// Dustbin Digital Twin Simulation logic
function actuateSimulator(targetClass) {
    const targetAngle = classAngles[targetClass];
    
    // Log simulation steps
    logHardwareAction(`[DETECTION] Classified as ${targetClass.toUpperCase()} (Confidence: ${targetAngle}° target angle)`);
    
    // 1. Highlight sector in the circular graphic
    const sectors = document.querySelectorAll('.compartment-sector');
    sectors.forEach(s => s.classList.remove('highlighted'));
    
    const targetSector = document.getElementById(`sector-${targetClass}`);
    if (targetSector) {
        targetSector.classList.add('highlighted');
    }
    
    // 2. Rotate Center Funnel pointer to point to the sector
    // We animate the funnel to rotate to the correct angle
    const funnel = document.getElementById('physical-funnel');
    const simServoVal = document.getElementById('sim-servo-angle');
    const simCompartmentVal = document.getElementById('sim-active-compartment');
    const simFunnelAction = document.getElementById('sim-funnel-action');
    
    simFunnelAction.innerText = "Rotating Funnel...";
    simFunnelAction.style.color = "#f59e0b"; // Orange indicator
    
    // Servo angles
    simServoVal.innerText = `${targetAngle}°`;
    simCompartmentVal.innerText = classLabels[targetClass];
    
    // Change color of arrow indicator dynamically to match class color
    const arrow = document.querySelector('.arrow-indicator');
    arrow.style.borderBottomColor = `var(--color-${targetClass.replace('_', '-')})`;
    arrow.style.filter = `drop-shadow(0 0 8px var(--color-${targetClass.replace('_', '-')}))`;
    
    // Apply CSS rotation to funnel
    funnel.style.transform = `rotate(${targetAngle}deg)`;
    
    // Simulate servo delay animation finish (1.2s transition in CSS)
    setTimeout(() => {
        simFunnelAction.innerText = "Locked & Open";
        simFunnelAction.style.color = "#10b981"; // Green indicator
        logHardwareAction(`[ACTUATOR] Servo rotated to ${targetAngle}°. Drop-funnel locked to ${targetClass.toUpperCase()} compartment.`);
    }, 1200);
}

// Tabs Navigation inside Input Center
function switchTab(tabName) {
    activeTab = tabName;
    
    // Toggle active tab buttons
    document.getElementById('tab-upload-btn').classList.toggle('active', tabName === 'upload');
    document.getElementById('tab-webcam-btn').classList.toggle('active', tabName === 'webcam');
    
    // Toggle tab panels
    document.getElementById('upload-tab').classList.toggle('hidden', tabName !== 'upload');
    document.getElementById('webcam-tab').classList.toggle('hidden', tabName !== 'webcam');
    
    // Hide upload button bar if on webcam, since webcam has its own capture button
    document.getElementById('upload-action-bar').classList.toggle('hidden', tabName === 'webcam');
    
    // Manage webcam state
    if (tabName !== 'webcam' && webcamStream) {
        stopWebcam();
    }
}

// Code documentation tab navigation
function switchDocTab(docName) {
    // Toggle tabs active styles
    document.getElementById('doc-tab-python').classList.toggle('active', docName === 'python');
    document.getElementById('doc-tab-arduino').classList.toggle('active', docName === 'arduino');
    document.getElementById('doc-tab-pi').classList.toggle('active', docName === 'pi');
    
    // Toggle docs contents
    document.getElementById('doc-python').classList.toggle('hidden', docName !== 'python');
    document.getElementById('doc-arduino').classList.toggle('hidden', docName !== 'arduino');
    document.getElementById('doc-pi').classList.toggle('hidden', docName !== 'pi');
}

// Hardware Action Logger helper
function logHardwareAction(text) {
    const logBox = document.getElementById('actuation-log-text');
    const timestamp = new Date().toLocaleTimeString();
    logBox.innerText += `\n[${timestamp}] ${text}`;
    logBox.scrollTop = logBox.scrollHeight;
}
