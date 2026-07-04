/**
 * JALSAFAYOO AI — Dashboard Client Application
 * Socket.IO real-time updates, API integration, charts, and controls.
 */

(function () {
    'use strict';

    // =========================================================================
    // Constants & Source Labels
    // =========================================================================

    const SOURCE_LABELS = {
        none: 'No Source',
        ip_webcam: 'Live IP Webcam',
        demo_video: 'Demo Video',
        upload: 'Uploaded Video',
    };

    const CHART_HISTORY = 60;
    const SETTINGS_DEBOUNCE_MS = 300;

    // =========================================================================
    // DOM References
    // =========================================================================

    const DOM = {
        loadingScreen: document.getElementById('loading-screen'),
        loadingBar: document.getElementById('loading-bar-fill'),
        loadingStatus: document.getElementById('loading-status'),
        connectionDot: document.getElementById('connection-dot'),
        connectionLabel: document.getElementById('connection-label'),
        navbarDate: document.getElementById('navbar-date'),
        navbarTime: document.getElementById('navbar-time'),
        videoStream: document.getElementById('video-stream'),
        videoOverlay: document.getElementById('video-overlay'),
        videoContainer: document.getElementById('video-container'),
        videoSkeleton: document.getElementById('video-skeleton'),
        liveBadge: document.getElementById('live-badge'),
        statFps: document.getElementById('stat-fps'),
        statTotal: document.getElementById('stat-total'),
        statCurrent: document.getElementById('stat-current'),
        statModel: document.getElementById('stat-model'),
        statCamera: document.getElementById('stat-camera'),
        statSource: document.getElementById('stat-source'),
        statConfidence: document.getElementById('stat-confidence'),
        statInference: document.getElementById('stat-inference'),
        tableBody: document.getElementById('detection-table-body'),
        tableCount: document.getElementById('table-count'),
        detectionLog: document.getElementById('detection-log'),
        toastContainer: document.getElementById('toast-container'),
        webcamUrl: document.getElementById('webcam-url'),
        demoSelect: document.getElementById('demo-video-select'),
        uploadZone: document.getElementById('upload-zone'),
        uploadInput: document.getElementById('video-upload-input'),
        uploadProgress: document.getElementById('upload-progress'),
        uploadProgressFill: document.getElementById('upload-progress-fill'),
        uploadProgressText: document.getElementById('upload-progress-text'),
        confidenceSlider: document.getElementById('confidence-slider'),
        confidenceValue: document.getElementById('confidence-value'),
        iouSlider: document.getElementById('iou-slider'),
        iouValue: document.getElementById('iou-value'),
        themeCheckbox: document.getElementById('theme-checkbox'),
    };

    // =========================================================================
    // Application State
    // =========================================================================

    const state = {
        connected: false,
        isRunning: false,
        isPaused: false,
        lastTotal: 0,
        settingsTimer: null,
    };

    let socket = null;
    let fpsChart = null;
    let detectionsChart = null;

    // =========================================================================
    // Utility Functions
    // =========================================================================

    function $(selector) {
        return document.querySelector(selector);
    }

    function $$(selector) {
        return document.querySelectorAll(selector);
    }

    async function api(url, options = {}) {
        const defaults = {
            headers: { 'Content-Type': 'application/json' },
        };

        const config = { ...defaults, ...options };

        if (config.body && typeof config.body === 'object' && !(config.body instanceof FormData)) {
            config.body = JSON.stringify(config.body);
        }

        if (config.body instanceof FormData) {
            delete config.headers['Content-Type'];
        }

        const response = await fetch(url, config);
        const data = await response.json().catch(() => ({}));

        if (!response.ok) {
            throw new Error(data.error || `Request failed (${response.status})`);
        }

        return data;
    }

    function formatConfidence(value) {
        return parseFloat(value).toFixed(2);
    }

    function getConfidenceClass(confidence) {
        if (confidence >= 0.7) return 'conf-high';
        if (confidence >= 0.5) return 'conf-mid';
        return 'conf-low';
    }

    function bumpElement(element) {
        if (!element) return;
        element.classList.remove('bump');
        void element.offsetWidth;
        element.classList.add('bump');
    }

    function sourceLabel(key) {
        return SOURCE_LABELS[key] || key;
    }

    // =========================================================================
    // Toast Notifications
    // =========================================================================

    function showToast(message, type = 'info', duration = 4000) {
        const icons = {
            success: 'bi-check-circle-fill',
            error: 'bi-x-circle-fill',
            info: 'bi-info-circle-fill',
            warning: 'bi-exclamation-triangle-fill',
        };

        const toast = document.createElement('div');
        toast.className = `toast-item toast-${type}`;
        toast.innerHTML = `
            <i class="bi ${icons[type] || icons.info} toast-icon"></i>
            <span>${message}</span>
        `;

        DOM.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // =========================================================================
    // Loading Screen
    // =========================================================================

    function runLoadingSequence() {
        const steps = [
            { progress: 20, text: 'Loading UI components…' },
            { progress: 45, text: 'Connecting to AI engine…' },
            { progress: 70, text: 'Initializing detection pipeline…' },
            { progress: 90, text: 'Preparing dashboard…' },
            { progress: 100, text: 'Ready' },
        ];

        let index = 0;

        const interval = setInterval(() => {
            if (index >= steps.length) {
                clearInterval(interval);
                setTimeout(hideLoadingScreen, 400);
                return;
            }

            const step = steps[index];
            DOM.loadingBar.style.width = `${step.progress}%`;
            DOM.loadingStatus.textContent = step.text;
            index += 1;
        }, 450);
    }

    function hideLoadingScreen() {
        DOM.loadingScreen.classList.add('fade-out');
    }

    // =========================================================================
    // Clock
    // =========================================================================

    function updateClock() {
        const now = new Date();

        DOM.navbarDate.textContent = now.toLocaleDateString('en-US', {
            weekday: 'long',
            year: 'numeric',
            month: 'long',
            day: 'numeric',
        });

        DOM.navbarTime.textContent = now.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: false,
        });
    }

    // =========================================================================
    // Video Feed UI
    // =========================================================================

    function showVideoFeed() {
        DOM.videoOverlay.classList.add('hidden');
        DOM.videoStream.classList.add('active');
        DOM.videoSkeleton.classList.remove('visible');
        DOM.liveBadge.classList.add('active');
    }

    function hideVideoFeed() {
        DOM.videoOverlay.classList.remove('hidden');
        DOM.videoStream.classList.remove('active');
        DOM.liveBadge.classList.remove('active');
    }

    function refreshVideoStream() {
        const base = '/video_feed';
        DOM.videoStream.src = `${base}?t=${Date.now()}`;
    }

    // =========================================================================
    // Statistics Update
    // =========================================================================

    function updateStats(data) {
        if (data.fps !== undefined) {
            DOM.statFps.textContent = parseFloat(data.fps).toFixed(1);
        }

        if (data.total_floaters !== undefined) {
            DOM.statTotal.textContent = data.total_floaters;
            if (data.total_floaters !== state.lastTotal) {
                bumpElement(DOM.statTotal);
                state.lastTotal = data.total_floaters;
            }
        }

        if (data.current_detections !== undefined) {
            DOM.statCurrent.textContent = data.current_detections;
            DOM.tableCount.textContent = `${data.current_detections} active`;
            bumpElement(DOM.statCurrent);
        }

        if (data.model_status) {
            DOM.statModel.textContent = data.model_status;
        }

        if (data.camera_status) {
            DOM.statCamera.textContent = data.camera_status;
        }

        if (data.input_source) {
            DOM.statSource.textContent = sourceLabel(data.input_source);
        }

        if (data.confidence_threshold !== undefined) {
            DOM.statConfidence.textContent = formatConfidence(data.confidence_threshold);
        }

        if (data.inference_time_ms !== undefined) {
            DOM.statInference.textContent = `${data.inference_time_ms} ms`;
        }

        if (data.current_time) {
            DOM.navbarTime.textContent = data.current_time;
        }

        if (data.current_date) {
            DOM.navbarDate.textContent = data.current_date;
        }

        updateCharts(data.fps, data.current_detections);
    }

    function updateStatus(data) {
        state.isRunning = data.is_running;
        state.isPaused = data.is_paused;

        if (data.is_running) {
            showVideoFeed();
        } else {
            hideVideoFeed();
        }

        if (data.camera_status) {
            DOM.statCamera.textContent = data.camera_status;
        }

        if (data.input_source) {
            DOM.statSource.textContent = sourceLabel(data.input_source);
        }

        if (data.model_status) {
            DOM.statModel.textContent = data.model_status;
        }
    }

    // =========================================================================
    // Detection Table
    // =========================================================================

    function renderDetectionTable(detections) {
        if (!detections || detections.length === 0) {
            DOM.tableBody.innerHTML = `
                <tr class="empty-row">
                    <td colspan="8">No detections in current frame</td>
                </tr>
            `;
            return;
        }

        DOM.tableBody.innerHTML = detections.map((det) => `
            <tr class="new-row">
                <td>#${det.detection_id}</td>
                <td>${det.center_x}</td>
                <td>${det.center_y}</td>
                <td>${det.width}</td>
                <td>${det.height}</td>
                <td>${det.area.toLocaleString()}</td>
                <td class="${getConfidenceClass(det.confidence)}">${formatConfidence(det.confidence)}</td>
                <td>${det.timestamp}</td>
            </tr>
        `).join('');
    }

    // =========================================================================
    // Detection Log
    // =========================================================================

    function renderDetectionLog(logEntries) {
        const emptyEl = DOM.detectionLog.querySelector('.log-empty');

        if (!logEntries || logEntries.length === 0) {
            if (emptyEl) emptyEl.classList.remove('hidden');
            DOM.detectionLog.querySelectorAll('.log-entry').forEach((el) => el.remove());
            return;
        }

        if (emptyEl) emptyEl.classList.add('hidden');

        DOM.detectionLog.innerHTML = logEntries.map((entry) => `
            <div class="log-entry">
                <span class="log-entry-id">#${entry.detection_id}</span>
                <span class="log-entry-info">
                    <strong>Floater</strong> at (${entry.center_x}, ${entry.center_y})
                    — ${entry.width}×${entry.height}px, area ${entry.area.toLocaleString()}
                </span>
                <span class="log-entry-conf ${getConfidenceClass(entry.confidence)}">
                    ${formatConfidence(entry.confidence)}
                </span>
                <span class="log-entry-time">${entry.timestamp}</span>
            </div>
        `).join('');
    }

    function handleDetectionsUpdate(data) {
        renderDetectionTable(data.current || []);
        renderDetectionLog(data.log || []);
    }

    // =========================================================================
    // Chart.js — Live Charts
    // =========================================================================

    function getChartColors() {
        const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
        return {
            primary: '#4DA3FF',
            secondary: '#00D4B8',
            grid: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)',
            text: isDark ? '#94A3B8' : '#64748B',
        };
    }

    function createChartOptions(label) {
        const colors = getChartColors();
        return {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 300 },
            plugins: {
                legend: { display: false },
                title: {
                    display: true,
                    text: label,
                    color: colors.text,
                    font: { size: 11, weight: '600', family: 'Inter' },
                    padding: { bottom: 8 },
                },
            },
            scales: {
                x: {
                    display: false,
                    grid: { display: false },
                },
                y: {
                    beginAtZero: true,
                    grid: { color: colors.grid },
                    ticks: {
                        color: colors.text,
                        font: { size: 10 },
                        maxTicksLimit: 4,
                    },
                },
            },
            elements: {
                line: { tension: 0.4, borderWidth: 2 },
                point: { radius: 0, hoverRadius: 3 },
            },
        };
    }

    function destroyCharts() {
        if (fpsChart) {
            fpsChart.destroy();
            fpsChart = null;
        }
        if (detectionsChart) {
            detectionsChart.destroy();
            detectionsChart = null;
        }
    }

    function initCharts() {
        if (typeof Chart === 'undefined') {
            console.warn('Chart.js not loaded — charts disabled');
            return;
        }

        const colors = getChartColors();
        const labels = Array(CHART_HISTORY).fill('');

        const fpsCtx = document.getElementById('fps-chart');
        if (fpsCtx) {
            fpsChart = new Chart(fpsCtx, {
                type: 'line',
                data: {
                    labels,
                    datasets: [{
                        label: 'FPS',
                        data: Array(CHART_HISTORY).fill(0),
                        borderColor: colors.primary,
                        backgroundColor: 'rgba(77, 163, 255, 0.1)',
                        fill: true,
                    }],
                },
                options: createChartOptions('FPS Over Time'),
            });
        }

        const detCtx = document.getElementById('detections-chart');
        if (detCtx) {
            detectionsChart = new Chart(detCtx, {
                type: 'line',
                data: {
                    labels,
                    datasets: [{
                        label: 'Detections',
                        data: Array(CHART_HISTORY).fill(0),
                        borderColor: colors.secondary,
                        backgroundColor: 'rgba(0, 212, 184, 0.1)',
                        fill: true,
                    }],
                },
                options: createChartOptions('Detections Per Frame'),
            });
        }
    }

    function updateCharts(fps, detections) {
        if (fpsChart && fps !== undefined) {
            const data = fpsChart.data.datasets[0].data;
            data.push(parseFloat(fps));
            if (data.length > CHART_HISTORY) data.shift();
            fpsChart.update('none');
        }

        if (detectionsChart && detections !== undefined) {
            const data = detectionsChart.data.datasets[0].data;
            data.push(parseInt(detections, 10));
            if (data.length > CHART_HISTORY) data.shift();
            detectionsChart.update('none');
        }
    }

    // =========================================================================
    // Socket.IO
    // =========================================================================

    function initSocket() {
        if (typeof io === 'undefined') {
            DOM.connectionDot.classList.add('disconnected');
            DOM.connectionLabel.textContent = 'Socket.IO failed to load';
            showToast('Socket.IO library missing. Restart the server and hard-refresh (Ctrl+F5).', 'error', 8000);
            return;
        }

        socket = io({ transports: ['websocket', 'polling'] });

        socket.on('connect', () => {
            state.connected = true;
            DOM.connectionDot.classList.add('connected');
            DOM.connectionDot.classList.remove('disconnected');
            DOM.connectionLabel.textContent = 'Connected';
            showToast('Connected to JALSAFAYOO AI', 'success');
            socket.emit('request_status');
        });

        socket.on('disconnect', () => {
            state.connected = false;
            DOM.connectionDot.classList.remove('connected');
            DOM.connectionDot.classList.add('disconnected');
            DOM.connectionLabel.textContent = 'Disconnected';
            showToast('Connection lost', 'error');
        });

        socket.on('connected', (data) => {
            showToast(data.message || 'Dashboard ready', 'info', 3000);
        });

        socket.on('stats_update', updateStats);
        socket.on('status_update', updateStatus);
        socket.on('detections_update', handleDetectionsUpdate);

        socket.on('output_ready', (data) => {
            showToast(`Processed video saved: ${data.filename}`, 'success', 5000);
        });
    }

    // =========================================================================
    // API — Input Sources
    // =========================================================================

    async function loadDemoVideos() {
        try {
            const data = await api('/api/videos');
            const videos = data.videos || [];

            if (videos.length === 0) {
                DOM.demoSelect.innerHTML = '<option value="">No demo videos found in /videos</option>';
                return;
            }

            DOM.demoSelect.innerHTML = [
                '<option value="">Select a demo video…</option>',
                ...videos.map((v) =>
                    `<option value="${v.name}">${v.name} (${v.size_mb} MB)</option>`
                ),
            ].join('');
        } catch (err) {
            DOM.demoSelect.innerHTML = '<option value="">Failed to load videos</option>';
            showToast(err.message, 'error');
        }
    }

    async function connectWebcam() {
        const url = DOM.webcamUrl.value.trim();
        if (!url) {
            showToast('Please enter a webcam URL', 'warning');
            return;
        }

        try {
            DOM.videoSkeleton.classList.add('visible');
            await api('/api/connect/webcam', {
                method: 'POST',
                body: { url },
            });
            refreshVideoStream();
            showVideoFeed();
            showToast('Webcam connected — detection started', 'success');
        } catch (err) {
            DOM.videoSkeleton.classList.remove('visible');
            showToast(err.message, 'error');
        }
    }

    async function playDemoVideo() {
        const filename = DOM.demoSelect.value;
        if (!filename) {
            showToast('Please select a demo video', 'warning');
            return;
        }

        try {
            DOM.videoSkeleton.classList.add('visible');
            await api('/api/connect/demo', {
                method: 'POST',
                body: { filename },
            });
            refreshVideoStream();
            showVideoFeed();
            showToast(`Playing: ${filename}`, 'success');
        } catch (err) {
            DOM.videoSkeleton.classList.remove('visible');
            showToast(err.message, 'error');
        }
    }

    async function uploadVideo(file) {
        if (!file) return;

        const allowed = ['mp4', 'avi', 'mov', 'mkv'];
        const ext = file.name.split('.').pop().toLowerCase();

        if (!allowed.includes(ext)) {
            showToast('Unsupported format. Use MP4, AVI, MOV, or MKV.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('video', file);

        DOM.uploadZone.classList.add('hidden');
        DOM.uploadProgress.classList.remove('hidden');
        DOM.uploadProgressFill.style.width = '30%';
        DOM.uploadProgressText.textContent = `Uploading ${file.name}…`;

        try {
            DOM.uploadProgressFill.style.width = '60%';

            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Upload failed');
            }

            DOM.uploadProgressFill.style.width = '100%';
            DOM.uploadProgressText.textContent = 'Processing…';

            refreshVideoStream();
            showVideoFeed();
            showToast(`Upload complete — detecting: ${data.filename}`, 'success');

            setTimeout(resetUploadUI, 1500);
        } catch (err) {
            showToast(err.message, 'error');
            resetUploadUI();
        }
    }

    function resetUploadUI() {
        DOM.uploadZone.classList.remove('hidden');
        DOM.uploadProgress.classList.add('hidden');
        DOM.uploadProgressFill.style.width = '0%';
        DOM.uploadInput.value = '';
    }

    // =========================================================================
    // API — Playback Controls
    // =========================================================================

    async function controlAction(action) {
        try {
            const data = await api(`/api/control/${action}`, { method: 'POST' });

            if (action === 'stop') {
                hideVideoFeed();
            } else if (['play', 'resume', 'restart'].includes(action)) {
                refreshVideoStream();
                showVideoFeed();
            }

            const messages = {
                play: 'Detection started',
                pause: 'Detection paused',
                resume: 'Detection resumed',
                restart: 'Video restarted',
                stop: 'Detection stopped',
            };

            showToast(messages[action] || `Action: ${action}`, 'info', 2500);
            return data;
        } catch (err) {
            showToast(err.message, 'error');
        }
    }

    async function captureSnapshot() {
        try {
            const data = await api('/api/snapshot', { method: 'POST' });
            showToast('Snapshot captured', 'success');

            if (data.download_url) {
                window.open(data.download_url, '_blank');
            }
        } catch (err) {
            showToast(err.message, 'error');
        }
    }

    async function downloadLatestOutput() {
        try {
            window.open('/api/download/latest', '_blank');
            showToast('Downloading processed video…', 'info');
        } catch (err) {
            showToast(err.message, 'error');
        }
    }

    function toggleFullscreen() {
        const container = DOM.videoContainer;

        if (!document.fullscreenElement) {
            container.requestFullscreen?.().catch(() => {
                showToast('Fullscreen not supported', 'warning');
            });
        } else {
            document.exitFullscreen?.();
        }
    }

    // =========================================================================
    // API — Settings
    // =========================================================================

    async function applySettings() {
        const confidence = parseFloat(DOM.confidenceSlider.value);
        const iou = parseFloat(DOM.iouSlider.value);

        DOM.confidenceValue.textContent = formatConfidence(confidence);
        DOM.iouValue.textContent = formatConfidence(iou);

        try {
            await api('/api/settings', {
                method: 'POST',
                body: { confidence, iou },
            });
        } catch (err) {
            showToast(err.message, 'error');
        }
    }

    function debounceSettings() {
        clearTimeout(state.settingsTimer);
        state.settingsTimer = setTimeout(applySettings, SETTINGS_DEBOUNCE_MS);

        DOM.confidenceValue.textContent = formatConfidence(DOM.confidenceSlider.value);
        DOM.iouValue.textContent = formatConfidence(DOM.iouSlider.value);
    }

    async function resetSettings() {
        try {
            const data = await api('/api/settings/reset', { method: 'POST' });

            DOM.confidenceSlider.value = data.confidence;
            DOM.iouSlider.value = data.iou;
            DOM.confidenceValue.textContent = formatConfidence(data.confidence);
            DOM.iouValue.textContent = formatConfidence(data.iou);
            DOM.statConfidence.textContent = formatConfidence(data.confidence);

            showToast('Settings reset to defaults', 'info');
        } catch (err) {
            showToast(err.message, 'error');
        }
    }

    async function loadSettings() {
        try {
            const data = await api('/api/settings');
            DOM.confidenceSlider.value = data.confidence;
            DOM.iouSlider.value = data.iou;
            DOM.confidenceValue.textContent = formatConfidence(data.confidence);
            DOM.iouValue.textContent = formatConfidence(data.iou);
        } catch {
            /* use defaults from HTML */
        }
    }

    // =========================================================================
    // Theme
    // =========================================================================

    function setTheme(isDark) {
        document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light');
        localStorage.setItem('jalsafayoo-theme', isDark ? 'dark' : 'light');
        DOM.themeCheckbox.checked = isDark;

        const label = document.querySelector('.theme-toggle span');
        if (label) {
            label.textContent = isDark ? 'Dark Mode' : 'Light Mode';
        }

        if (fpsChart || detectionsChart) {
            destroyCharts();
            initCharts();
        }
    }

    function initTheme() {
        const saved = localStorage.getItem('jalsafayoo-theme') || 'dark';
        setTheme(saved === 'dark');
    }

    // =========================================================================
    // Export CSV
    // =========================================================================

    function exportCSV() {
        window.open('/api/export/csv', '_blank');
        showToast('Exporting detection log…', 'info');
    }

    // =========================================================================
    // Event Bindings
    // =========================================================================

    function bindEvents() {
        // Video controls
        $$('.ctrl-btn[data-action]').forEach((btn) => {
            btn.addEventListener('click', () => {
                const action = btn.dataset.action;

                switch (action) {
                    case 'play':
                    case 'pause':
                    case 'resume':
                    case 'restart':
                    case 'stop':
                        controlAction(action);
                        break;
                    case 'fullscreen':
                        toggleFullscreen();
                        break;
                    case 'screenshot':
                        captureSnapshot();
                        break;
                    case 'save-video':
                        showToast('Video is saved automatically during detection', 'info');
                        break;
                    case 'download-video':
                        downloadLatestOutput();
                        break;
                }
            });
        });

        // Input sources
        document.getElementById('btn-connect-webcam')?.addEventListener('click', connectWebcam);
        document.getElementById('btn-play-demo')?.addEventListener('click', playDemoVideo);

        DOM.webcamUrl?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') connectWebcam();
        });

        // Upload
        DOM.uploadZone?.addEventListener('click', () => DOM.uploadInput?.click());

        DOM.uploadInput?.addEventListener('change', (e) => {
            if (e.target.files[0]) uploadVideo(e.target.files[0]);
        });

        DOM.uploadZone?.addEventListener('dragover', (e) => {
            e.preventDefault();
            DOM.uploadZone.classList.add('drag-over');
        });

        DOM.uploadZone?.addEventListener('dragleave', () => {
            DOM.uploadZone.classList.remove('drag-over');
        });

        DOM.uploadZone?.addEventListener('drop', (e) => {
            e.preventDefault();
            DOM.uploadZone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (file) uploadVideo(file);
        });

        // Settings
        DOM.confidenceSlider?.addEventListener('input', debounceSettings);
        DOM.iouSlider?.addEventListener('input', debounceSettings);
        document.getElementById('btn-reset-settings')?.addEventListener('click', resetSettings);

        DOM.themeCheckbox?.addEventListener('change', (e) => {
            setTheme(e.target.checked);
        });

        // Export
        document.getElementById('btn-export-csv')?.addEventListener('click', exportCSV);

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.target.matches('input, textarea, select')) return;

            switch (e.key.toLowerCase()) {
                case 'p':
                    controlAction('play');
                    break;
                case ' ':
                    e.preventDefault();
                    controlAction(state.isPaused ? 'resume' : 'pause');
                    break;
                case 'r':
                    controlAction('resume');
                    break;
                case 's':
                    controlAction('stop');
                    break;
                case 'f':
                    toggleFullscreen();
                    break;
                case 'c':
                    captureSnapshot();
                    break;
                case 'd':
                    downloadLatestOutput();
                    break;
                case 'e':
                    exportCSV();
                    break;
            }
        });
    }

    // =========================================================================
    // Initialization
    // =========================================================================

    async function init() {
        runLoadingSequence();
        initTheme();
        initCharts();
        bindEvents();
        initSocket();

        updateClock();
        setInterval(updateClock, 1000);

        await loadDemoVideos();
        await loadSettings();

        try {
            const health = await api('/api/health');
            if (health.model_loaded) {
                DOM.statModel.textContent = health.model_status;
            } else {
                DOM.statModel.textContent = 'Model Not Found';
                showToast('Place best.pt in project root to enable detection', 'warning', 6000);
            }
        } catch {
            /* server may still be starting */
        }
    }

    document.addEventListener('DOMContentLoaded', init);
})();
