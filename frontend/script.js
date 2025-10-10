/**
 * medAI MVP Frontend JavaScript
 * Handles audio recording, WebSocket communication, and UI updates
 */

class MedAIApp {
    constructor() {
        this.websocket = null;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.isRecording = false;
        this.sessionId = null;
        this.encounterId = null;
        this.startTime = null;
        this.durationInterval = null;
        this.chunkCount = 0;
        this.totalSize = 0;

        this.initializeElements();
        this.attachEventListeners();
        this.generateSessionId();
        this.initializeUI();
    }

    initializeElements() {
        // Form elements
        this.encounterIdInput = document.getElementById('encounterId');
        this.taskTypeSelect = document.getElementById('taskType');
        this.translateToSelect = document.getElementById('translateTo');

        // Control elements
        this.recordButton = document.getElementById('recordButton');
        this.recordText = document.getElementById('recordText');
        this.startButton = document.getElementById('startButton');
        this.stopButton = document.getElementById('stopButton');
        this.clearButton = document.getElementById('clearButton');
        this.saveButton = document.getElementById('saveButton');
        this.downloadButton = document.getElementById('downloadButton');

        // Status elements
        this.statusDiv = document.getElementById('status');
        this.progressFill = document.getElementById('progressFill');

        // Display elements
        this.transcriptText = document.getElementById('transcriptText');
        this.entitiesText = document.getElementById('entitiesText');
        this.summaryText = document.getElementById('summaryText');

        console.log('Elements initialized:', {
            transcriptText: this.transcriptText,
            entitiesText: this.entitiesText,
            summaryText: this.summaryText
        });

        // Stats elements
        this.durationValue = document.getElementById('durationValue');
        this.chunksValue = document.getElementById('chunksValue');
        this.sizeValue = document.getElementById('sizeValue');

        // Messages container
        this.messagesContainer = document.getElementById('messages');
    }

    attachEventListeners() {
        this.recordButton.addEventListener('click', () => {
            console.log('Record button clicked, disabled:', this.recordButton.disabled);
            this.toggleRecording();
        });
        this.startButton.addEventListener('click', () => {
            console.log('Start button clicked');
            this.startSession();
        });
        this.stopButton.addEventListener('click', () => this.stopSession());
        this.clearButton.addEventListener('click', () => this.clearAll());
        this.saveButton.addEventListener('click', () => this.saveResults());
        this.downloadButton.addEventListener('click', () => this.downloadResults());
    }

    generateSessionId() {
        this.sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
    }

    initializeUI() {
        // Disable record button initially - user must start session first
        this.recordButton.disabled = true;
        this.stopButton.disabled = true;
        this.saveButton.disabled = true;
        this.downloadButton.disabled = true;

        console.log('UI initialized - record button disabled:', this.recordButton.disabled);

        // Set initial status
        this.updateStatus('Bereit für Sitzung', 'ready');
    }

    async startSession() {
        try {
            console.log('Starting session...');
            this.encounterId = this.encounterIdInput.value.trim();
            if (!this.encounterId) {
                this.showMessage('Bitte geben Sie eine Encounter ID ein.', 'error');
                return;
            }

            console.log('Encounter ID:', this.encounterId);

            // Initialize WebSocket connection
            console.log('Connecting WebSocket...');
            await this.connectWebSocket();
            console.log('WebSocket connected');

            // Initialize audio recording
            console.log('Initializing audio...');
            await this.initializeAudio();
            console.log('Audio initialized');

            this.startButton.disabled = true;
            this.recordButton.disabled = false;
            console.log('Record button enabled:', !this.recordButton.disabled);
            this.updateStatus('Bereit für Aufnahme', 'ready');

            this.showMessage('Sitzung gestartet. Sie können jetzt mit der Aufnahme beginnen.', 'success');

        } catch (error) {
            console.error('Failed to start session:', error);
            this.showMessage(`Fehler beim Starten der Sitzung: ${error.message}`, 'error');
        }
    }

    async connectWebSocket() {
        return new Promise((resolve, reject) => {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}?user_id=demo_user&organization_id=demo_org`;

            console.log('Attempting WebSocket connection to:', wsUrl);
            this.websocket = new WebSocket(wsUrl);

            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.sendWebSocketMessage('start_session', {
                    encounter_id: this.encounterId,
                    task_type: this.taskTypeSelect.value,
                    translate_to: this.translateToSelect.value || null
                });
                resolve();
            };

            this.websocket.onmessage = (event) => {
                this.handleWebSocketMessage(JSON.parse(event.data));
            };

            this.websocket.onclose = () => {
                console.log('WebSocket disconnected');
                this.updateStatus('Verbindung getrennt', 'error');
            };

            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                console.error('WebSocket connection failed - check if server is running');
                reject(new Error('WebSocket connection failed'));
            };
        });
    }

    sendWebSocketMessage(type, data = {}) {
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify({ type, data }));
        }
    }

    handleWebSocketMessage(message) {
        console.log('WebSocket message:', message);

        switch (message.type) {
            case 'connected':
                console.log('WebSocket session connected');
                break;

            case 'session_started':
                console.log('Session started:', message.data);
                break;

            case 'audio_received':
                console.log('Audio chunk received:', message.data);
                this.updateProgress(message.data.total_size);
                break;

            case 'partial_transcription':
                console.log('Partial transcription received:', message.data);
                console.log('Calling updateTranscript with:', message.data.transcription);
                this.updateTranscript(message.data.transcription, false);
                console.log('updateTranscript call completed');
                break;

            case 'processing_started':
                this.updateStatus('Verarbeitung läuft...', 'processing');
                this.recordButton.disabled = true;
                break;

            case 'processing_completed':
                this.handleProcessingCompleted(message.data);
                break;

            case 'error':
                this.showMessage(`Fehler: ${message.error}`, 'error');
                break;

            default:
                console.log('Unknown message type:', message.type);
        }
    }

    async initializeAudio() {
        try {
            console.log('Requesting microphone access...');
            const stream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                }
            });
            console.log('Microphone access granted');

            console.log('Creating MediaRecorder...');
            this.mediaRecorder = new MediaRecorder(stream, {
                mimeType: 'audio/webm;codecs=opus',
                audioBitsPerSecond: 16000
            });
            console.log('MediaRecorder created successfully');

            this.mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) {
                    this.audioChunks.push(event.data);
                    this.sendAudioChunk(event.data);
                }
            };

            this.mediaRecorder.onstop = () => {
                this.handleRecordingStop();
            };

        } catch (error) {
            console.error('Failed to initialize audio:', error);
            throw new Error('Mikrofonzugriff fehlgeschlagen. Bitte erlauben Sie den Zugriff auf das Mikrofon.');
        }
    }

    async toggleRecording() {
        console.log('toggleRecording called, mediaRecorder:', !!this.mediaRecorder, 'isRecording:', this.isRecording);

        // Check if session is started
        if (!this.mediaRecorder) {
            console.log('No mediaRecorder, showing error message');
            this.showMessage('Bitte starten Sie zuerst eine Sitzung mit dem "Sitzung starten" Button.', 'error');
            return;
        }

        if (!this.isRecording) {
            console.log('Starting recording...');
            await this.startRecording();
        } else {
            console.log('Stopping recording...');
            this.stopRecording();
        }
    }

    async startRecording() {
        try {
            // Check if mediaRecorder is initialized
            if (!this.mediaRecorder) {
                this.showMessage('Bitte starten Sie zuerst eine Sitzung.', 'error');
                return;
            }

            // Check if WebSocket is connected
            if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
                this.showMessage('WebSocket-Verbindung nicht verfügbar. Bitte starten Sie eine neue Sitzung.', 'error');
                return;
            }

            this.audioChunks = [];
            this.startTime = Date.now();
            this.chunkCount = 0;
            this.totalSize = 0;

            this.mediaRecorder.start(1000); // Collect data every second
            this.isRecording = true;

            this.recordButton.classList.add('recording');
            this.recordText.textContent = 'Aufnahme stoppen';
            this.stopButton.disabled = false;

            this.updateStatus('Aufnahme läuft...', 'recording');
            this.startDurationTimer();

        } catch (error) {
            console.error('Failed to start recording:', error);
            this.showMessage(`Fehler beim Starten der Aufnahme: ${error.message}`, 'error');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;

            this.recordButton.classList.remove('recording');
            this.recordText.textContent = 'Aufnahme starten';
            this.stopButton.disabled = true;

            this.updateStatus('Aufnahme beendet, verarbeite...', 'processing');
            this.stopDurationTimer();
        }
    }

    handleRecordingStop() {
        // Send end session message to trigger final processing
        this.sendWebSocketMessage('end_session', {});
    }

    async sendAudioChunk(audioBlob) {
        try {
            const arrayBuffer = await audioBlob.arrayBuffer();
            const base64 = this.arrayBufferToBase64(arrayBuffer);

            this.chunkCount++;
            this.totalSize += audioBlob.size;

            this.sendWebSocketMessage('audio_chunk', {
                audio_data: base64
            });

            this.updateStats();

        } catch (error) {
            console.error('Failed to send audio chunk:', error);
        }
    }

    arrayBufferToBase64(buffer) {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        return btoa(binary);
    }

    startDurationTimer() {
        this.durationInterval = setInterval(() => {
            if (this.startTime) {
                const duration = Math.floor((Date.now() - this.startTime) / 1000);
                this.durationValue.textContent = duration;
            }
        }, 1000);
    }

    stopDurationTimer() {
        if (this.durationInterval) {
            clearInterval(this.durationInterval);
            this.durationInterval = null;
        }
    }

    updateStats() {
        this.chunksValue.textContent = this.chunkCount;
        this.sizeValue.textContent = Math.round(this.totalSize / 1024);
    }

    updateProgress(totalSize) {
        const maxSize = 1024 * 1024; // 1MB max
        const progress = Math.min((totalSize / maxSize) * 100, 100);
        this.progressFill.style.width = `${progress}%`;
    }

    updateTranscript(text, isFinal = false) {
        console.log('updateTranscript called with:', { text, isFinal, element: this.transcriptText });

        if (!this.transcriptText) {
            console.error('transcriptText element not found!');
            return;
        }

        if (isFinal) {
            this.transcriptText.textContent = text;
            console.log('Final transcript set to:', text);
        } else {
            this.transcriptText.textContent = text + '...';
            console.log('Partial transcript set to:', text + '...');
        }

        // Force a visual update
        this.transcriptText.style.display = 'none';
        this.transcriptText.offsetHeight; // Trigger reflow
        this.transcriptText.style.display = '';
    }

    updateEntities(entities) {
        if (!entities || entities.length === 0) {
            this.entitiesText.innerHTML = '<em>Keine physiotherapeutischen Entitäten gefunden.</em>';
            return;
        }

        const entitiesHtml = entities.map(entity => `
            <div class="entity-item">
                <div class="entity-label">${entity.label}</div>
                <div class="entity-text">${entity.text}</div>
                ${entity.icd_code ? `<div class="entity-icd">ICD: ${entity.icd_code}</div>` : ''}
            </div>
        `).join('');

        this.entitiesText.innerHTML = entitiesHtml;
    }

    updateSummary(summary) {
        if (typeof summary === 'string') {
            this.summaryText.innerHTML = `<pre>${summary}</pre>`;
        } else if (typeof summary === 'object') {
            // Handle structured notes
            let html = '';
            for (const [key, value] of Object.entries(summary)) {
                if (value && (Array.isArray(value) ? value.length > 0 : true)) {
                    html += `
                        <div class="summary-section">
                            <h3>${this.formatKey(key)}</h3>
                            <div class="summary-content">
                                ${Array.isArray(value) ? value.join(', ') : value}
                            </div>
                        </div>
                    `;
                }
            }
            this.summaryText.innerHTML = html || '<em>Keine Zusammenfassung verfügbar.</em>';
        } else {
            this.summaryText.innerHTML = '<em>Keine Zusammenfassung verfügbar.</em>';
        }
    }

    formatKey(key) {
        const keyMap = {
            'hauptbeschwerden': 'Hauptbeschwerden',
            'aktuelle_symptome': 'Aktuelle Symptome',
            'medizinische_vorgeschichte': 'Medizinische Vorgeschichte',
            'schmerzanalyse': 'Schmerzanalyse',
            'bewegungseinschraenkungen': 'Bewegungseinschränkungen',
            'funktionelle_einschraenkungen': 'Funktionelle Einschränkungen',
            'therapieziele': 'Therapieziele',
            'medikamente': 'Medikamente',
            'allergien': 'Allergien',
            'soziale_angelegenheiten': 'Soziale Angelegenheiten',
            'befunde': 'Befunde',
            'diagnose_verdacht': 'Diagnose/Verdacht',
            'behandlungsplan': 'Behandlungsplan',
            'naechste_schritte': 'Nächste Schritte'
        };
        return keyMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }

    handleProcessingCompleted(data) {
        console.log('Processing completed:', data);

        this.updateStatus('Verarbeitung abgeschlossen', 'ready');
        this.recordButton.disabled = false;

        if (data.success) {
            this.updateTranscript(data.transcription, true);
            this.updateEntities(data.entities);
            this.updateSummary(data.structured_notes);

            this.saveButton.disabled = false;
            this.downloadButton.disabled = false;

            this.showMessage(`Verarbeitung erfolgreich abgeschlossen in ${Math.round(data.processing_time_ms)}ms`, 'success');

            // Store results for download
            this.lastResults = data;

        } else {
            this.showMessage(`Verarbeitung fehlgeschlagen: ${data.errors.join(', ')}`, 'error');
        }
    }

    async stopSession() {
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }

        if (this.mediaRecorder && this.isRecording) {
            this.stopRecording();
        }

        this.startButton.disabled = false;
        this.recordButton.disabled = true;
        this.stopButton.disabled = true;

        this.updateStatus('Sitzung beendet', 'ready');
        this.showMessage('Sitzung beendet', 'success');
    }

    clearAll() {
        this.transcriptText.textContent = 'Transkription wird hier angezeigt...';
        this.entitiesText.innerHTML = 'Physiotherapeutische Entitäten werden hier angezeigt...';
        this.summaryText.innerHTML = 'Physiotherapeutische Zusammenfassung wird hier angezeigt...';

        this.durationValue.textContent = '0';
        this.chunksValue.textContent = '0';
        this.sizeValue.textContent = '0';

        this.progressFill.style.width = '0%';

        this.saveButton.disabled = true;
        this.downloadButton.disabled = true;

        this.lastResults = null;

        // Reset recording state
        this.isRecording = false;
        this.recordButton.classList.remove('recording');
        this.recordText.textContent = 'Aufnahme starten';
        this.stopButton.disabled = true;

        this.showMessage('Alle Daten gelöscht', 'success');
    }

    async saveResults() {
        if (!this.lastResults) {
            this.showMessage('Keine Ergebnisse zum Speichern verfügbar', 'error');
            return;
        }

        try {
            // In a real implementation, this would call the API to save results
            this.showMessage('Ergebnisse gespeichert (Demo-Modus)', 'success');
        } catch (error) {
            console.error('Failed to save results:', error);
            this.showMessage(`Fehler beim Speichern: ${error.message}`, 'error');
        }
    }

    downloadResults() {
        if (!this.lastResults) {
            this.showMessage('Keine Ergebnisse zum Download verfügbar', 'error');
            return;
        }

        try {
            const results = {
                encounter_id: this.lastResults.encounter_id,
                transcription: this.lastResults.transcription,
                entities: this.lastResults.entities,
                clinical_summary: this.lastResults.clinical_summary,
                structured_notes: this.lastResults.structured_notes,
                processing_time_ms: this.lastResults.processing_time_ms,
                timestamp: new Date().toISOString()
            };

            const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `medai_results_${this.encounterId}_${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.showMessage('Ergebnisse heruntergeladen', 'success');

        } catch (error) {
            console.error('Failed to download results:', error);
            this.showMessage(`Fehler beim Download: ${error.message}`, 'error');
        }
    }

    updateStatus(message, type) {
        this.statusDiv.textContent = message;
        this.statusDiv.className = `status ${type}`;
    }

    showMessage(message, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `${type}-message`;
        messageDiv.textContent = message;

        this.messagesContainer.appendChild(messageDiv);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.parentNode.removeChild(messageDiv);
            }
        }, 5000);
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.medAIApp = new MedAIApp();
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.medAIApp && window.medAIApp.websocket) {
        window.medAIApp.websocket.close();
    }
});
