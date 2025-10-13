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
        this.latestTranscript = '';
        this.lastResults = null;
        this.reportManager = null;
        this.isReportMode = false;

        this.initializeElements();
        this.attachEventListeners();
        this.generateSessionId();
        this.initializeUI();
        this.initializeReportForm();
        this.handleTaskTypeChange();
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
        this.resultsSaveButton = document.getElementById('resultsSaveButton');
        this.resultsDownloadButton = document.getElementById('resultsDownloadButton');
        this.reportSuggestButton = document.getElementById('reportSuggestButton');
        this.reportSaveButton = document.getElementById('reportSaveButton');
        this.reportDownloadButton = document.getElementById('reportDownloadButton');

        // Panel elements
        this.resultsCard = document.getElementById('resultsCard');
        this.reportCard = document.getElementById('reportCard');

        // Status elements
        this.statusDiv = document.getElementById('status');
        this.progressFill = document.getElementById('progressFill');

        // Display elements
        this.transcriptText = document.getElementById('transcriptText');
        this.reportTranscriptText = document.getElementById('reportTranscriptText');
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
        if (this.taskTypeSelect) {
            this.taskTypeSelect.addEventListener('change', () => this.handleTaskTypeChange());
        }
        if (this.resultsSaveButton) {
            this.resultsSaveButton.addEventListener('click', () => this.saveResults());
        }
        if (this.resultsDownloadButton) {
            this.resultsDownloadButton.addEventListener('click', () => this.downloadResults());
        }
        if (this.reportSuggestButton) {
            this.reportSuggestButton.addEventListener('click', () => this.fillReportFromTranscript());
        }
        if (this.reportSaveButton) {
            this.reportSaveButton.addEventListener('click', () => this.saveReport());
        }
        if (this.reportDownloadButton) {
            this.reportDownloadButton.addEventListener('click', () => this.downloadReport());
        }
    }

    generateSessionId() {
        this.sessionId = 'session_' + Math.random().toString(36).substr(2, 9);
    }

    initializeUI() {
        // Disable buttons initially - user must start session first
        this.recordButton.disabled = true;
        this.stopButton.disabled = true;
        this.updateResultsAvailability();

        console.log('UI initialized - record button disabled:', this.recordButton.disabled);

        // Set initial status
        this.updateStatus('Bereit für Sitzung', 'ready');
    }

    initializeReportForm() {
        this.reportManager = new ReportFormManager(this);
    }

    handleTaskTypeChange() {
        this.isReportMode = this.taskTypeSelect && this.taskTypeSelect.value === 'treatment_report';
        if (this.resultsCard) {
            this.resultsCard.classList.toggle('hidden', this.isReportMode);
        }
        if (this.reportCard) {
            this.reportCard.classList.toggle('hidden', !this.isReportMode);
        }
        this.updateResultsAvailability();
        if (this.reportManager) {
            this.reportManager.setActive(this.isReportMode);
        }
    }

    updateResultsAvailability() {
        const hasResults = Boolean(this.lastResults);
        const enableResults = hasResults && !this.isReportMode;
        if (this.resultsSaveButton) {
            this.resultsSaveButton.disabled = !enableResults;
        }
        if (this.resultsDownloadButton) {
            this.resultsDownloadButton.disabled = !enableResults;
        }
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
                    translate_to: this.translateToSelect ? this.translateToSelect.value || null : null
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

            case 'recording_started':
                console.log('Recording started:', message.data);
                this.updateStatus('Recording started, ready for audio', 'success');
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
                this.updateStatusWithLoading('Transcript and Physiotherapy summary is ready', 'processing');
                this.recordButton.disabled = true;
                break;

            case 'processing_completed':
                this.handleProcessingCompleted(message.data);
                break;

            case 'session_ended':
                console.log('Session ended:', message.data);
                this.updateStatus('Session ended successfully', 'success');
                this.resetUI();
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

            // Send start_recording message to clear buffers
            this.sendWebSocketMessage('start_recording', {});

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

            // Show loading icon with specific message
            this.updateStatusWithLoading('Transcript and Physiotherapy summary is ready', 'processing');
            this.stopDurationTimer();
        }
    }

    handleRecordingStop() {
        // Send end recording message to trigger processing of current recording
        this.sendWebSocketMessage('end_recording', {});
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
        console.log('updateTranscript called with:', { text, isFinal });

        this.updateTranscriptElement(this.transcriptText, text, isFinal);
        this.updateTranscriptElement(this.reportTranscriptText, text, isFinal);
        this.latestTranscript = text;
        if (this.reportManager) {
            this.reportManager.handleTranscriptUpdate(this.latestTranscript);
        }

        this.forceTranscriptReflow(this.transcriptText);
        this.forceTranscriptReflow(this.reportTranscriptText);
    }

    updateTranscriptElement(element, text, isFinal) {
        if (!element) {
            return;
        }
        element.textContent = isFinal ? text : `${text}...`;
    }

    forceTranscriptReflow(element) {
        if (!element) {
            return;
        }
        element.style.display = 'none';
        element.offsetHeight;
        element.style.display = '';
    }

    updateEntities(entities) {
        if (!this.entitiesText) {
            return;
        }
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
        if (!this.summaryText) {
            return;
        }
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

        // Remove loading spinner and show completion message
        this.updateStatus('Verarbeitung abgeschlossen', 'ready');
        this.recordButton.disabled = false;

        if (data.success) {
            this.updateTranscript(data.transcription, true);
            this.updateEntities(data.entities);

            // Handle both clinical_summary and structured_notes
            if (data.clinical_summary) {
                this.updateSummary(data.clinical_summary);
            } else if (data.structured_notes) {
                this.updateSummary(data.structured_notes);
            }

            this.showMessage(`Verarbeitung erfolgreich abgeschlossen in ${Math.round(data.processing_time_ms)}ms`, 'success');

            // Store results for download
            this.lastResults = data;
            this.updateResultsAvailability();

        } else {
            this.showMessage(`Verarbeitung fehlgeschlagen: ${data.errors.join(', ')}`, 'error');
        }
    }

    async stopSession() {
        // Stop any active recording first
        if (this.mediaRecorder && this.isRecording) {
            this.stopRecording();
        }

        // Send end_session message to properly close the session
        if (this.websocket && this.websocket.readyState === WebSocket.OPEN) {
            this.sendWebSocketMessage('end_session', {});
            // Don't close WebSocket immediately - let the server handle it
        } else {
            // If WebSocket is not available, just reset UI
            this.resetUI();
        }
    }

    resetUI() {
        // Reset UI state
        this.startButton.disabled = false;
        this.recordButton.disabled = true;
        this.stopButton.disabled = true;

        // Reset recording state
        this.isRecording = false;
        this.audioChunks = [];
        this.chunkCount = 0;
        this.totalSize = 0;
        this.latestTranscript = '';

        // Reset UI elements
        this.recordButton.classList.remove('recording');
        this.recordText.textContent = 'Aufnahme starten';

        // Stop any timers
        if (this.durationInterval) {
            clearInterval(this.durationInterval);
            this.durationInterval = null;
        }

        // Close WebSocket if still open
        if (this.websocket) {
            this.websocket.close();
            this.websocket = null;
        }

        // Reset status
        this.updateStatus('Bereit für neue Sitzung', 'ready');
    }

    async fillReportFromTranscript() {
        if (!this.reportManager) {
            this.showMessage('Berichtsformular nicht verfügbar.', 'error');
            return;
        }
        if (!this.isReportMode) {
            this.showMessage('Bitte wählen Sie den Aufgabentyp "Behandlungsbericht" aus.', 'error');
            return;
        }
        await this.reportManager.suggestFromTranscript();
    }

    clearAll() {
        if (this.transcriptText) {
            this.transcriptText.textContent = 'Transkription wird hier angezeigt...';
        }
        if (this.reportTranscriptText) {
            this.reportTranscriptText.textContent = 'Transkription wird hier angezeigt...';
        }
        if (this.entitiesText) {
            this.entitiesText.innerHTML = 'Physiotherapeutische Entitäten werden hier angezeigt...';
        }
        if (this.summaryText) {
            this.summaryText.innerHTML = 'Physiotherapeutische Zusammenfassung wird hier angezeigt...';
        }

        this.durationValue.textContent = '0';
        this.chunksValue.textContent = '0';
        this.sizeValue.textContent = '0';

        this.progressFill.style.width = '0%';

        this.lastResults = null;
        this.latestTranscript = '';
        this.updateResultsAvailability();
        if (this.reportManager) {
            this.reportManager.reset();
        }

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
            this.showMessage('Ergebnisse gespeichert (Demo-Modus)', 'success');
        } catch (error) {
            console.error('Failed to save results:', error);
            this.showMessage(`Fehler beim Speichern: ${error.message}`, 'error');
        }
    }

    async downloadResults() {
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

    async saveReport() {
        if (!this.reportManager) {
            this.showMessage('Berichtsformular nicht verfügbar.', 'error');
            return;
        }
        await this.reportManager.saveReport();
    }

    async downloadReport() {
        if (!this.reportManager) {
            this.showMessage('Berichtsformular nicht verfügbar.', 'error');
            return;
        }
        await this.reportManager.downloadPdf();
    }

    updateStatus(message, type) {
        this.statusDiv.textContent = message;
        this.statusDiv.className = `status ${type}`;
    }

    updateStatusWithLoading(message, type) {
        // Clear the status div
        this.statusDiv.innerHTML = '';

        // Create loading spinner element
        const spinner = document.createElement('span');
        spinner.className = 'loading-spinner';

        // Create text node for the message
        const textNode = document.createTextNode(` ${message}`);

        // Append spinner and text to status div
        this.statusDiv.appendChild(spinner);
        this.statusDiv.appendChild(textNode);

        // Set the status class
        this.statusDiv.className = `status ${type}`;

        // Debug logging
        console.log('Loading spinner added:', this.statusDiv.innerHTML);
        console.log('Status div class:', this.statusDiv.className);
        console.log('Spinner element:', spinner);
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


class ReportFormManager {
    constructor(app) {
        this.app = app;
        this.form = document.getElementById('reportForm');
        this.defaults = null;
        this.readOnly = {
            doctor_name: document.getElementById('doctorName'),
            patient_name: document.getElementById('patientName'),
            patient_dob: document.getElementById('patientDob'),
            prescription_date: document.getElementById('prescriptionDate'),
            treatment_date_from: document.getElementById('treatmentDateFrom'),
            treatment_date_to: document.getElementById('treatmentDateTo'),
            physiotherapist_name: document.getElementById('physiotherapistName')
        };
        this.editable = {
            report_city: document.getElementById('reportCity'),
            report_date: document.getElementById('reportDate'),
            insurance_type: document.getElementById('insuranceType'),
            diagnoses: document.getElementById('diagnoses'),
            prescribed_therapy_type: document.getElementById('therapyType'),
            patient_problem_statement: document.getElementById('problemStatement'),
            treatment_outcome: document.getElementById('treatmentOutcome'),
            therapy_status_note: document.getElementById('therapyStatus'),
            follow_up_recommendation: document.getElementById('followUp')
        };

        // Debug: Check if form elements are found
        console.log('Form elements found:');
        Object.entries(this.editable).forEach(([key, element]) => {
            console.log(`  ${key}: ${element ? 'FOUND' : 'NOT FOUND'}`);
        });
        this.buttons = {
            suggest: document.getElementById('reportSuggestButton'),
            save: document.getElementById('reportSaveButton'),
            download: document.getElementById('reportDownloadButton')
        };
        this.active = false;
        this.latestTranscript = '';
        if (this.buttons.suggest) {
            this.buttons.suggest.disabled = true;
        }
        this.reportData = {
            doctor_name: '',
            patient_name: '',
            patient_dob: '',
            prescription_date: '',
            treatment_date_from: '',
            treatment_date_to: '',
            physiotherapist_name: '',
            report_city: '',
            report_date: this.getToday(),
            insurance_type: 'UNKLAR',
            diagnoses: [],
            prescribed_therapy_type: '',
            patient_problem_statement: '',
            treatment_outcome: 'UNKLAR',
            therapy_status_note: '',
            follow_up_recommendation: '',
            transcript: null
        };

        if (this.form) {
            this.initialize();
        }
    }

    async initialize() {
        await this.loadDefaults();
        this.bindInputs();
        this.evaluateActionAvailability();
    }

    setActive(isActive) {
        this.active = Boolean(isActive);
        if (this.form) {
            this.form.classList.toggle('report-form--inactive', !this.active);
        }
        Object.values(this.editable).forEach((element) => {
            if (!element) {
                return;
            }
            element.disabled = !this.active;
        });
        this.evaluateActionAvailability();
    }

    getToday() {
        return new Date().toISOString().split('T')[0];
    }

    async loadDefaults() {
        try {
            const response = await fetch('/api/reports/defaults');
            if (!response.ok) {
                throw new Error(`Serverantwort ${response.status}`);
            }
            this.defaults = await response.json();
            this.applyDefaults();
        } catch (error) {
            console.error('Failed to load report defaults:', error);
            this.app.showMessage(`Standards konnten nicht geladen werden: ${error.message}`, 'error');
        }
    }

    applyDefaults() {
        const defaults = this.defaults || {};
        this.reportData = {
            ...this.reportData,
            doctor_name: defaults.doctor_name ?? this.reportData.doctor_name,
            patient_name: defaults.patient_name ?? this.reportData.patient_name,
            patient_dob: defaults.patient_dob ?? this.reportData.patient_dob,
            prescription_date: defaults.prescription_date ?? this.reportData.prescription_date,
            treatment_date_from: defaults.treatment_date_from ?? this.reportData.treatment_date_from,
            treatment_date_to: defaults.treatment_date_to ?? this.reportData.treatment_date_to,
            physiotherapist_name: defaults.physiotherapist_name ?? this.reportData.physiotherapist_name,
            report_city: this.reportData.report_city || (defaults.report_city || ''),
            report_date: this.reportData.report_date || this.getToday(),
            transcript: this.reportData.transcript
        };
        Object.entries(this.readOnly).forEach(([key, element]) => {
            if (element && key in this.reportData) {
                element.value = this.reportData[key] || '';
            }
        });
        if (this.editable.report_city) {
            this.editable.report_city.value = this.reportData.report_city || '';
        }
        if (this.editable.report_date) {
            this.editable.report_date.value = this.reportData.report_date || this.getToday();
        }
        if (this.editable.insurance_type) {
            this.editable.insurance_type.value = this.reportData.insurance_type;
        }
        if (this.editable.diagnoses) {
            this.editable.diagnoses.value = this.reportData.diagnoses.join('\n');
        }
        if (this.editable.prescribed_therapy_type) {
            this.editable.prescribed_therapy_type.value = this.reportData.prescribed_therapy_type;
        }
        if (this.editable.patient_problem_statement) {
            this.editable.patient_problem_statement.value = this.reportData.patient_problem_statement;
        }
        if (this.editable.treatment_outcome) {
            this.editable.treatment_outcome.value = this.reportData.treatment_outcome;
        }
        if (this.editable.therapy_status_note) {
            this.editable.therapy_status_note.value = this.reportData.therapy_status_note;
        }
        if (this.editable.follow_up_recommendation) {
            this.editable.follow_up_recommendation.value = this.reportData.follow_up_recommendation;
        }
    }
    bindInputs() {
        Object.entries(this.editable).forEach(([key, element]) => {
            if (!element) {
                return;
            }
            const eventName = element.tagName === 'SELECT' ? 'change' : 'input';
            element.addEventListener(eventName, (event) => {
                this.updateField(key, event.target.value);
            });
        });
    }

    updateField(key, rawValue) {
        const value = typeof rawValue === 'string' ? rawValue.trim() : rawValue;
        switch (key) {
            case 'diagnoses':
                this.reportData.diagnoses = value
                    ? value.split(/\n+/).map((item) => item.trim()).filter(Boolean)
                    : [];
                break;
            case 'report_date':
                this.reportData.report_date = value || this.getToday();
                break;
            default:
                this.reportData[key] = value;
                break;
        }
        this.evaluateActionAvailability();
    }

    evaluateActionAvailability() {
        const ready = this.active && Boolean(
            (this.reportData.diagnoses && this.reportData.diagnoses.length > 0) ||
            this.reportData.prescribed_therapy_type ||
            this.reportData.patient_problem_statement ||
            this.reportData.therapy_status_note ||
            this.reportData.follow_up_recommendation
        );
        if (this.buttons.save) {
            this.buttons.save.disabled = !ready;
        }
        if (this.buttons.download) {
            this.buttons.download.disabled = !ready;
        }
        if (this.buttons.suggest) {
            this.buttons.suggest.disabled = !this.active || !this.latestTranscript;
        }
    }

    handleTranscriptUpdate(transcript) {
        this.latestTranscript = transcript || '';
        this.reportData.transcript = this.latestTranscript;
        this.evaluateActionAvailability();
    }

    async suggestFromTranscript() {
        if (!this.active) {
            this.app.showMessage('Berichtsformular ist für diese Aufgabe deaktiviert.', 'error');
            return;
        }
        if (!this.latestTranscript) {
            this.app.showMessage('Kein Transkript verfügbar.', 'error');
            return;
        }
        try {
            const response = await fetch('/api/reports/suggest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ transcript: this.latestTranscript })
            });
            if (!response.ok) {
                throw new Error(`Serverantwort ${response.status}`);
            }
            const draft = await response.json();
            this.applyDraft(draft);
            this.app.showMessage('Vorschlag erfolgreich geladen.', 'success');
        } catch (error) {
            console.error('Failed to suggest report:', error);
            this.app.showMessage(`Ausfüllen fehlgeschlagen: ${error.message}`, 'error');
        }
    }

    applyDraft(draft) {
        this.reportData = {
            ...this.reportData,
            doctor_name: draft.doctor_name,
            patient_name: draft.patient_name,
            patient_dob: draft.patient_dob,
            prescription_date: draft.prescription_date,
            treatment_date_from: draft.treatment_date_from,
            treatment_date_to: draft.treatment_date_to,
            physiotherapist_name: draft.physiotherapist_name,
            report_city: draft.report_city,
            report_date: draft.report_date || this.getToday(),
            insurance_type: draft.insurance_type,
            diagnoses: draft.diagnoses || [],
            prescribed_therapy_type: draft.prescribed_therapy_type || '',
            patient_problem_statement: draft.patient_problem_statement || '',
            treatment_outcome: draft.treatment_outcome,
            therapy_status_note: draft.therapy_status_note || '',
            follow_up_recommendation: draft.follow_up_recommendation || '',
            transcript: this.latestTranscript
        };
        this.applyDefaults();
        this.evaluateActionAvailability();
    }

    reset() {
        this.latestTranscript = '';
        if (this.buttons.suggest) {
            this.buttons.suggest.disabled = true;
        }
        this.reportData = {
            ...this.reportData,
            report_city: this.defaults ? this.defaults.report_city : '',
            report_date: this.getToday(),
            insurance_type: 'UNKLAR',
            diagnoses: [],
            prescribed_therapy_type: '',
            patient_problem_statement: '',
            treatment_outcome: 'UNKLAR',
            therapy_status_note: '',
            follow_up_recommendation: '',
            transcript: null
        };
        this.applyDefaults();
        this.evaluateActionAvailability();
    }

    buildPayload() {
        // Read current form values to ensure we have the latest data
        const currentFormData = this.getCurrentFormData();

        // Debug logging
        console.log('Building payload with current form data:', currentFormData);
        console.log('Report data:', this.reportData);
        console.log('Final payload being sent:', {
            doctor_name: this.reportData.doctor_name,
            patient_name: this.reportData.patient_name,
            patient_dob: this.reportData.patient_dob,
            prescription_date: this.reportData.prescription_date,
            treatment_date_from: this.reportData.treatment_date_from,
            treatment_date_to: this.reportData.treatment_date_to,
            physiotherapist_name: this.reportData.physiotherapist_name,
            report_city: currentFormData.report_city || (this.defaults ? this.defaults.report_city : ''),
            report_date: currentFormData.report_date || this.getToday(),
            insurance_type: currentFormData.insurance_type,
            diagnoses: currentFormData.diagnoses,
            prescribed_therapy_type: currentFormData.prescribed_therapy_type,
            patient_problem_statement: currentFormData.patient_problem_statement,
            treatment_outcome: currentFormData.treatment_outcome,
            therapy_status_note: currentFormData.therapy_status_note,
            follow_up_recommendation: currentFormData.follow_up_recommendation,
            transcript: this.reportData.transcript || this.latestTranscript
        });

        return {
            doctor_name: this.reportData.doctor_name,
            patient_name: this.reportData.patient_name,
            patient_dob: this.reportData.patient_dob,
            prescription_date: this.reportData.prescription_date,
            treatment_date_from: this.reportData.treatment_date_from,
            treatment_date_to: this.reportData.treatment_date_to,
            physiotherapist_name: this.reportData.physiotherapist_name,
            report_city: currentFormData.report_city || (this.defaults ? this.defaults.report_city : ''),
            report_date: currentFormData.report_date || this.getToday(),
            insurance_type: currentFormData.insurance_type,
            diagnoses: currentFormData.diagnoses,
            prescribed_therapy_type: currentFormData.prescribed_therapy_type,
            patient_problem_statement: currentFormData.patient_problem_statement,
            treatment_outcome: currentFormData.treatment_outcome,
            therapy_status_note: currentFormData.therapy_status_note,
            follow_up_recommendation: currentFormData.follow_up_recommendation,
            transcript: this.reportData.transcript || this.latestTranscript
        };
    }

    getCurrentFormData() {
        const formData = {};

        // Read current values from editable form fields
        Object.entries(this.editable).forEach(([key, element]) => {
            if (!element) {
                console.warn(`Form element not found for key: ${key}`);
                formData[key] = this.reportData[key] || '';
                return;
            }

            const value = element.value.trim();
            console.log(`Form field ${key}: "${value}"`);

            switch (key) {
                case 'diagnoses':
                    formData.diagnoses = value
                        ? value.split(/\n+/).map((item) => item.trim()).filter(Boolean)
                        : [];
                    break;
                case 'report_date':
                    formData.report_date = value || this.getToday();
                    break;
                default:
                    formData[key] = value;
                    break;
            }
        });

        return formData;
    }

    async saveReport() {
        if (!this.active) {
            this.app.showMessage('Berichtsformular ist für diese Aufgabe deaktiviert.', 'error');
            return;
        }
        const payload = this.buildPayload();
        try {
            const response = await fetch('/api/reports/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) {
                throw new Error(`Serverantwort ${response.status}`);
            }
            const result = await response.json();
            this.app.showMessage(`Bericht gespeichert: ${result.path}`, 'success');
        } catch (error) {
            console.error('Failed to save report:', error);
            this.app.showMessage(`Speichern fehlgeschlagen: ${error.message}`, 'error');
        }
    }

    async downloadPdf() {
        if (!this.active) {
            this.app.showMessage('Berichtsformular ist für diese Aufgabe deaktiviert.', 'error');
            return;
        }
        const payload = this.buildPayload();
        try {
            const response = await fetch('/api/reports/pdf', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!response.ok) {
                throw new Error(`Serverantwort ${response.status}`);
            }
            const blob = await response.blob();
            const filename = this.extractFilename(response.headers.get('Content-Disposition')) || `Behandlungsbericht_${Date.now()}.pdf`;
            const url = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = url;
            link.download = filename;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
            this.app.showMessage('PDF heruntergeladen.', 'success');
        } catch (error) {
            console.error('Failed to download PDF:', error);
            this.app.showMessage(`PDF konnte nicht erstellt werden: ${error.message}`, 'error');
        }
    }

    extractFilename(disposition) {
        if (!disposition) {
            return null;
        }
        const match = disposition.match(/filename\*=UTF-8''([^;]+)/);
        if (match && match[1]) {
            return decodeURIComponent(match[1]);
        }
        const simple = disposition.match(/filename="?([^";]+)"?/);
        return simple ? simple[1] : null;
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
