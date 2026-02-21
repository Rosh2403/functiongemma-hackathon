#!/usr/bin/env python3
"""
Simple web server for the product demo allowing interactive tool execution with voice support
"""
from flask import Flask, render_template_string, request, jsonify
from product_demo import run_text_command
from voice_action import run_voice_command
import json
import tempfile
import os
import soundfile as sf
import numpy as np

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 25 * 1024 * 1024  # 25MB max file size


def normalize_audio(audio_path: str) -> str:
    """
    Convert browser-recorded audio to standard WAV format that Cactus Whisper expects.

    Args:
        audio_path: Path to input audio file

    Returns:
        Path to normalized audio file
    """
    try:
        import librosa

        # Load audio with librosa (handles various formats)
        print(f"Loading audio from: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Ensure audio is not silent and is in proper format
        if np.max(np.abs(audio)) < 0.01:
            print("Warning: Audio appears to be very quiet or silent")

        # Save as standard WAV format (16kHz, mono, PCM)
        normalized_path = audio_path.replace('.wav', '_normalized.wav')
        sf.write(normalized_path, audio, 16000, subtype='PCM_16')

        print(f"Audio normalized to: {normalized_path}")
        return normalized_path
    except Exception as e:
        print(f"Audio normalization failed: {e}, using original file")
        return audio_path

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FunctionGemma Product Demo</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            max-width: 800px;
            width: 100%;
            padding: 40px;
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
        }
        .subtitle {
            color: #666;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .input-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            color: #333;
            margin-bottom: 8px;
            font-weight: 500;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 32px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        button:active {
            transform: translateY(0);
        }
        .loading {
            display: none;
            color: #667eea;
            margin-top: 10px;
            font-size: 14px;
        }
        .loading.show {
            display: block;
        }
        .spinner {
            display: inline-block;
            width: 12px;
            height: 12px;
            border: 2px solid #e0e0e0;
            border-top-color: #667eea;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .results {
            margin-top: 40px;
            display: none;
        }
        .results.show {
            display: block;
        }
        .result-section {
            background: #f8f9fa;
            border-left: 4px solid #667eea;
            padding: 16px;
            margin-bottom: 16px;
            border-radius: 4px;
        }
        .result-section h3 {
            color: #333;
            margin-bottom: 12px;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .result-content {
            background: white;
            padding: 12px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 12px;
            color: #333;
            overflow-x: auto;
            max-height: 300px;
            overflow-y: auto;
        }
        .result-content pre {
            margin: 0;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .examples {
            background: #f0f4ff;
            border-left: 4px solid #667eea;
            padding: 16px;
            margin-bottom: 30px;
            border-radius: 4px;
        }
        .examples h3 {
            color: #667eea;
            margin-bottom: 12px;
            font-size: 14px;
            font-weight: 600;
        }
        .example-list {
            list-style: none;
            font-size: 13px;
            color: #666;
        }
        .example-list li {
            margin-bottom: 8px;
        }
        .example-list li:before {
            content: "→ ";
            color: #667eea;
            font-weight: 600;
        }
        .quick-actions {
            margin-bottom: 30px;
        }
        .action-group {
            margin-bottom: 24px;
        }
        .action-group h4 {
            color: #667eea;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            font-weight: 700;
        }
        .button-grid {
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 10px;
        }
        .action-btn {
            background: white;
            border: 2px solid #e0e0e0;
            color: #333;
            padding: 12px 16px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            text-align: center;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .action-btn:hover {
            border-color: #667eea;
            background: #f5f7ff;
            color: #667eea;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
        }
        .action-btn:active {
            transform: translateY(0);
        }
        .input-group {
            margin-top: 24px;
        }
        .button-row {
            display: flex;
            gap: 12px;
            margin-top: 12px;
        }
        #executeBtn {
            flex: 1;
        }
        .clear-btn {
            background: #f0f0f0;
            color: #666;
            flex: 1;
        }
        .clear-btn:hover {
            background: #e0e0e0;
            box-shadow: none;
        }
        .tabs {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
            border-bottom: 2px solid #e0e0e0;
        }
        .tab-btn {
            background: none;
            border: none;
            padding: 12px 16px;
            font-size: 14px;
            font-weight: 600;
            color: #999;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            margin-bottom: -2px;
            transition: all 0.2s;
        }
        .tab-btn.active {
            color: #667eea;
            border-bottom-color: #667eea;
        }
        .tab-btn:hover {
            color: #667eea;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .voice-controls {
            background: #f8f9fa;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
        .voice-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            width: 120px;
            height: 120px;
            border-radius: 50%;
            font-size: 48px;
            cursor: pointer;
            transition: all 0.2s;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        .voice-btn:hover:not(:disabled) {
            transform: scale(1.05);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        .voice-btn:active:not(:disabled) {
            transform: scale(0.98);
        }
        .voice-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .voice-btn.recording {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3); }
            50% { box-shadow: 0 4px 25px rgba(255, 107, 107, 0.6); }
        }
        .voice-status {
            margin-top: 16px;
            font-size: 13px;
            color: #666;
        }
        .transcript-section {
            background: #f0f4ff;
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 16px;
            margin-top: 16px;
            display: none;
        }
        .transcript-section.show {
            display: block;
        }
        .transcript-section h4 {
            color: #667eea;
            margin-bottom: 8px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .transcript-text {
            background: white;
            padding: 12px;
            border-radius: 4px;
            font-size: 14px;
            color: #333;
            line-height: 1.5;
        }
        .transcript-text code {
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 12px;
            color: #d63384;
        }
        .transcript-text strong {
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 FunctionGemma Product Demo</h1>
        <p class="subtitle">Route and execute tool calls using hybrid routing strategy</p>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('text')">📝 Text Input</button>
            <button class="tab-btn" onclick="switchTab('voice')">🎤 Voice Input</button>
        </div>

        <!-- Text Input Tab -->
        <div id="text" class="tab-content active">
            <div class="quick-actions">
                <div class="action-group">
                    <h4>⚡ Quick Add</h4>
                    <div class="button-grid">
                        <button class="action-btn" onclick="quickAction('Set an alarm for 3 PM')">📍 Set Alarm</button>
                        <button class="action-btn" onclick="quickAction('Set a 10 minute timer')">⏱️ Set Timer</button>
                        <button class="action-btn" onclick="quickAction('Create a reminder to follow up at 5 PM')">📌 Reminder</button>
                    </div>
                </div>

                <div class="action-group">
                    <h4>📋 Planning</h4>
                    <div class="button-grid">
                        <button class="action-btn" onclick="quickAction('What\\'s the weather in New York?')">🌤️ Weather</button>
                        <button class="action-btn" onclick="quickAction('Play some lo-fi music')">🎵 Play Music</button>
                        <button class="action-btn" onclick="quickAction('Send a message to Alice saying Hello')">💬 Message</button>
                    </div>
                </div>

                <div class="action-group">
                    <h4>🎯 Smart Assist</h4>
                    <div class="button-grid">
                        <button class="action-btn" onclick="quickAction('Find John in contacts')">🔍 Search Contacts</button>
                        <button class="action-btn" onclick="quickAction('Set alarm for 8 AM and create reminder at 9 AM')">⚙️ Multi-Task</button>
                        <button class="action-btn" onclick="quickAction('Get weather in San Francisco and set 15 minute timer')">🔗 Combined</button>
                    </div>
                </div>
            </div>

            <div class="input-group">
                <label for="userInput">Enter your command:</label>
                <input
                    type="text"
                    id="userInput"
                    placeholder="e.g., What's the weather in New York?"
                    autocomplete="off"
                />
            </div>

            <div class="button-row">
                <button id="executeBtn" onclick="executeCommand()">Execute</button>
                <button class="clear-btn" onclick="clearOutput()">Clear</button>
            </div>
        </div>

        <!-- Voice Input Tab -->
        <div id="voice" class="tab-content">
            <div class="voice-controls">
                <button id="voiceBtn" class="voice-btn" onclick="toggleVoiceRecording()" title="Click to record">🎙️</button>
                <div class="voice-status" id="voiceStatus">Click the microphone to start recording</div>
            </div>

            <div style="background: #fff3cd; border: 2px solid #ffc107; border-radius: 8px; padding: 16px; margin: 20px 0; text-align: center;">
                <strong>💡 Test Mode:</strong> Whisper model not available. Use text input below to simulate voice:
            </div>

            <div class="input-group">
                <label for="voiceTextInput">Simulate voice input (for testing):</label>
                <input
                    type="text"
                    id="voiceTextInput"
                    placeholder="e.g., Set an alarm for 3 PM"
                    autocomplete="off"
                />
            </div>

            <div class="button-row">
                <button onclick="processVoiceText()">🎤 Process as Voice</button>
                <button class="clear-btn" onclick="clearVoiceOutput()">Clear</button>
            </div>

            <div class="transcript-section" id="transcriptSection">
                <h4>📝 Parsed Transcript</h4>
                <div class="transcript-text" id="transcriptText"></div>
            </div>
        </div>
        <div class="loading" id="loading">
            <span class="spinner"></span>Processing...
        </div>

        <div class="results" id="results">
            <div class="result-section">
                <h3>Input</h3>
                <div class="result-content">
                    <pre id="inputResult"></pre>
                </div>
            </div>

            <div class="result-section">
                <h3>Routing Decision</h3>
                <div class="result-content">
                    <pre id="routeResult"></pre>
                </div>
            </div>

            <div class="result-section">
                <h3>Tool Execution Results</h3>
                <div class="result-content">
                    <pre id="executionResult"></pre>
                </div>
            </div>
        </div>
    </div>


    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;

        const userInput = document.getElementById('userInput');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        const voiceBtn = document.getElementById('voiceBtn');
        const voiceStatus = document.getElementById('voiceStatus');

        // Tab switching
        function switchTab(tabName) {
            // Hide all tabs
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            document.querySelectorAll('.tab-btn').forEach(btn => {
                btn.classList.remove('active');
            });

            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            event.target.classList.add('active');
        }

        // Text input handlers
        userInput.addEventListener('keyup', (e) => {
            if (e.key === 'Enter') {
                executeCommand();
            }
        });

        function quickAction(command) {
            userInput.value = command;
            userInput.focus();
            setTimeout(() => executeCommand(), 100);
        }

        function clearOutput() {
            userInput.value = '';
            results.classList.remove('show');
            userInput.focus();
        }

        async function executeCommand() {
            const text = userInput.value.trim();
            if (!text) {
                alert('Please enter a command');
                return;
            }

            loading.classList.add('show');
            results.classList.remove('show');

            try {
                const response = await fetch('/api/execute', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text })
                });

                const data = await response.json();

                document.getElementById('inputResult').textContent = data.input;
                document.getElementById('routeResult').textContent = JSON.stringify(data.route, null, 2);
                document.getElementById('executionResult').textContent = JSON.stringify(data.executions, null, 2);

                results.classList.add('show');
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                loading.classList.remove('show');
            }
        }

        // Voice recording handlers
        async function toggleVoiceRecording() {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        }

        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.addEventListener('dataavailable', (event) => {
                    audioChunks.push(event.data);
                });

                mediaRecorder.addEventListener('stop', () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    processVoiceRecording(audioBlob);
                });

                mediaRecorder.start();
                isRecording = true;
                voiceBtn.classList.add('recording');
                voiceStatus.textContent = '🔴 Recording... Click to stop';
            } catch (error) {
                voiceStatus.textContent = '❌ Microphone access denied';
                console.error('Error accessing microphone:', error);
            }
        }

        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                voiceBtn.classList.remove('recording');
                voiceStatus.textContent = '⏳ Processing audio...';
            }
        }

        async function processVoiceRecording(audioBlob) {
            loading.classList.add('show');
            console.log('Audio blob size:', audioBlob.size, 'type:', audioBlob.type);

            try {
                const formData = new FormData();
                formData.append('audio', audioBlob, 'recording.wav');

                const response = await fetch('/api/voice', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();
                console.log('Server response status:', response.status);
                console.log('Server response:', data);

                if (!response.ok || data.error) {
                    // Show detailed error message
                    const transcriptSection = document.getElementById('transcriptSection');
                    const transcriptText = document.getElementById('transcriptText');

                    let errorHtml = `<strong>❌ Error:</strong><br>${data.error || 'Unknown error occurred'}`;
                    if (data.help) {
                        errorHtml += `<br><br><strong>💡 Help:</strong><br>${data.help}`;
                    }
                    if (data.debug && data.debug.trim()) {
                        errorHtml += `<br><br><strong>Debug Info:</strong><br><code style="display:block; white-space:pre-wrap; word-break:break-word;">${data.debug}</code>`;
                    }

                    transcriptText.innerHTML = errorHtml;
                    transcriptSection.classList.add('show');
                    voiceStatus.textContent = '❌ ' + (data.error || 'Processing failed');
                } else if (data.transcript) {
                    // Show successful transcript
                    const transcriptSection = document.getElementById('transcriptSection');
                    const transcriptText = document.getElementById('transcriptText');

                    let transcriptHtml = `<strong>📝 Transcript:</strong><br>${data.transcript}`;
                    if (data.route) {
                        transcriptHtml += `<br><br><strong>Routing Strategy:</strong> ${data.route.strategy || 'unknown'}`;
                        transcriptHtml += `<br><strong>Function Calls:</strong> ${data.route.function_calls?.length || 0}`;
                        if (data.route.source) {
                            transcriptHtml += `<br><strong>Source:</strong> ${data.route.source}`;
                        }
                    }

                    transcriptText.innerHTML = transcriptHtml;
                    transcriptSection.classList.add('show');
                    voiceStatus.innerHTML = `✅ Processed successfully`;
                } else {
                    voiceStatus.textContent = '⚠️ No transcript returned';
                }
            } catch (error) {
                const transcriptSection = document.getElementById('transcriptSection');
                const transcriptText = document.getElementById('transcriptText');

                transcriptText.innerHTML = `<strong>❌ Network Error:</strong><br>${error.message}<br><br><code>${error.stack}</code>`;
                transcriptSection.classList.add('show');
                voiceStatus.textContent = '❌ Error: ' + error.message;
                console.error('Error processing voice:', error);
            } finally {
                loading.classList.remove('show');
            }
        }

        function clearVoiceOutput() {
            document.getElementById('transcriptSection').classList.remove('show');
            document.getElementById('voiceTextInput').value = '';
            voiceStatus.textContent = 'Click the microphone to start recording';
            audioChunks = [];
        }

        async function processVoiceText() {
            const text = document.getElementById('voiceTextInput').value.trim();
            if (!text) {
                alert('Please enter a command to simulate voice');
                return;
            }

            loading.classList.add('show');

            try {
                const response = await fetch('/api/execute', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text })
                });

                const data = await response.json();

                // Show successful transcript with routing info
                const transcriptSection = document.getElementById('transcriptSection');
                const transcriptText = document.getElementById('transcriptText');

                let transcriptHtml = `<strong>📝 Simulated Transcript:</strong><br>${data.input}`;

                if (data.route) {
                    transcriptHtml += `<br><br><strong>🎯 Routing Strategy:</strong> ${data.route.strategy || 'unknown'}`;
                    transcriptHtml += `<br><strong>Source:</strong> ${data.route.source || 'unknown'}`;
                    transcriptHtml += `<br><strong>Function Calls:</strong> ${data.route.function_calls?.length || 0}`;

                    if (data.route.function_calls && data.route.function_calls.length > 0) {
                        transcriptHtml += `<br><br><strong>Actions Triggered:</strong>`;
                        data.route.function_calls.forEach(call => {
                            transcriptHtml += `<br>  • <strong>${call.name}</strong>`;
                            if (call.arguments && Object.keys(call.arguments).length > 0) {
                                transcriptHtml += `: ${JSON.stringify(call.arguments)}`;
                            }
                        });
                    }
                }

                transcriptText.innerHTML = transcriptHtml;
                transcriptSection.classList.add('show');
                voiceStatus.innerHTML = `✅ Processed (Test Mode)`;
            } catch (error) {
                const transcriptSection = document.getElementById('transcriptSection');
                const transcriptText = document.getElementById('transcriptText');

                transcriptText.innerHTML = `<strong>❌ Error:</strong><br>${error.message}`;
                transcriptSection.classList.add('show');
                voiceStatus.textContent = '❌ Error: ' + error.message;
                console.error('Error processing voice text:', error);
            } finally {
                loading.classList.remove('show');
            }
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/execute', methods=['POST'])
def execute():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        result = run_text_command(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/voice', methods=['POST'])
def voice():
    """
    Handle voice recording: transcribe audio and route through hybrid system
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({'error': 'No audio file selected'}), 400

        # Save audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            audio_file.save(tmp.name)
            tmp_path = tmp.name

        normalized_path = None
        try:
            # Normalize audio to proper WAV format
            print(f"Original audio file: {tmp_path}")
            normalized_path = normalize_audio(tmp_path)

            # Use Cactus Whisper model - path to the model weights
            whisper_model_path = os.environ.get("WHISPER_MODEL_PATH", "weights/whisper-small")

            # Try to transcribe using voice_action with Cactus
            try:
                result = run_voice_command(normalized_path, whisper_model_path)

                # Check if transcript is empty
                if not result.get('transcript', '').strip():
                    return jsonify({
                        'error': 'No speech detected in audio. Please ensure you speak clearly and loudly.',
                        'transcript': '',
                        'audio': result.get('audio'),
                        'debug': 'Empty transcript returned from transcription'
                    }), 400

                return jsonify(result)
            except Exception as transcribe_error:
                # Fallback: provide helpful error message
                error_msg = str(transcribe_error)
                print(f"Transcription error: {error_msg}")
                if 'whisper' in error_msg.lower() or 'transcribe' in error_msg.lower():
                    return jsonify({
                        'error': 'Whisper transcription model not available. Please ensure the Cactus Whisper model is installed.',
                        'debug': error_msg,
                        'help': 'Set WHISPER_MODEL_PATH to point to your Cactus Whisper model directory (e.g., weights/whisper-small).'
                    }), 400
                raise
        finally:
            # Clean up temporary files
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            if normalized_path and normalized_path != tmp_path and os.path.exists(normalized_path):
                os.remove(normalized_path)

    except Exception as e:
        return jsonify({
            'error': f'Voice processing failed: {str(e)}',
            'debug': str(e)
        }), 500

if __name__ == '__main__':
    print("Starting FunctionGemma Product Demo Server...")
    print("Open your browser to: http://localhost:8000")
    app.run(debug=True, port=8000, host='0.0.0.0')
