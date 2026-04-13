package main

func init() {
	uiHTML = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover">
    <title>NanoChat</title>
    <style>
        :root { color-scheme: light; }
        * { box-sizing: border-box; }
        html, body { height: 100%; margin: 0; }
        body {
            font-family: ui-sans-serif, -apple-system, system-ui, "Segoe UI", Helvetica, Arial, sans-serif;
            background-color: #ffffff; color: #111827;
            min-height: 100dvh; display: flex; flex-direction: column;
        }
        .header { background-color: #ffffff; padding: 1.25rem 1.5rem; }
        .header-left { display: flex; align-items: center; gap: 0.75rem; }
        .header h1 { font-size: 1.25rem; font-weight: 600; margin: 0; color: #111827; }
        .new-conversation-btn {
            width: 32px; height: 32px; padding: 0; border: 1px solid #e5e7eb;
            border-radius: 0.5rem; background-color: #ffffff; color: #6b7280;
            cursor: pointer; display: flex; align-items: center; justify-content: center;
            transition: all 0.2s ease;
        }
        .new-conversation-btn:hover { background-color: #f3f4f6; border-color: #d1d5db; color: #374151; }
        .chat-container { flex: 1; overflow-y: auto; background-color: #ffffff; }
        .chat-wrapper { max-width: 48rem; margin: 0 auto; padding: 2rem 1.5rem 3rem; display: flex; flex-direction: column; gap: 0.75rem; }
        .message { display: flex; justify-content: flex-start; margin-bottom: 0.5rem; color: #0d0d0d; }
        .message.assistant { justify-content: flex-start; }
        .message.user { justify-content: flex-end; }
        .message-content { white-space: pre-wrap; line-height: 1.6; max-width: 100%; }
        .message.assistant .message-content {
            background: transparent; border: none; cursor: pointer;
            border-radius: 0.5rem; padding: 0.5rem; margin-left: -0.5rem;
            transition: background-color 0.2s ease;
        }
        .message.assistant .message-content:hover { background-color: #f9fafb; }
        .message.user .message-content {
            background-color: #f3f4f6; border-radius: 1.25rem; padding: 0.8rem 1rem;
            max-width: 65%; cursor: pointer; transition: background-color 0.2s ease;
        }
        .message.user .message-content:hover { background-color: #e5e7eb; }
        .message.console .message-content {
            font-family: 'Monaco', 'Menlo', monospace; font-size: 0.875rem;
            background-color: #fafafa; padding: 0.75rem 1rem; color: #374151; max-width: 80%;
        }
        .input-container { background-color: #ffffff; padding: 1rem; padding-bottom: calc(1rem + env(safe-area-inset-bottom)); }
        .input-wrapper { max-width: 48rem; margin: 0 auto; display: flex; gap: 0.75rem; align-items: flex-end; }
        .chat-input {
            flex: 1; padding: 0.8rem 1rem; border: 1px solid #d1d5db; border-radius: 0.75rem;
            background-color: #ffffff; color: #111827; font-size: 1rem; line-height: 1.5;
            resize: none; outline: none; min-height: 54px; max-height: 200px;
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
        }
        .chat-input::placeholder { color: #9ca3af; }
        .chat-input:focus { border-color: #2563eb; box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1); }
        .send-button {
            flex-shrink: 0; padding: 0; width: 54px; height: 54px;
            border: 1px solid #111827; border-radius: 0.75rem;
            background-color: #111827; color: #ffffff;
            display: flex; align-items: center; justify-content: center;
            cursor: pointer; transition: background-color 0.2s ease;
        }
        .send-button:hover:not(:disabled) { background-color: #2563eb; border-color: #2563eb; }
        .send-button:disabled { cursor: not-allowed; border-color: #d1d5db; background-color: #e5e7eb; color: #9ca3af; }
        .typing-indicator { display: inline-block; color: #6b7280; letter-spacing: 0.15em; }
        .typing-indicator::after { content: '···'; animation: typing 1.4s infinite; }
        @keyframes typing { 0%, 60%, 100% { opacity: 0.2; } 30% { opacity: 1; } }
        .error-message { background-color: #fee2e2; border: 1px solid #fecaca; color: #b91c1c; padding: 0.75rem 1rem; border-radius: 0.75rem; }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-left">
            <button class="new-conversation-btn" onclick="newConversation()" title="New Conversation">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 5v14"></path><path d="M5 12h14"></path></svg>
            </button>
            <h1>nanochat <small style="color:#9ca3af;font-weight:400">(go)</small></h1>
        </div>
    </div>
    <div class="chat-container" id="chatContainer">
        <div class="chat-wrapper" id="chatWrapper"></div>
    </div>
    <div class="input-container">
        <div class="input-wrapper">
            <textarea id="chatInput" class="chat-input" placeholder="Ask anything" rows="1" onkeydown="handleKeyDown(event)"></textarea>
            <button id="sendButton" class="send-button" onclick="sendMessage()" disabled>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 2L11 13"></path><path d="M22 2l-7 20-4-9-9-4 20-7z"></path></svg>
            </button>
        </div>
    </div>
    <script>
        const API_URL = '';
        const chatContainer = document.getElementById('chatContainer');
        const chatWrapper = document.getElementById('chatWrapper');
        const chatInput = document.getElementById('chatInput');
        const sendButton = document.getElementById('sendButton');
        let messages = [];
        let isGenerating = false;
        let currentTemperature = 0.8;
        let currentTopK = 50;

        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
            sendButton.disabled = !this.value.trim() || isGenerating;
        });

        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }
        }

        function newConversation() {
            messages = []; chatWrapper.innerHTML = ''; chatInput.value = '';
            chatInput.style.height = 'auto'; sendButton.disabled = false;
            isGenerating = false; chatInput.focus();
        }

        function addMessage(role, content, messageIndex) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + role;
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            if (role === 'user' && messageIndex !== undefined) {
                contentDiv.setAttribute('title', 'Click to edit');
                contentDiv.addEventListener('click', function() { if (!isGenerating) editMessage(messageIndex); });
            }
            if (role === 'assistant' && messageIndex !== undefined) {
                contentDiv.setAttribute('title', 'Click to regenerate');
                contentDiv.addEventListener('click', function() { if (!isGenerating) regenerateMessage(messageIndex); });
            }
            messageDiv.appendChild(contentDiv);
            chatWrapper.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
            return contentDiv;
        }

        function editMessage(idx) {
            if (idx < 0 || idx >= messages.length) return;
            chatInput.value = messages[idx].content;
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
            messages = messages.slice(0, idx);
            const all = chatWrapper.querySelectorAll('.message');
            for (let i = idx; i < all.length; i++) all[i].remove();
            sendButton.disabled = false; chatInput.focus();
        }

        async function generateAssistantResponse() {
            isGenerating = true; sendButton.disabled = true;
            const assistantContent = addMessage('assistant', '');
            assistantContent.innerHTML = '<span class="typing-indicator"></span>';
            try {
                const response = await fetch(API_URL + '/chat/completions', {
                    method: 'POST', headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({ messages, temperature: currentTemperature, top_k: currentTopK, max_tokens: 512 }),
                });
                if (!response.ok) throw new Error('HTTP ' + response.status);
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let fullResponse = '';
                assistantContent.textContent = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    const chunk = decoder.decode(value);
                    for (const line of chunk.split('\n')) {
                        if (line.startsWith('data: ')) {
                            try {
                                const data = JSON.parse(line.slice(6));
                                if (data.token) {
                                    fullResponse += data.token;
                                    assistantContent.textContent = fullResponse;
                                    chatContainer.scrollTop = chatContainer.scrollHeight;
                                }
                            } catch(e) {}
                        }
                    }
                }
                const idx = messages.length;
                messages.push({ role: 'assistant', content: fullResponse });
                assistantContent.setAttribute('title', 'Click to regenerate');
                assistantContent.addEventListener('click', function() { if (!isGenerating) regenerateMessage(idx); });
            } catch (error) {
                assistantContent.innerHTML = '<div class="error-message">Error: ' + error.message + '</div>';
            } finally {
                isGenerating = false; sendButton.disabled = !chatInput.value.trim();
            }
        }

        async function regenerateMessage(idx) {
            if (idx < 0 || idx >= messages.length) return;
            messages = messages.slice(0, idx);
            const all = chatWrapper.querySelectorAll('.message');
            for (let i = idx; i < all.length; i++) all[i].remove();
            await generateAssistantResponse();
        }

        function handleSlashCommand(cmd) {
            const parts = cmd.trim().split(/\s+/);
            const c = parts[0].toLowerCase();
            const arg = parts[1];
            if (c === '/temperature') {
                if (!arg) { addMessage('console', 'Temperature: ' + currentTemperature); return true; }
                const t = parseFloat(arg);
                if (isNaN(t) || t < 0 || t > 2) { addMessage('console', 'Invalid (0.0-2.0)'); return true; }
                currentTemperature = t; addMessage('console', 'Temperature: ' + t); return true;
            } else if (c === '/topk') {
                if (!arg) { addMessage('console', 'Top-k: ' + currentTopK); return true; }
                const k = parseInt(arg);
                if (isNaN(k) || k < 1 || k > 200) { addMessage('console', 'Invalid (1-200)'); return true; }
                currentTopK = k; addMessage('console', 'Top-k: ' + k); return true;
            } else if (c === '/clear') { newConversation(); return true; }
            else if (c === '/help') {
                addMessage('console', '/temperature [val]  /topk [val]  /clear  /help');
                return true;
            }
            return false;
        }

        async function sendMessage() {
            const message = chatInput.value.trim();
            if (!message || isGenerating) return;
            if (message.startsWith('/')) { chatInput.value = ''; chatInput.style.height = 'auto'; handleSlashCommand(message); return; }
            chatInput.value = ''; chatInput.style.height = 'auto';
            const idx = messages.length;
            messages.push({ role: 'user', content: message });
            addMessage('user', message, idx);
            await generateAssistantResponse();
        }

        sendButton.disabled = false;
        chatInput.focus();
        fetch(API_URL + '/health').then(r => r.json()).then(d => console.log('Engine:', d))
            .catch(e => { console.error(e); chatWrapper.innerHTML = '<div class="error-message">Engine not running.</div>'; });
    </script>
</body>
</html>`
}
