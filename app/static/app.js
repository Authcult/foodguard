/**
 * 食鉴 FoodGuard — 前端逻辑
 *
 * SSE 流式接收思考过程和回答，处理步骤状态提示
 */

// marked.js 配置
marked.setOptions({
    breaks: true,
    gfm: true,
});

// ============================================================
// 全局状态
// ============================================================
const state = {
    sessionId: null,
    messages: [],
    isGenerating: false,
    allergens: [],
    // 流式渲染 buffer
    _thinkBuf: '',
    _answerBuf: '',
};

const $ = (sel) => document.querySelector(sel);
const chatContainer = $('#chatContainer');
const welcome = $('#welcome');
const userInput = $('#userInput');
const btnSend = $('#btnSend');
const btnClear = $('#btnClear');
const btnMenu = $('#btnMenu');
const btnUpload = $('#btnUpload');
const fileInput = $('#fileInput');
const sidebar = $('#sidebar');
const statusBar = $('#statusBar');
const stepIntent = $('#stepIntent');
const stepQuery = $('#stepQuery');
const stepGenerate = $('#stepGenerate');
const allergenList = $('#allergenList');

// ============================================================
// 初始化
// ============================================================
async function init() {
    const resp = await fetch('/api/session', { method: 'POST' });
    const data = await resp.json();
    state.sessionId = data.session_id;
    await loadProfile();
    renderAllergens();
    bindEvents();
}

async function loadProfile() {
    try {
        const resp = await fetch('/api/profile');
        const profile = await resp.json();
        state.allergens = profile.known_allergens || [];
    } catch (e) {
        console.error('加载用户画像失败:', e);
    }
}

// ============================================================
// 事件绑定
// ============================================================
function bindEvents() {
    btnSend.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    userInput.addEventListener('input', autoResize);
    btnClear.addEventListener('click', clearChat);
    btnMenu.addEventListener('click', () => sidebar.classList.toggle('open'));
    btnUpload.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleImageUpload);
    document.querySelectorAll('.example-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            userInput.value = btn.dataset.msg;
            autoResize();
            sendMessage();
        });
    });
}

function autoResize() {
    userInput.style.height = 'auto';
    userInput.style.height = Math.min(userInput.scrollHeight, 150) + 'px';
}

// ============================================================
// 过敏原管理
// ============================================================
const COMMON_ALLERGENS = [
    '花生', '牛奶（乳制品）', '鸡蛋', '大豆',
    '小麦（麸质）', '坚果', '鱼类及海鲜', '甲壳类（虾蟹）',
    '芝麻', '芹菜', '苯丙氨酸', '二氧化硫及亚硫酸盐',
];

function renderAllergens() {
    allergenList.innerHTML = COMMON_ALLERGENS.map(name => {
        const checked = state.allergens.includes(name) ? 'checked' : '';
        return `
            <label class="allergen-item">
                <input type="checkbox" value="${name}" ${checked}>
                <span class="allergen-check"></span>
                <span class="allergen-label">${name}</span>
            </label>
        `;
    }).join('');

    allergenList.querySelectorAll('input').forEach(cb => {
        cb.addEventListener('change', async () => {
            state.allergens = Array.from(allergenList.querySelectorAll('input:checked'))
                .map(el => el.value);
            await saveProfile();
        });
    });
}

async function saveProfile() {
    try {
        await fetch('/api/profile', {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                known_allergens: state.allergens,
                dietary_preferences: [],
                family_members: [],
            }),
        });
    } catch (e) {
        console.error('保存失败:', e);
    }
}

// ============================================================
// 消息发送与 SSE 流式接收
// ============================================================
async function sendMessage() {
    const text = userInput.value.trim();
    if (!text || state.isGenerating) return;

    welcome.classList.add('hidden');
    state.messages.push({ role: 'user', content: text });
    renderUserMessage(text);
    userInput.value = '';
    autoResize();

    state.isGenerating = true;
    btnSend.disabled = true;

    const { thinkingEl, thinkingContentEl, answerEl } = renderAssistantPlaceholder();
    scrollToBottom();
    showStatus();

    let fullThinking = '';
    let fullAnswer = '';
    state._thinkBuf = '';
    state._answerBuf = '';

    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, session_id: state.sessionId }),
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let currentEvent = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ') && currentEvent) {
                    const raw = line.slice(6);
                    try {
                        const data = JSON.parse(raw);
                        handleEvent(currentEvent, data, {
                            thinkingEl, thinkingContentEl, answerEl
                        });
                        if (currentEvent === 'thinking') fullThinking += data.token;
                        if (currentEvent === 'answer') fullAnswer += data.token;
                    } catch (e) { /* ignore non-JSON */ }
                    currentEvent = '';
                }
            }
        }

        // 最终渲染一次完整 markdown
        if (fullAnswer) {
            answerEl.innerHTML = marked.parse(fullAnswer);
        }

        state.messages.push({ role: 'assistant', content: fullAnswer, thinking: fullThinking });

    } catch (e) {
        console.error('请求失败:', e);
        answerEl.textContent = '请求失败，请重试。';
    } finally {
        state.isGenerating = false;
        btnSend.disabled = false;
        hideStatus();
        document.querySelectorAll('.cursor').forEach(c => c.remove());
        scrollToBottom();
    }
}

function handleEvent(type, data, els) {
    switch (type) {
        case 'status':
            updateStatus(data.step, data.detail);
            break;

        case 'thinking':
            els.thinkingEl.style.display = 'block';
            removeDots(els.thinkingContentEl);
            state._thinkBuf += data.token;
            els.thinkingContentEl.innerHTML = marked.parse(state._thinkBuf) + '<span class="cursor"></span>';
            scrollToBottom();
            break;

        case 'answer':
            removeDots(els.answerEl);
            removeCursor(els.thinkingContentEl);
            state._answerBuf += data.token;
            els.answerEl.innerHTML = marked.parse(state._answerBuf) + '<span class="cursor"></span>';
            scrollToBottom();
            break;

        case 'done':
            break;

        case 'error':
            removeDots(els.answerEl);
            els.answerEl.textContent = data.message || '发生错误';
            break;
    }
}

// ============================================================
// 渲染函数
// ============================================================

function renderUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `
        <div class="msg-avatar">👤</div>
        <div class="msg-body">
            <div class="msg-content">${escapeHtml(text)}</div>
        </div>
    `;
    chatContainer.appendChild(div);
    scrollToBottom();
}

function renderHistoryMessage(msg) {
    const div = document.createElement('div');
    div.className = `message ${msg.role}`;

    let thinkingHtml = '';
    if (msg.thinking) {
        thinkingHtml = `
            <div class="thinking-block">
                <div class="thinking-toggle">
                    <span class="arrow">▶</span>
                    <span>💭 思考过程</span>
                </div>
                <div class="thinking-content">${marked.parse(msg.thinking)}</div>
            </div>
        `;
    }

    const avatar = msg.role === 'user' ? '👤' : '🛡️';
    const contentHtml = msg.role === 'assistant' ? marked.parse(msg.content) : escapeHtml(msg.content);

    div.innerHTML = `
        <div class="msg-avatar">${avatar}</div>
        <div class="msg-body">
            ${thinkingHtml}
            <div class="msg-content">${contentHtml}</div>
        </div>
    `;

    // 折叠切换
    const toggle = div.querySelector('.thinking-toggle');
    const content = div.querySelector('.thinking-content');
    if (toggle && content) {
        toggle.addEventListener('click', () => {
            toggle.classList.toggle('open');
            content.classList.toggle('open');
        });
    }

    chatContainer.appendChild(div);
}

function renderAssistantPlaceholder() {
    const div = document.createElement('div');
    div.className = 'message assistant';

    const thinkingEl = document.createElement('div');
    thinkingEl.className = 'thinking-block';
    thinkingEl.style.display = 'none';
    thinkingEl.innerHTML = `
        <div class="thinking-toggle open">
            <span class="arrow">▶</span>
            <span>💭 思考过程</span>
        </div>
        <div class="thinking-content open"></div>
    `;

    const toggle = thinkingEl.querySelector('.thinking-toggle');
    const content = thinkingEl.querySelector('.thinking-content');
    toggle.addEventListener('click', () => {
        toggle.classList.toggle('open');
        content.classList.toggle('open');
    });

    const answerEl = document.createElement('div');
    answerEl.className = 'msg-content';
    answerEl.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';

    const bodyEl = document.createElement('div');
    bodyEl.className = 'msg-body';
    bodyEl.appendChild(thinkingEl);
    bodyEl.appendChild(answerEl);

    div.innerHTML = '<div class="msg-avatar">🛡️</div>';
    div.appendChild(bodyEl);
    chatContainer.appendChild(div);
    scrollToBottom();

    return { thinkingEl, thinkingContentEl: content, answerEl };
}

function appendText(el, text) {
    removeCursor(el);
    el.appendChild(document.createTextNode(text));
    const cursor = document.createElement('span');
    cursor.className = 'cursor';
    el.appendChild(cursor);
}

function removeDots(el) {
    const dots = el.querySelector('.loading-dots');
    if (dots) dots.remove();
}

function removeCursor(el) {
    const c = el.querySelector('.cursor');
    if (c) c.remove();
}

// ============================================================
// 状态条
// ============================================================

function showStatus() {
    statusBar.classList.add('active');
    [stepIntent, stepQuery, stepGenerate].forEach(s => s.classList.remove('active', 'done'));
}

function hideStatus() {
    statusBar.classList.remove('active');
}

function updateStatus(step) {
    [stepIntent, stepQuery, stepGenerate].forEach(s => s.classList.remove('active'));

    if (step.includes('意图') || step.includes('识别')) {
        stepIntent.classList.add('active');
    } else if (step.includes('知识库') || step.includes('查询')) {
        stepIntent.classList.add('done');
        stepQuery.classList.add('active');
    } else if (step.includes('生成') || step.includes('回答') || step.includes('输出')) {
        stepIntent.classList.add('done');
        stepQuery.classList.add('done');
        stepGenerate.classList.add('active');
    }
}

// ============================================================
// 图片上传 + OCR
// ============================================================

async function handleImageUpload(e) {
    const file = e.target.files[0];
    if (!file) return;

    // 清空 file input 允许重复上传同一文件
    fileInput.value = '';

    welcome.classList.add('hidden');

    // 显示用户上传的图片消息
    const imgUrl = URL.createObjectURL(file);
    state.messages.push({ role: 'user', content: `📷 上传了配料表图片: ${file.name}` });
    renderImageMessage(imgUrl, file.name);

    // 显示 OCR 处理状态
    const { thinkingEl, thinkingContentEl, answerEl } = renderAssistantPlaceholder();
    answerEl.innerHTML = '<div class="loading-dots"><span></span><span></span><span></span></div>';
    scrollToBottom();

    try {
        // 调用 OCR 接口
        const formData = new FormData();
        formData.append('file', file);

        const resp = await fetch('/api/ocr', { method: 'POST', body: formData });
        const data = await resp.json();

        if (data.error) {
            answerEl.textContent = `OCR 识别失败: ${data.error}`;
            return;
        }

        const ocrText = data.text;
        if (!ocrText) {
            answerEl.textContent = '未能从图片中识别出文字，请确保图片清晰。';
            return;
        }

        // 显示识别结果
        answerEl.innerHTML = marked.parse(`**📷 OCR 识别结果：**\n\n${ocrText}\n\n---\n\n正在为您分析配料表...`);

        // 自动将识别文本送入对话分析
        state.messages.push({ role: 'assistant', content: ocrText, thinking: '' });

        // 用识别出的文本调用分析接口
        state.isGenerating = true;
        btnSend.disabled = true;
        showStatus();

        state._thinkBuf = '';
        state._answerBuf = '';

        const analysisResp = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: `请帮我分析这个配料表：\n${ocrText}`,
                session_id: state.sessionId,
            }),
        });

        // 创建新的助手消息用于显示分析结果
        answerEl.innerHTML = '';
        const analysisResult = { thinkingEl, thinkingContentEl, answerEl };

        const reader = analysisResp.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let currentEvent = '';
        let fullThinking = '';
        let fullAnswer = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';

            for (const line of lines) {
                if (line.startsWith('event: ')) {
                    currentEvent = line.slice(7).trim();
                } else if (line.startsWith('data: ') && currentEvent) {
                    const raw = line.slice(6);
                    try {
                        const parsed = JSON.parse(raw);
                        handleEvent(currentEvent, parsed, analysisResult);
                        if (currentEvent === 'thinking') fullThinking += parsed.token;
                        if (currentEvent === 'answer') fullAnswer += parsed.token;
                    } catch (e) {}
                    currentEvent = '';
                }
            }
        }

        if (fullAnswer) {
            answerEl.innerHTML = marked.parse(fullAnswer);
        }
        state.messages.push({ role: 'assistant', content: fullAnswer, thinking: fullThinking });

    } catch (e) {
        console.error('OCR 失败:', e);
        answerEl.textContent = '图片识别失败，请重试。';
    } finally {
        state.isGenerating = false;
        btnSend.disabled = false;
        hideStatus();
        document.querySelectorAll('.cursor').forEach(c => c.remove());
        scrollToBottom();
    }
}

function renderImageMessage(imgUrl, filename) {
    const div = document.createElement('div');
    div.className = 'message user';
    div.innerHTML = `
        <div class="msg-avatar">👤</div>
        <div class="msg-body">
            <div class="msg-content">
                <div class="img-preview">
                    <img src="${imgUrl}" alt="${filename}">
                </div>
                <span class="img-filename">📷 ${escapeHtml(filename)}</span>
            </div>
        </div>
    `;
    chatContainer.appendChild(div);
    scrollToBottom();
}

// ============================================================
// 清除对话
// ============================================================

async function clearChat() {
    if (state.isGenerating) return;
    state.messages = [];
    chatContainer.innerHTML = '';
    welcome.classList.remove('hidden');
    chatContainer.appendChild(welcome);
    const resp = await fetch('/api/session', { method: 'POST' });
    state.sessionId = (await resp.json()).session_id;
}

// ============================================================
// 工具函数
// ============================================================

function scrollToBottom() {
    requestAnimationFrame(() => {
        chatContainer.scrollTop = chatContainer.scrollHeight;
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

init();
