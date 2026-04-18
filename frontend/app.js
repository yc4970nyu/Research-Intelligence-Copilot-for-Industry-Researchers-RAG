const API = "http://localhost:8000";

// ---- DOM refs ----
const dropZone     = document.getElementById("dropZone");
const fileInput    = document.getElementById("fileInput");
const uploadStatus = document.getElementById("uploadStatus");
const ingestBtn    = document.getElementById("ingestBtn");
const refreshBtn   = document.getElementById("refreshBtn");
const indexStats   = document.getElementById("indexStats");
const chatMessages = document.getElementById("chatMessages");
const questionInput = document.getElementById("questionInput");
const sendBtn      = document.getElementById("sendBtn");

let selectedFiles = [];

// ---- file selection ----

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropZone.classList.add("drag-over");
});

dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));

dropZone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropZone.classList.remove("drag-over");
  handleFiles(Array.from(e.dataTransfer.files).filter(f => f.name.endsWith(".pdf")));
});

fileInput.addEventListener("change", () => {
  handleFiles(Array.from(fileInput.files));
});

function handleFiles(files) {
  if (!files.length) return;
  selectedFiles = files;
  uploadStatus.classList.remove("hidden");
  uploadStatus.textContent = files.map(f => f.name).join("\n");
  ingestBtn.disabled = false;
}

// ---- ingest ----

ingestBtn.addEventListener("click", async () => {
  if (!selectedFiles.length) return;

  ingestBtn.classList.add("loading");
  ingestBtn.disabled = true;
  ingestBtn.textContent = "Ingesting...";

  const formData = new FormData();
  selectedFiles.forEach(f => formData.append("files", f));

  try {
    const resp = await fetch(`${API}/ingest`, { method: "POST", body: formData });
    const data = await resp.json();

    if (!resp.ok) {
      uploadStatus.textContent = "Error: " + (data.detail || "unknown error");
    } else {
      uploadStatus.textContent = data.message + ` (${data.total_chunks} chunks)`;
      selectedFiles = [];
      fileInput.value = "";
      loadIndexStats();
    }
  } catch (err) {
    uploadStatus.textContent = "Network error: " + err.message;
  }

  ingestBtn.textContent = "Ingest selected files";
  ingestBtn.classList.remove("loading");
  ingestBtn.disabled = true;
});

// ---- index stats ----

async function loadIndexStats() {
  try {
    const resp = await fetch(`${API}/index/stats`);
    const data = await resp.json();

    if (data.total_chunks === 0) {
      indexStats.innerHTML = '<span class="muted">No documents yet</span>';
      return;
    }

    const docs = data.documents;
    indexStats.innerHTML = Object.entries(docs).map(([name, count]) => `
      <div class="index-doc-item">
        <span class="index-doc-name" title="${name}">${name}</span>
        <span class="index-doc-count">${count}</span>
      </div>
    `).join("");
  } catch (_) {
    indexStats.innerHTML = '<span class="muted">Could not load stats</span>';
  }
}

refreshBtn.addEventListener("click", loadIndexStats);
loadIndexStats();

// ---- chat ----

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

sendBtn.addEventListener("click", sendMessage);

// auto-resize textarea
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 150) + "px";
});

async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question) return;

  questionInput.value = "";
  questionInput.style.height = "auto";
  sendBtn.disabled = true;

  // remove welcome message if present
  const welcome = chatMessages.querySelector(".welcome-msg");
  if (welcome) welcome.remove();

  // add user bubble
  appendMessage("user", question);

  // add thinking indicator
  const thinkingEl = appendThinking();

  try {
    const resp = await fetch(`${API}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await resp.json();
    thinkingEl.remove();

    if (!resp.ok) {
      appendAssistantError(data.detail || "Something went wrong");
    } else {
      appendAssistantAnswer(data);
    }
  } catch (err) {
    thinkingEl.remove();
    appendAssistantError("Network error: " + err.message);
  }

  sendBtn.disabled = false;
  scrollToBottom();
}

function appendMessage(role, text) {
  const el = document.createElement("div");
  el.className = `message ${role}`;
  el.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatMessages.appendChild(el);
  scrollToBottom();
  return el;
}

function appendThinking() {
  const el = document.createElement("div");
  el.className = "thinking";
  el.innerHTML = `
    <div class="thinking-dots">
      <span></span><span></span><span></span>
    </div>
    <span>Searching documents...</span>
  `;
  chatMessages.appendChild(el);
  scrollToBottom();
  return el;
}

function appendAssistantAnswer(data) {
  const el = document.createElement("div");
  el.className = "message assistant";

  const insufficient = !data.sufficient_evidence && data.intent !== "chitchat" && data.intent !== "refusal";
  const bubbleClass = insufficient ? "bubble insufficient" : "bubble";

  // format inline citations [1] as styled spans
  const formattedAnswer = formatCitations(escapeHtml(data.answer));

  let html = `<div class="${bubbleClass}">${formattedAnswer}</div>`;

  // intent badge + rewritten query
  html += `<div class="meta-row">`;
  if (data.intent) {
    html += `<span class="intent-badge intent-${data.intent}">${data.intent}</span>`;
  }
  if (data.rewritten_query && data.rewritten_query !== data.question) {
    html += `<span class="rewritten-query">searched: "${escapeHtml(data.rewritten_query)}"</span>`;
  }
  html += `</div>`;

  // citations
  if (data.citations && data.citations.length > 0) {
    html += `<div class="citations">`;
    html += `<div class="citations-label">Sources</div>`;
    data.citations.forEach((c, i) => {
      html += `
        <div class="citation-card" onclick="this.classList.toggle('expanded')">
          <div class="citation-header">
            <span class="citation-source">[${i+1}] ${escapeHtml(c.filename)}, p.${c.page}</span>
            <span class="citation-score">${(c.score * 100).toFixed(0)}%</span>
          </div>
          <div class="citation-text">${escapeHtml(c.text.slice(0, 280))}</div>
        </div>
      `;
    });
    html += `</div>`;
  }

  el.innerHTML = html;
  chatMessages.appendChild(el);
  scrollToBottom();
}

function appendAssistantError(msg) {
  const el = document.createElement("div");
  el.className = "message assistant";
  el.innerHTML = `<div class="bubble error">${escapeHtml(msg)}</div>`;
  chatMessages.appendChild(el);
  scrollToBottom();
}

// ---- helpers ----

function formatCitations(text) {
  // turn [1], [2] etc into styled inline badges
  return text.replace(/\[(\d+)\]/g, '<span style="color:#a78bfa;font-weight:600">[$1]</span>');
}

function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/\n/g, "<br>");
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}
