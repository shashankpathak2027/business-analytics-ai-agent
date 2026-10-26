/* =========================================================================
   ANALYTICA — Enterprise Business Analytics AI Agent
   Frontend logic for frontend/index.html
   ========================================================================= */

const API_BASE_URL = "http://localhost:8000";

let currentThreadId = null;
let isUploading = false;
let isAsking = false;

// ---------------------------------------------------------------------
// DOM REFERENCES
// ---------------------------------------------------------------------
const csvFileInput = document.getElementById("csvFileInput");
const fileDropText = document.getElementById("fileDropText");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");

const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const suggestedQuestions = document.getElementById("suggestedQuestions");

const loadingIndicator = document.getElementById("loadingIndicator");
const loadingText = document.getElementById("loadingText");

const conversationArea = document.getElementById("conversationArea");
const conversationEmptyState = document.getElementById("conversationEmptyState");

const chartsContainer = document.getElementById("chartsContainer");

const reportContent = document.getElementById("reportContent");
const reportEmptyState = document.getElementById("reportEmptyState");

const statusDot = document.getElementById("statusDot");
const statusLabel = document.getElementById("statusLabel");
const threadIdDisplay = document.getElementById("threadIdDisplay");

// ---------------------------------------------------------------------
// EVENT LISTENERS
// ---------------------------------------------------------------------
csvFileInput.addEventListener("change", handleFileSelected);
uploadBtn.addEventListener("click", handleUpload);
askBtn.addEventListener("click", handleAsk);

questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    handleAsk();
  }
});

suggestedQuestions.addEventListener("click", (event) => {
  const item = event.target.closest("li[data-question]");
  if (!item || askBtn.disabled) return;
  questionInput.value = item.getAttribute("data-question");
  handleAsk();
});

// ---------------------------------------------------------------------
// FILE SELECTION
// ---------------------------------------------------------------------
function handleFileSelected() {
  const file = csvFileInput.files[0];
  if (file) {
    fileDropText.textContent = file.name;
    uploadBtn.disabled = false;
  } else {
    fileDropText.textContent = "Choose a CSV file";
    uploadBtn.disabled = true;
  }
}

// ---------------------------------------------------------------------
// UPLOAD CSV
// ---------------------------------------------------------------------
async function handleUpload() {
  const file = csvFileInput.files[0];
  if (!file || isUploading) return;

  isUploading = true;
  uploadBtn.disabled = true;
  setUploadStatus("Uploading dataset…", null);
  showLoading("Uploading and profiling dataset…");
  setSessionStatus("loading", "Uploading dataset…");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API_BASE_URL}/api/upload`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Upload failed.");
    }

    currentThreadId = data.thread_id;
    threadIdDisplay.textContent = currentThreadId;

    setUploadStatus(`Loaded "${data.filename}" successfully.`, "success");
    setSessionStatus("active", `Dataset loaded: ${data.filename}`);

    questionInput.disabled = false;
    askBtn.disabled = false;

    addMessage("agent", `Dataset "${data.filename}" uploaded. Ask a question to begin the analysis.`);
  } catch (error) {
    setUploadStatus(error.message || "Something went wrong while uploading.", "error");
    setSessionStatus("idle", "No dataset loaded");
  } finally {
    isUploading = false;
    uploadBtn.disabled = false;
    hideLoading();
  }
}

// ---------------------------------------------------------------------
// ASK A QUESTION
// ---------------------------------------------------------------------
async function handleAsk() {
  const question = questionInput.value.trim();
  if (!question || isAsking) return;

  if (!currentThreadId) {
    addMessage("agent", "Please upload a CSV dataset before asking a question.");
    return;
  }

  isAsking = true;
  askBtn.disabled = true;
  questionInput.disabled = true;

  addMessage("user", question);
  questionInput.value = "";

  showLoading("Analyzing your question…");
  setSessionStatus("loading", "Analyzing…");

  try {
    const response = await fetch(`${API_BASE_URL}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: currentThreadId, question }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "The agent could not process your question.");
    }

    if (data.insights) {
      addMessage("agent", data.insights);
    }

    renderCharts(data.charts);
    renderReport(data.final_report);

    setSessionStatus("active", "Dataset loaded — ready for follow-up questions");
  } catch (error) {
    addMessage("agent", `Error: ${error.message}`);
    setSessionStatus("active", "Dataset loaded — ready for follow-up questions");
  } finally {
    isAsking = false;
    askBtn.disabled = false;
    questionInput.disabled = false;
    hideLoading();
    questionInput.focus();
  }
}

// ---------------------------------------------------------------------
// CONVERSATION RENDERING
// ---------------------------------------------------------------------
function addMessage(role, text) {
  conversationEmptyState?.remove();

  const messageEl = document.createElement("div");
  messageEl.className = `message ${role}`;

  const roleLabel = document.createElement("span");
  roleLabel.className = "message-role";
  roleLabel.textContent = role === "user" ? "You" : "Analytica";

  const contentEl = document.createElement("span");
  contentEl.textContent = text;

  messageEl.appendChild(roleLabel);
  messageEl.appendChild(contentEl);
  conversationArea.appendChild(messageEl);

  conversationArea.scrollTop = conversationArea.scrollHeight;
}

// ---------------------------------------------------------------------
// CHARTS RENDERING
// ---------------------------------------------------------------------
// Charts are fully re-rendered on every response rather than appended, so
// a thread's chart panel always reflects exactly the charts belonging to
// its most recent analysis - never a growing pile of duplicates from every
// turn in the conversation.
function renderCharts(charts) {
  chartsContainer.innerHTML = "";

  if (!charts || charts.length === 0) {
    chartsContainer.innerHTML =
      '<div class="empty-state">Charts generated during analysis will appear here.</div>';
    return;
  }

  charts.forEach((chart) => {
    const card = document.createElement("div");
    card.className = "chart-card";

    const img = document.createElement("img");
    img.src = `${API_BASE_URL}${chart.url}`;
    img.alt = chart.title || "Generated chart";

    const caption = document.createElement("div");
    caption.className = "chart-card-caption";

    if (chart.title) {
      const titleEl = document.createElement("div");
      titleEl.className = "chart-card-title";
      titleEl.textContent = chart.title;
      caption.appendChild(titleEl);
    }

    if (chart.explanation) {
      const explanationEl = document.createElement("div");
      explanationEl.className = "chart-card-explanation";
      explanationEl.textContent = chart.explanation;
      caption.appendChild(explanationEl);
    }

    card.appendChild(img);
    card.appendChild(caption);
    chartsContainer.appendChild(card);
  });
}

// ---------------------------------------------------------------------
// REPORT RENDERING
// ---------------------------------------------------------------------
function renderReport(reportMarkdown) {
  if (!reportMarkdown) return;

  reportEmptyState?.remove();
  reportContent.innerHTML = markdownToHtml(reportMarkdown);
}

function markdownToHtml(markdown) {
  const escapeHtml = (text) =>
    text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

  const lines = escapeHtml(markdown).split("\n");
  const htmlParts = [];
  let inList = false;

  function closeList() {
    if (inList) {
      htmlParts.push("</ul>");
      inList = false;
    }
  }

  lines.forEach((rawLine) => {
    const line = rawLine.trim();

    if (line.startsWith("### ")) {
      closeList();
      htmlParts.push(`<h3>${inlineMarkdown(line.slice(4))}</h3>`);
    } else if (line.startsWith("## ")) {
      closeList();
      htmlParts.push(`<h2>${inlineMarkdown(line.slice(3))}</h2>`);
    } else if (line.startsWith("# ")) {
      closeList();
      htmlParts.push(`<h1>${inlineMarkdown(line.slice(2))}</h1>`);
    } else if (line.startsWith("- ") || line.startsWith("* ")) {
      if (!inList) {
        htmlParts.push("<ul>");
        inList = true;
      }
      htmlParts.push(`<li>${inlineMarkdown(line.slice(2))}</li>`);
    } else if (line.length === 0) {
      closeList();
    } else {
      closeList();
      htmlParts.push(`<p>${inlineMarkdown(line)}</p>`);
    }
  });

  closeList();
  return htmlParts.join("\n");
}

function inlineMarkdown(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/`(.+?)`/g, "<code>$1</code>");
}

// ---------------------------------------------------------------------
// UI STATE HELPERS
// ---------------------------------------------------------------------
function showLoading(message) {
  loadingText.textContent = message;
  loadingIndicator.hidden = false;
}

function hideLoading() {
  loadingIndicator.hidden = true;
}

function setUploadStatus(message, type) {
  uploadStatus.textContent = message;
  uploadStatus.classList.remove("is-success", "is-error");
  if (type === "success") {
    uploadStatus.classList.add("is-success");
  } else if (type === "error") {
    uploadStatus.classList.add("is-error");
  }
}

function setSessionStatus(state, label) {
  statusLabel.textContent = label;
  statusDot.classList.remove("is-active", "is-loading");
  if (state === "active") {
    statusDot.classList.add("is-active");
  } else if (state === "loading") {
    statusDot.classList.add("is-loading");
  }
}