const articleInput = document.getElementById("articleInput");
const wordCounter = document.getElementById("wordCounter");
const summarizeBtn = document.getElementById("summarizeBtn");
const errorMsg = document.getElementById("errorMsg");
const loading = document.getElementById("loading");
const summaryOutput = document.getElementById("summaryOutput");
const statsRow = document.getElementById("statsRow");

const historyBtn = document.getElementById("historyBtn");
const historyModal = document.getElementById("historyModal");
const closeHistory = document.getElementById("closeHistory");
const clearHistoryBtn = document.getElementById("clearHistory");
const historyList = document.getElementById("historyList");

function countWords(text) {
  const trimmed = text.trim();
  if (!trimmed) return 0;
  return trimmed.split(/\s+/).length;
}

articleInput.addEventListener("input", () => {
  wordCounter.textContent = `${countWords(articleInput.value)} words`;
});

function showError(msg) {
  errorMsg.textContent = msg;
  errorMsg.classList.remove("hidden");
}
function clearError() {
  errorMsg.classList.add("hidden");
  errorMsg.textContent = "";
}

summarizeBtn.addEventListener("click", async () => {
  clearError();
  const text = articleInput.value.trim();
  if (!text) {
    showError("Please paste some article text first.");
    return;
  }

  const maxLength = document.getElementById("maxLength").value;
  const minLength = document.getElementById("minLength").value;
  const saveHistory = document.getElementById("saveHistory").checked;

  loading.classList.remove("hidden");
  summaryOutput.innerHTML = "";
  statsRow.classList.add("hidden");
  summarizeBtn.disabled = true;

  try {
    const res = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        text,
        max_length: maxLength,
        min_length: minLength,
        save_history: saveHistory,
      }),
    });
    const data = await res.json();

    if (!res.ok) {
      showError(data.error || "Something went wrong.");
      summaryOutput.innerHTML = '<p class="placeholder">Your summary will appear here.</p>';
      return;
    }

    summaryOutput.innerHTML = `<p>${data.summary}</p>`;
    document.getElementById("statOriginal").textContent = data.original_word_count;
    document.getElementById("statSummary").textContent = data.summary_word_count;
    document.getElementById("statCompression").textContent = `${data.compression_percent}%`;
    document.getElementById("statReadability").textContent =
      `${data.readability_score} (${data.readability_label})`;
    statsRow.classList.remove("hidden");
  } catch (err) {
    showError("Network or server error: " + err.message);
  } finally {
    loading.classList.add("hidden");
    summarizeBtn.disabled = false;
  }
});

// ---- History ----
historyBtn.addEventListener("click", async () => {
  historyModal.classList.remove("hidden");
  historyList.innerHTML = "Loading...";
  try {
    const res = await fetch("/history");
    const items = await res.json();
    if (!items.length) {
      historyList.innerHTML = '<p class="placeholder">No history yet.</p>';
      return;
    }
    historyList.innerHTML = items
      .map(
        (item) => `
        <div class="history-item">
          <div class="ts">${new Date(item.timestamp).toLocaleString()}</div>
          <div><strong>Original:</strong> ${item.original_preview}</div>
          <div><strong>Summary:</strong> ${item.summary}</div>
          <div class="ts">${item.summary_word_count} words · Readability: ${item.readability_label}</div>
        </div>`
      )
      .join("");
  } catch (err) {
    historyList.innerHTML = `<p class="error">Failed to load history.</p>`;
  }
});

closeHistory.addEventListener("click", () => historyModal.classList.add("hidden"));

clearHistoryBtn.addEventListener("click", async () => {
  await fetch("/history/clear", { method: "POST" });
  historyList.innerHTML = '<p class="placeholder">No history yet.</p>';
});
