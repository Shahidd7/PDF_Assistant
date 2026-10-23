const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("question");
const sendBtn = document.getElementById("send");
const fileInput = document.getElementById("file");
const uploadBtn = document.getElementById("upload-btn");
const statusEl = document.getElementById("status");

let docId = null;   // set after a successful upload

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.className = isError ? "error" : "";
}

function addMessage(role, text = "") {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
  return div;
}

function formatAnswer(text) {
  return DOMPurify.sanitize(marked.parse(text, { breaks: true }));
}

function renderAnswer(div, data) {
  div.className = "msg bot";
  div.innerHTML = formatAnswer(data.answer);

  const pages = document.createElement("div");
  pages.className = "pages";
  pages.textContent = "Pages retrieved: " + data.pages.join(", ");
  div.appendChild(pages);

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Show source snippets";
  details.appendChild(summary);

  data.sources.forEach((s) => {
    const p = document.createElement("div");
    p.className = "source";
    p.textContent = `[p. ${s.page}] ${s.snippet}…`;   // textContent = safe
    details.appendChild(p);
  });
  div.appendChild(details);
}

async function readError(res) {
  try {
    const err = await res.json();
    if (typeof err.detail === "string") return err.detail;
  } catch (_) {}
  return `Request failed (${res.status}).`;
}

/* ---------- Upload ---------- */
uploadBtn.addEventListener("click", async () => {
  const file = fileInput.files[0];
  if (!file) {
    setStatus("Choose a PDF first.", true);
    return;
  }

  uploadBtn.disabled = true;
  input.disabled = true;
  sendBtn.disabled = true;
  docId = null;
  setStatus("Processing… this can take a minute for large files.");

  const body = new FormData();
  body.append("file", file);   // do NOT set Content-Type; the browser adds the boundary

  try {
    const res = await fetch("/api/upload", { method: "POST", body });
    if (!res.ok) {
      setStatus(await readError(res), true);
      return;
    }
    const data = await res.json();
    docId = data.doc_id;
    chat.innerHTML = "";
    setStatus(`Ready: ${data.pages} pages, ${data.chunks} chunks.`);
    input.disabled = false;
    sendBtn.disabled = false;
    input.placeholder = "Ask a question about the PDF…";
    input.focus();
  } catch (err) {
    setStatus("Can't reach the server. Is it running?", true);
  } finally {
    uploadBtn.disabled = false;
  }
});

/* ---------- Ask ---------- */
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question || !docId) return;

  addMessage("user", question);
  input.value = "";
  sendBtn.disabled = true;
  const pending = addMessage("bot loading", "Searching the document…");

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc_id: docId, question }),
    });

    if (!res.ok) {
      pending.className = "msg error";
      pending.textContent = await readError(res);
      return;
    }

    renderAnswer(pending, await res.json());
  } catch (err) {
    pending.className = "msg error";
    pending.textContent = "Can't reach the server. Is it running?";
  } finally {
    sendBtn.disabled = false;
    input.focus();
    chat.scrollTop = chat.scrollHeight;
  }
});