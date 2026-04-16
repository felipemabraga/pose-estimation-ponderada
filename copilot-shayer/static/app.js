const messages = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const sendButton = document.getElementById("sendButton");
const fileInput = document.getElementById("fileInput");
const documentStatus = document.getElementById("documentStatus");
const clearDocumentButton = document.getElementById("clearDocumentButton");
const apiKeyInput = document.getElementById("apiKeyInput");

function addMessage(role, content) {
  const article = document.createElement("article");
  article.className = `message ${role}`;

  const roleLabel = document.createElement("div");
  roleLabel.className = "message-role";
  roleLabel.textContent = role === "user" ? "You" : "Assistant";

  const body = document.createElement("div");
  body.className = "message-content";
  body.textContent = content;

  article.append(roleLabel, body);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  fileInput.disabled = isBusy;
  clearDocumentButton.disabled = isBusy;
  sendButton.textContent = isBusy ? "Thinking..." : "Send";
}

async function refreshDocumentStatus() {
  const response = await fetch("/api/document");
  const data = await response.json();
  if (data.loaded) {
    documentStatus.textContent = `Document loaded: ${data.filename}`;
  } else {
    documentStatus.textContent = "No document loaded";
  }
}

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch("/api/upload", {
    method: "POST",
    body: formData,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Upload failed.");
  }

  addMessage("assistant", `Document loaded: ${data.filename}`);
  await refreshDocumentStatus();
}

fileInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) return;

  try {
    setBusy(true);
    await uploadFile(file);
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
    fileInput.value = "";
  }
});

clearDocumentButton.addEventListener("click", async () => {
  try {
    setBusy(true);
    const response = await fetch("/api/document", { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Could not clear document.");
    }
    addMessage("assistant", data.message);
    await refreshDocumentStatus();
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;

  addMessage("user", message);
  messageInput.value = "";

  try {
    setBusy(true);
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message, api_key: apiKeyInput.value.trim() }),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Chat request failed.");
    }

    addMessage("assistant", data.answer);
    await refreshDocumentStatus();
  } catch (error) {
    addMessage("assistant", error.message);
  } finally {
    setBusy(false);
  }
});

refreshDocumentStatus();
