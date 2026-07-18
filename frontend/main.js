/*
  AssistArena Frontend Controller.
  Manages UI interactions, NDJSON streaming, and accessibility considerations.
  All dynamic rendering uses textContent/createTextNode (no innerHTML, preventing XSS).
*/

"use strict";

const MAX_TURNS_LIMIT = 20;
const MAX_QUERY_LEN = 2000;

const SPEAKER_LABELS = {
  user: "You",
  assistant: "AssistArena",
  status: "AssistArena",
};

let conversationHistory = [];

const ui = {
  stadium: document.getElementById("stadium-selector"),
  language: document.getElementById("lang-selector"),
  chatLog: document.getElementById("chat-log"),
  form: document.getElementById("consult-composer"),
  input: document.getElementById("chat-query-input"),
  submit: document.getElementById("submit-query-btn"),
  banner: document.getElementById("offline-status-banner"),
};

const onboarding = {
  overlay: document.getElementById("onboarding-overlay"),
  stadium: document.getElementById("onboarding-stadium-selector"),
  btnNext1: document.getElementById("onboarding-btn-next-1"),
  btnYes: document.getElementById("onboarding-btn-assistance-yes"),
  btnNo: document.getElementById("onboarding-btn-assistance-no"),
  btnFinish: document.getElementById("onboarding-btn-finish"),
  reconfigureBtn: document.getElementById("reconfigure-setup-btn"),
  step1: document.getElementById("onboarding-step-1"),
  step2: document.getElementById("onboarding-step-2"),
  step3: document.getElementById("onboarding-step-3"),
};

function insertMessage(role, text) {
  const container = document.createElement("div");
  container.className = "chat-bubble chat-bubble--" + role;

  if (role === "status") {
    container.setAttribute("role", "alert");
  }

  const authorSpan = document.createElement("span");
  authorSpan.className = "chat-bubble__author";
  authorSpan.textContent = SPEAKER_LABELS[role] + ":";

  const textSpan = document.createElement("span");
  textSpan.className = "chat-bubble__text";
  textSpan.textContent = text;

  container.appendChild(authorSpan);
  container.appendChild(textSpan);

  const activeLang = ui.language.value;
  container.setAttribute("lang", activeLang);
  container.setAttribute("dir", activeLang === "ar" ? "rtl" : "ltr");

  ui.chatLog.appendChild(container);
  ui.chatLog.scrollTop = ui.chatLog.scrollHeight;
  return container;
}

function insertPendingMessage() {
  const container = insertMessage("assistant", "");
  container.classList.add("chat-bubble--pending");
  const textSpan = container.querySelector(".chat-bubble__text");
  if (textSpan) {
    textSpan.textContent = "";
  }
  return container;
}

function recordInteraction(role, text) {
  conversationHistory.push({ role, text });
  if (conversationHistory.length > MAX_TURNS_LIMIT) {
    conversationHistory = conversationHistory.slice(conversationHistory.length - MAX_TURNS_LIMIT);
  }
}

function updateOfflineBanner(isOffline) {
  if (ui.banner) {
    ui.banner.hidden = !isOffline;
  }
}

function refreshLanguageDirection() {
  const lang = ui.language.value;
  const dir = lang === "ar" ? "rtl" : "ltr";
  ui.chatLog.setAttribute("lang", lang);
  ui.chatLog.setAttribute("dir", dir);
  ui.input.setAttribute("lang", lang);
  ui.input.setAttribute("dir", dir);
}

async function populateStadiums() {
  try {
    const res = await fetch("/api/stadiums", { headers: { Accept: "application/json" } });
    if (!res.ok) {
      throw new Error("API error: " + res.status);
    }
    const data = await res.json();
    const stadiums = Array.isArray(data.stadiums) ? data.stadiums : [];
    for (const s of stadiums) {
      const opt = document.createElement("option");
      opt.value = String(s.id);
      const labelParts = [s.stadiumName];
      if (s.city) {
        labelParts.push(s.city);
      } else if (s.country) {
        labelParts.push(s.country);
      }
      opt.textContent = labelParts.join(" — ");
      ui.stadium.appendChild(opt);
      onboarding.stadium.appendChild(opt.cloneNode(true));
    }
  } catch (err) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Stadiums database offline — ask queries directly";
    opt.disabled = true;
    ui.stadium.appendChild(opt);
    onboarding.stadium.appendChild(opt.cloneNode(true));
  }
}

async function performHealthCheck() {
  try {
    const res = await fetch("/api/healthz", { headers: { Accept: "application/json" } });
    if (!res.ok) {
      return;
    }
    const data = await res.json();
    if (data && data.llm === "offline") {
      updateOfflineBanner(true);
    }
  } catch (err) {
    // Settle on first message request instead.
  }
}

function buildSpectatorProfile() {
  const selectedNeeds = Array.from(
    document.querySelectorAll('input[name="needs"]:checked')
  ).map((el) => el.value);

  return {
    language: ui.language.value,
    needs: selectedNeeds,
    stadium_id: ui.stadium.value ? ui.stadium.value : null,
  };
}

let requestInFlight = false;

async function transmitQuery(rawText) {
  const text = rawText.trim();
  if (!text || requestInFlight) {
    return;
  }
  if (text.length > MAX_QUERY_LEN) {
    insertMessage("status", "Query exceeds the maximum length of 2000 characters.");
    return;
  }

  insertMessage("user", text);
  recordInteraction("user", text);

  requestInFlight = true;
  ui.submit.disabled = true;
  ui.input.value = "";
  ui.chatLog.setAttribute("aria-busy", "true");

  const pendingBubble = insertPendingMessage();
  pendingBubble.setAttribute("aria-hidden", "true");
  const pendingTextSpan = pendingBubble.querySelector(".chat-bubble__text");

  const requestPayload = {
    query: text,
    profile: buildSpectatorProfile(),
    history: conversationHistory.slice(0, MAX_TURNS_LIMIT),
  };

  let aggregatedReply = "";
  let isStreamActive = false;
  let isErrorEvent = false;

  const processNdjsonFrame = (frameLine) => {
    const cleanLine = frameLine.trim();
    if (!cleanLine) {
      return;
    }
    let frame;
    try {
      frame = JSON.parse(cleanLine);
    } catch (e) {
      return;
    }
    if (frame.type === "meta") {
      updateOfflineBanner(frame.mode === "offline");
    } else if (frame.type === "delta" && typeof frame.text === "string") {
      if (!isStreamActive) {
        isStreamActive = true;
        pendingBubble.classList.remove("chat-bubble--pending");
      }
      aggregatedReply += frame.text;
      if (pendingTextSpan) {
        pendingTextSpan.textContent = aggregatedReply;
      }
      ui.chatLog.scrollTop = ui.chatLog.scrollHeight;
    } else if (frame.type === "error") {
      isErrorEvent = true;
    }
  };

  try {
    const response = await fetch("/api/consult/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/x-ndjson",
      },
      body: JSON.stringify(requestPayload),
    });

    if (!response.ok || !response.body) {
      throw new Error("HTTP error " + response.status);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let streamBuffer = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      streamBuffer += decoder.decode(value, { stream: true });
      let newlineIdx;
      while ((newlineIdx = streamBuffer.indexOf("\n")) >= 0) {
        processNdjsonFrame(streamBuffer.slice(0, newlineIdx));
        streamBuffer = streamBuffer.slice(newlineIdx + 1);
      }
    }
    processNdjsonFrame(streamBuffer);

    if (isErrorEvent && !aggregatedReply) {
      throw new Error("Consultation processing error");
    }

    pendingBubble.remove();
    const finalCleanReply = aggregatedReply || "(Empty reply received.)";
    insertMessage("assistant", finalCleanReply);
    if (aggregatedReply) {
      recordInteraction("assistant", aggregatedReply);
    }
  } catch (err) {
    pendingBubble.remove();
    insertMessage("status", "Failed to contact the copilot. Please check your network connection.");
  } finally {
    requestInFlight = false;
    ui.submit.disabled = false;
    ui.chatLog.setAttribute("aria-busy", "false");
    ui.input.focus();
  }
}

function registerEventListeners() {
  ui.form.addEventListener("submit", (e) => {
    e.preventDefault();
    transmitQuery(ui.input.value);
  });

  ui.language.addEventListener("change", refreshLanguageDirection);

  const actionChips = document.querySelectorAll(".query-chip");
  actionChips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt") || chip.textContent || "";
      transmitQuery(prompt);
    });
  });

  const sidebarCheckboxes = document.querySelectorAll('input[name="needs"]');
  sidebarCheckboxes.forEach((cb) => {
    cb.addEventListener("change", () => {
      const selectedNeeds = Array.from(
        document.querySelectorAll('input[name="needs"]:checked')
      ).map((el) => el.value);
      localStorage.setItem("needs", JSON.stringify(selectedNeeds));
      refreshTelemetryDisplay();
    });
  });

  ui.stadium.addEventListener("change", () => {
    localStorage.setItem("assistarena_stadium", ui.stadium.value);
    onboarding.stadium.value = ui.stadium.value;
    refreshTelemetryDisplay();
  });
}

async function refreshTelemetryDisplay() {
  const stadiumId = ui.stadium.value;
  const selectedNeeds = Array.from(
    document.querySelectorAll('input[name="needs"]:checked')
  ).map((el) => el.value);

  const needPills = document.querySelectorAll(".need-status-pill");
  needPills.forEach((pill) => {
    const needVal = pill.getAttribute("data-need");
    const indicator = pill.querySelector(".need-status-indicator");
    if (selectedNeeds.includes(needVal)) {
      pill.classList.add("active");
      if (indicator) {
        indicator.style.backgroundColor = "var(--indicator-online)";
      }
    } else {
      pill.classList.remove("active");
      if (indicator) {
        indicator.style.backgroundColor = "var(--text-sub)";
      }
    }
  });

  const nameEl = document.getElementById("telemetry-stadium-name");
  const locEl = document.getElementById("telemetry-stadium-location");
  const capEl = document.getElementById("telemetry-stadium-capacity");

  if (!stadiumId) {
    if (nameEl) nameEl.textContent = "None Selected";
    if (locEl) locEl.textContent = "-";
    if (capEl) capEl.textContent = "-";
    return;
  }

  try {
    const res = await fetch(`/api/stadiums/${stadiumId}`, { headers: { Accept: "application/json" } });
    if (res.ok) {
      const data = await res.json();
      if (data && data.stadium) {
        const s = data.stadium;
        if (nameEl) nameEl.textContent = s.stadiumName || "-";
        if (locEl) {
          const locParts = [];
          if (s.city) locParts.push(s.city);
          if (s.country) locParts.push(s.country);
          locEl.textContent = locParts.join(", ") || "-";
        }
        if (capEl) capEl.textContent = s.capacity ? Number(s.capacity).toLocaleString() : "-";
      }
    }
  } catch (err) {
    // ignore
  }
}

function trapFocus(stepElement) {
  const focusables = Array.from(stepElement.querySelectorAll('select, button, input')).filter(
    (el) => !el.disabled
  );
  if (focusables.length > 0) {
    focusables[0].focus();
  }
}

function showOnboardingOverlay() {
  onboarding.overlay.classList.remove("hidden");
  onboarding.overlay.hidden = false;
  onboarding.overlay.style.opacity = "1";
  onboarding.step1.classList.add("active");
  onboarding.step2.classList.remove("active");
  onboarding.step3.classList.remove("active");

  onboarding.stadium.value = ui.stadium.value;
  onboarding.btnNext1.disabled = !onboarding.stadium.value;

  const heading = onboarding.step1.querySelector(".onboarding-header");
  if (heading) {
    onboarding.overlay.setAttribute("aria-labelledby", heading.id);
    heading.focus();
  } else {
    trapFocus(onboarding.step1);
  }
}

function hideOnboardingOverlay() {
  onboarding.overlay.style.opacity = "0";
  setTimeout(() => {
    onboarding.overlay.classList.add("hidden");
    onboarding.overlay.hidden = true;
    ui.input.focus();
  }, 400);
}

function transitionToStep(fromStep, toStep) {
  fromStep.classList.remove("active");
  setTimeout(() => {
    toStep.classList.add("active");
    const heading = toStep.querySelector(".onboarding-header");
    if (heading) {
      onboarding.overlay.setAttribute("aria-labelledby", heading.id);
      heading.focus();
    } else {
      trapFocus(toStep);
    }
  }, 200);
}

function setupFocusTrap() {
  onboarding.overlay.addEventListener("keydown", (e) => {
    if (onboarding.overlay.classList.contains("hidden")) return;
    if (e.key === "Tab") {
      const activeStep = onboarding.overlay.querySelector(".onboarding-step.active");
      if (!activeStep) return;
      const heading = activeStep.querySelector(".onboarding-header");
      const focusables = Array.from(
        activeStep.querySelectorAll('select, button, input[type="checkbox"]')
      ).filter((el) => !el.disabled);
      if (focusables.length === 0) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey) {
        if (document.activeElement === first || document.activeElement === heading) {
          last.focus();
          e.preventDefault();
        }
      } else {
        if (document.activeElement === last) {
          first.focus();
          e.preventDefault();
        } else if (document.activeElement === heading) {
          first.focus();
          e.preventDefault();
        }
      }
    }
  });
}

function registerOnboardingEventListeners() {
  setupFocusTrap();

  onboarding.stadium.addEventListener("change", () => {
    onboarding.btnNext1.disabled = !onboarding.stadium.value;
  });

  onboarding.btnNext1.addEventListener("click", () => {
    if (!onboarding.stadium.value) return;
    ui.stadium.value = onboarding.stadium.value;
    localStorage.setItem("assistarena_stadium", onboarding.stadium.value);
    ui.stadium.dispatchEvent(new Event("change"));
    transitionToStep(onboarding.step1, onboarding.step2);
  });

  onboarding.btnYes.addEventListener("click", () => {
    transitionToStep(onboarding.step2, onboarding.step3);
  });

  onboarding.btnNo.addEventListener("click", () => {
    const onboardingCheckboxes = onboarding.step3.querySelectorAll('input[type="checkbox"]');
    onboardingCheckboxes.forEach((cb) => { cb.checked = false; });
    const sidebarCheckboxes = document.querySelectorAll('input[name="needs"]');
    sidebarCheckboxes.forEach((cb) => { cb.checked = false; });

    localStorage.setItem("needs", JSON.stringify([]));
    localStorage.setItem("assistarena_onboarding_completed", "true");
    hideOnboardingOverlay();
    refreshTelemetryDisplay();
  });

  onboarding.btnFinish.addEventListener("click", () => {
    const onboardingCheckboxes = onboarding.step3.querySelectorAll('input[type="checkbox"]');
    const selectedNeeds = [];
    onboardingCheckboxes.forEach((cb) => {
      const sidebarCb = document.querySelector(`input[name="needs"][value="${cb.value}"]`);
      if (sidebarCb) {
        sidebarCb.checked = cb.checked;
      }
      if (cb.checked) {
        selectedNeeds.push(cb.value);
      }
    });

    localStorage.setItem("needs", JSON.stringify(selectedNeeds));
    localStorage.setItem("assistarena_onboarding_completed", "true");
    hideOnboardingOverlay();
    refreshTelemetryDisplay();
  });

  onboarding.reconfigureBtn.addEventListener("click", () => {
    localStorage.removeItem("assistarena_onboarding_completed");
    const sidebarCheckboxes = document.querySelectorAll('input[name="needs"]');
    sidebarCheckboxes.forEach((cb) => {
      const onboardingCb = onboarding.step3.querySelector(`input[type="checkbox"][value="${cb.value}"]`);
      if (onboardingCb) {
        onboardingCb.checked = cb.checked;
      }
    });
    showOnboardingOverlay();
  });
}

function loadStoredOnboarding() {
  const completed = localStorage.getItem("assistarena_onboarding_completed");
  const storedStadium = localStorage.getItem("assistarena_stadium");
  const storedNeedsStr = localStorage.getItem("needs");

  if (storedStadium) {
    ui.stadium.value = storedStadium;
    onboarding.stadium.value = storedStadium;
  }

  if (storedNeedsStr) {
    try {
      const storedNeeds = JSON.parse(storedNeedsStr);
      if (Array.isArray(storedNeeds)) {
        const sidebarCheckboxes = document.querySelectorAll('input[name="needs"]');
        sidebarCheckboxes.forEach((cb) => {
          cb.checked = storedNeeds.includes(cb.value);
        });
        const onboardingCheckboxes = onboarding.step3.querySelectorAll('input[type="checkbox"]');
        onboardingCheckboxes.forEach((cb) => {
          cb.checked = storedNeeds.includes(cb.value);
        });
      }
    } catch (e) {
      // ignore
    }
  }

  if (completed !== "true") {
    showOnboardingOverlay();
  } else {
    onboarding.overlay.classList.add("hidden");
    onboarding.overlay.hidden = true;
  }
}

const WELCOME_GREETING =
  "Welcome! I am AssistArena, your stadium operations copilot for the FIFA World Cup 2026. " +
  "Select a stadium and configure your support needs, or enter any query directly " +
  "to ask about entrances, step-free access, quiet spaces, water, or realtime conditions.";

function displayWelcomeMessage() {
  insertMessage("assistant", WELCOME_GREETING);
}

function initializeApp() {
  refreshLanguageDirection();
  displayWelcomeMessage();
  registerEventListeners();
  registerOnboardingEventListeners();
  populateStadiums().then(() => {
    loadStoredOnboarding();
    refreshTelemetryDisplay();
  });
  performHealthCheck();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeApp);
} else {
  initializeApp();
}
