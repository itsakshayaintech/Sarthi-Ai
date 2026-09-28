const csrfToken = document.querySelector('meta[name="csrf-token"]')?.content || "";
let voiceFeatures = null;
let voiceTranscript = "";

document.querySelectorAll('input[type="range"]').forEach((input) => {
    const output = input.parentElement.querySelector("output");
    input.addEventListener("input", () => {
        output.value = input.value;
        output.textContent = input.value;
    });
});

const audioInput = document.getElementById("audio-file");
if (audioInput) {
    audioInput.addEventListener("change", analyzeVoice);
}

const recordButton = document.getElementById("record-button");
let activeRecorder = null;

recordButton?.addEventListener("click", async () => {
    const status = document.getElementById("audio-status");
    const recordingStatus = document.getElementById("recording-status");
    if (!document.getElementById("ai-consent").checked) {
        status.textContent = "Please give consent above before recording audio.";
        return;
    }

    if (activeRecorder?.state === "recording") {
        activeRecorder.stop();
        recordButton.disabled = true;
        recordButton.textContent = "Preparing recording...";
        recordingStatus.textContent = "Finishing recording";
        return;
    }

    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
        status.textContent = "Live recording is not supported by this browser.";
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mimeType = ["audio/webm;codecs=opus", "audio/ogg;codecs=opus", "audio/mp4"]
            .find((type) => MediaRecorder.isTypeSupported(type));
        activeRecorder = mimeType
            ? new MediaRecorder(stream, { mimeType })
            : new MediaRecorder(stream);
        const chunks = [];
        activeRecorder.addEventListener("dataavailable", (event) => {
            if (event.data.size) chunks.push(event.data);
        });
        activeRecorder.addEventListener("stop", async () => {
            stream.getTracks().forEach((track) => track.stop());
            recordingStatus.textContent = "";
            try {
                await processAudioSources([
                    new File(chunks, "live-recording", { type: activeRecorder.mimeType || "audio/webm" }),
                ]);
            } finally {
                activeRecorder = null;
                recordButton.disabled = false;
                recordButton.textContent = "Start recording";
            }
        }, { once: true });
        activeRecorder.start();
        recordButton.textContent = "Stop recording";
        recordingStatus.textContent = "Recording";
        status.textContent = "Speak clearly, then select Stop recording.";
    } catch (error) {
        status.textContent = error.name === "NotAllowedError"
            ? "Microphone access was denied. Allow microphone access and try again."
            : "Could not start the microphone. Check that it is connected and available.";
    }
});

async function analyzeVoice() {
    const files = Array.from(audioInput.files || []);
    if (!files.length) return;
    await processAudioSources(files);
    audioInput.value = "";
}

async function processAudioSources(sources) {
    const status = document.getElementById("audio-status");
    if (!document.getElementById("ai-consent").checked) {
        status.textContent = "Please give consent above before uploading audio.";
        return;
    }

    try {
        status.textContent = "Converting audio in your browser...";
        const wavFile = await normalizeAudioSources(sources);
        status.textContent = "Transcribing and analyzing audio...";
        const formData = new FormData();
        formData.append("audio", wavFile, wavFile.name);
        formData.append("ai_consent", "true");
        const response = await fetch("/analyze-audio", {
            method: "POST",
            headers: { "X-CSRF-Token": csrfToken },
            body: formData,
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Audio analysis failed.");
        voiceFeatures = result.voice_features;
        voiceTranscript = result.transcript || "";
        const narrative = document.getElementById("reported-context");
        if (voiceTranscript) {
            narrative.value = [narrative.value.trim(), voiceTranscript].filter(Boolean).join("\n");
        }
        status.textContent = voiceTranscript
            ? `Transcript added to your check-in: "${voiceTranscript}" Run the assessment to review results.`
            : "Audio features analyzed, but no transcript was detected. Run the assessment to review results.";
    } catch (error) {
        voiceFeatures = null;
        status.textContent = error.message || "Audio could not be processed. Try another supported audio format.";
    }
}

async function normalizeAudioSources(sources) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const OfflineAudioContextClass = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!AudioContextClass || !OfflineAudioContextClass) {
        throw new Error("Audio conversion is not supported by this browser.");
    }

    const decoder = new AudioContextClass();
    try {
        const decoded = await Promise.all(sources.map(async (source) =>
            decoder.decodeAudioData(await source.arrayBuffer())
        ));
        const totalDuration = decoded.reduce((total, clip) => total + clip.duration, 0);
        if (totalDuration > 300) {
            throw new Error("The combined audio must be 5 minutes or shorter.");
        }
        const sampleRate = 16000;
        const frameCount = Math.ceil(totalDuration * sampleRate);
        if (!frameCount) throw new Error("The selected audio is empty.");

        const renderer = new OfflineAudioContextClass(1, frameCount, sampleRate);
        let offset = 0;
        decoded.forEach((clip) => {
            const source = renderer.createBufferSource();
            source.buffer = clip;
            source.connect(renderer.destination);
            source.start(offset);
            offset += clip.duration;
        });
        const rendered = await renderer.startRendering();
        const samples = rendered.getChannelData(0);
        const wav = new ArrayBuffer(44 + samples.length * 2);
        const view = new DataView(wav);
        const writeString = (start, value) => {
            for (let index = 0; index < value.length; index += 1) {
                view.setUint8(start + index, value.charCodeAt(index));
            }
        };
        writeString(0, "RIFF");
        view.setUint32(4, wav.byteLength - 8, true);
        writeString(8, "WAVE");
        writeString(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, "data");
        view.setUint32(40, samples.length * 2, true);
        samples.forEach((sample, index) => {
            const clipped = Math.max(-1, Math.min(1, sample));
            view.setInt16(44 + index * 2, clipped < 0 ? clipped * 0x8000 : clipped * 0x7fff, true);
        });
        return new File([wav], "sarthi-voice-input.wav", { type: "audio/wav" });
    } finally {
        await decoder.close();
    }
}

document.getElementById("assess-button")?.addEventListener("click", submitAssessment);

async function submitAssessment() {
    const button = document.getElementById("assess-button");
    const resultPanel = document.getElementById("assessment-result");
    if (!document.getElementById("ai-consent").checked) {
        resultPanel.classList.remove("hidden");
        resultPanel.textContent = "Please confirm consent before running AI analysis.";
        return;
    }

    const incidentCategory = document.getElementById("incident-category").value;
    const reportedContext = document.getElementById("reported-context").value.trim() || voiceTranscript;
    const assessmentContext = [incidentCategory, reportedContext].filter(Boolean).join("\n");

    const payload = {
        ai_consent: true,
        indicators: Object.fromEntries(
            ["fear", "threat", "anxiety", "trauma", "social_isolation", "displacement", "legal_stress", "safety_concern"]
                .map((key) => [key, Number(document.getElementById(key).value)])
        ),
        text: assessmentContext,
        case_context: {
            state_ut: document.getElementById("state-ut").value || null,
            victim_group: null,
            reported_context: assessmentContext || null,
            immediate_safety_concern: Number(document.getElementById("immediate-safety").value),
            social_isolation_indicator: Number(document.getElementById("context-isolation").value),
            displacement_indicator: Number(document.getElementById("context-displacement").value),
            legal_delay_indicator: Number(document.getElementById("legal-delay").value),
        },
    };
    if (voiceFeatures) payload.voice_features = voiceFeatures;

    button.disabled = true;
    button.textContent = "Generating results...";
    resultPanel.classList.remove("hidden");
    resultPanel.setAttribute("aria-busy", "true");
    resultPanel.textContent = "Your assessment is being analyzed.";
    try {
        const response = await fetch("/assess", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "X-CSRF-Token": csrfToken,
            },
            body: JSON.stringify(payload),
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Assessment failed.");
        renderResult(resultPanel, result, payload);
    } catch (error) {
        resultPanel.replaceChildren();
        const message = document.createElement("p");
        message.className = "result-error";
        message.setAttribute("role", "alert");
        message.textContent = error.message || "We couldn't generate the assessment. Please try again.";
        resultPanel.append(message);
    } finally {
        button.disabled = false;
        button.textContent = "Generate assessment results";
        resultPanel.setAttribute("aria-busy", "false");
    }
}

function renderResult(panel, result, submitted) {
    const category = String(result.risk_category || "").toLowerCase();
    const modelLabels = {
        indicator_svi: "Stress indicators",
        text_svi: "Written or transcribed text",
        voice_svi: "Voice features",
        context_svi: "Case context",
    };
    const inputLabels = {
        fear: "Fear",
        threat: "Threat concern",
        anxiety: "Anxiety",
        trauma: "Distress",
        social_isolation: "Social isolation",
        displacement: "Displacement",
        legal_stress: "Legal stress",
        safety_concern: "Safety concern",
    };
    const modelScores = Object.entries(result.assessment_scores || {})
        .filter(([, score]) => score !== null && score !== undefined && Number.isFinite(Number(score)))
        .map(([name, score]) => [modelLabels[name] || name, Number(score)]);
    const indicatorScores = Object.entries(submitted.indicators || {})
        .map(([name, score]) => [inputLabels[name] || name, Number(score)])
        .filter(([, score]) => Number.isFinite(score));

    const makeChart = (title, entries, className) => {
        const section = document.createElement("section");
        section.className = `result-chart ${className}`;
        const heading = document.createElement("h3");
        heading.textContent = title;
        const rows = document.createElement("div");
        rows.className = "result-chart-rows";
        rows.setAttribute("role", "list");
        entries.forEach(([label, value]) => {
            const row = document.createElement("div");
            row.className = "result-chart-row";
            row.setAttribute("role", "listitem");
            const rowHeading = document.createElement("div");
            rowHeading.className = "result-chart-heading";
            const rowLabel = document.createElement("span");
            rowLabel.textContent = label;
            const rowValue = document.createElement("strong");
            rowValue.textContent = `${value.toFixed(1)} / 100`;
            const track = document.createElement("div");
            track.className = "result-chart-track";
            track.setAttribute("role", "img");
            track.setAttribute("aria-label", `${label}: ${value.toFixed(1)} out of 100`);
            const fill = document.createElement("span");
            fill.className = "result-chart-fill";
            fill.style.width = `${Math.min(100, Math.max(0, value))}%`;
            track.append(fill);
            rowHeading.append(rowLabel, rowValue);
            row.append(rowHeading, track);
            rows.append(row);
        });
        section.append(heading, rows);
        return section;
    };

    panel.replaceChildren();
    panel.classList.remove("hidden");
    const heading = document.createElement("p");
    heading.className = "eyebrow";
    heading.textContent = "Assessment results";
    const summary = document.createElement("div");
    summary.className = "result-summary";
    const scoreBlock = document.createElement("div");
    scoreBlock.className = "result-total";
    const scoreLabel = document.createElement("span");
    scoreLabel.textContent = "Stress Vulnerability Index";
    const score = document.createElement("strong");
    score.className = "result-score";
    score.textContent = `${Number(result.final_svi).toFixed(1)} / 100`;
    scoreBlock.append(scoreLabel, score);
    const risk = document.createElement("p");
    risk.className = `risk-pill result-risk risk-${category}`;
    risk.textContent = `${result.risk_category} model category`;
    summary.append(scoreBlock, risk);

    const charts = document.createElement("div");
    charts.className = "result-charts";
    if (modelScores.length) {
        charts.append(makeChart("Scores by source", modelScores, "model-chart"));
    }
    if (indicatorScores.length) {
        charts.append(makeChart("Your indicator inputs", indicatorScores, "indicator-chart"));
    }

    const details = document.createElement("div");
    details.className = "result-followup";
    const recommendationsTitle = document.createElement("h3");
    recommendationsTitle.textContent = "Recommended next steps";
    const recommendations = document.createElement("ul");
    (result.support_recommendation || []).forEach((item) => {
        const listItem = document.createElement("li");
        listItem.textContent = item;
        recommendations.append(listItem);
    });
    details.append(recommendationsTitle, recommendations);
    const note = document.createElement("p");
    note.className = "result-note";
    note.textContent = result.important_note;
    details.append(note);

    const actions = document.createElement("div");
    actions.className = "result-actions";
    const savedMessage = document.createElement("p");
    savedMessage.textContent = "Assessment saved to My history.";
    const newAssessment = document.createElement("a");
    newAssessment.className = "button button-secondary";
    newAssessment.href = "/dashboard";
    newAssessment.textContent = "Start a new assessment";
    actions.append(savedMessage, newAssessment);

    panel.append(heading, summary, charts, details, actions);
    panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

const chatAnswers = [
    { terms: ["danger", "unsafe", "emergency", "hurt", "threat"], reply: "If you or someone else is in immediate danger in India, call 112. Move to a safer place if you can and contact someone you trust." },
    { terms: ["helpline", "hotline", "contact", "phone number", "telephone number", "call", "nhaa", "14566"], reply: "For direct assistance, call NHAA (National Helpline Against Atrocities) at 14566." },
    { terms: ["talk", "counsel", "stress", "anxious", "feel"], reply: "Consider speaking with a qualified counsellor or a trusted healthcare professional. You can also ask your local helpline worker about available services." },
    { terms: ["legal", "police", "report", "rights"], reply: "A local legal-aid service or authorized helpline worker may explain options for your situation. Avoid sharing identifying details in this chat." },
    { terms: ["hello", "hi", "help", "support"], reply: "I can point you to general safety, counselling, and legal-support options. What would you like information about?" },
];

function getChatReply(message) {
    const normalized = message.toLowerCase();
    const matches = (answer) => answer.terms.some((term) => normalized.includes(term));
    const emergencyAnswer = chatAnswers[0];
    const contactAnswer = chatAnswers[1];

    if (matches(emergencyAnswer)) {
        const contactDetails = matches(contactAnswer)
            ? " For direct assistance, call NHAA (National Helpline Against Atrocities) at 14566."
            : "";
        return `${emergencyAnswer.reply}${contactDetails}`;
    }

    const answer = matches(contactAnswer)
        ? contactAnswer
        : chatAnswers.slice(2).find(matches);
    return answer?.reply || "I don't have information on that. A helpline worker can help you find an appropriate local service.";
}

document.getElementById("chat-form")?.addEventListener("submit", (event) => {
    event.preventDefault();
    const input = document.getElementById("chat-input");
    const message = input.value.trim();
    if (!message) return;
    addChatMessage(message, "chat-user");
    addChatMessage(getChatReply(message), "chat-assistant");
    input.value = "";
});

function addChatMessage(text, className) {
    const message = document.createElement("p");
    message.className = `chat-message ${className}`;
    message.textContent = text;
    const log = document.getElementById("chat-log");
    log.append(message);
    log.scrollTop = log.scrollHeight;
}
let aiConsent = null;


// -------------------------------------------------
// CONSENT SELECTION
// -------------------------------------------------

function selectConsent(consent) {

    aiConsent = consent;

    document
        .getElementById("consent-section")
        .classList
        .add("hidden");


    if (consent === true) {

        document
            .getElementById("ai-section")
            .classList
            .remove("hidden");

    } else {

        document
            .getElementById("human-section")
            .classList
            .remove("hidden");

    }
}


// -------------------------------------------------
// GET NUMBER VALUE SAFELY
// -------------------------------------------------

function getNumber(id) {

    const element = document.getElementById(id);

    if (!element) {
        return 0;
    }

    const value = element.value;

    return value === "" ? 0 : Number(value);
}


// -------------------------------------------------
// GET TEXT VALUE SAFELY
// -------------------------------------------------

function getText(id) {

    const element = document.getElementById(id);

    if (!element) {
        return "";
    }

    return element.value.trim();
}


// -------------------------------------------------
// SUBMIT AI ASSESSMENT
// -------------------------------------------------

async function submitLegacyAssessment() {

    // -------------------------------------------------
    // GET CASE TEXT
    // -------------------------------------------------

    const reportedContext = getText("reported_context");


    // -------------------------------------------------
    // CREATE REQUEST DATA
    // -------------------------------------------------

    const data = {

        ai_consent: true,


        // ---------------------------------------------
        // STRESS INDICATORS
        // ---------------------------------------------

        indicators: {

            fear:
                getNumber("fear"),

            threat:
                getNumber("threat"),

            anxiety:
                getNumber("anxiety"),

            trauma:
                getNumber("trauma"),

            social_isolation:
                getNumber("social_isolation"),

            displacement:
                getNumber("displacement"),

            legal_stress:
                getNumber("legal_stress"),

            safety_concern:
                getNumber("safety_concern")
        },


        // ---------------------------------------------
        // TEXT MODEL
        // ---------------------------------------------

        text: reportedContext,


        // ---------------------------------------------
        // CASE CONTEXT MODEL
        // ---------------------------------------------

        case_context: {

            state_ut:
                getText("state_ut"),

            victim_group:
                getText("victim_group"),

            reported_context:
                reportedContext,

            immediate_safety_concern:
                getNumber(
                    "immediate_safety_concern"
                ),

            social_isolation_indicator:
                getNumber(
                    "social_isolation_indicator"
                ),

            displacement_indicator:
                getNumber(
                    "displacement_indicator"
                ),

            legal_delay_indicator:
                getNumber(
                    "legal_delay_indicator"
                )
        }
    };


    // -------------------------------------------------
    // AUDIO FILE CHECK
    // -------------------------------------------------

    const audioInput =
        document.getElementById("audio_file");


    if (
        audioInput &&
        audioInput.files.length > 0
    ) {

        const audioStatus =
            document.getElementById("audio-status");


        if (audioStatus) {

            audioStatus.textContent =
                "Audio selected: " +
                audioInput.files[0].name +
                ". Voice feature extraction is not yet connected.";

        }
    }


    // -------------------------------------------------
    // BUTTON LOADING STATE
    // -------------------------------------------------

    const button =
        document.querySelector(".assess-btn");


    if (button) {

        button.disabled = true;

        button.textContent =
            "Analysing...";
    }


    try {

        // ---------------------------------------------
        // SEND DATA TO FLASK
        // ---------------------------------------------

        const response =
            await fetch(
                "/assess",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(data)
                }
            );


        // ---------------------------------------------
        // GET RESPONSE
        // ---------------------------------------------

        const result =
            await response.json();


        // ---------------------------------------------
        // HANDLE ERROR
        // ---------------------------------------------

        if (!response.ok) {

            alert(
                result.error ||
                "Assessment failed."
            );

            return;
        }


        // ---------------------------------------------
        // DISPLAY RESULT
        // ---------------------------------------------

        displayResults(result);


    } catch (error) {

        console.error(
            "Assessment Error:",
            error
        );


        alert(
            "Unable to connect to the Sarthi-AI server."
        );

    } finally {

        // ---------------------------------------------
        // RESET BUTTON
        // ---------------------------------------------

        if (button) {

            button.disabled = false;

            button.textContent =
                "Assess Vulnerability";
        }
    }
}


// -------------------------------------------------
// DISPLAY RESULTS
// -------------------------------------------------

function displayResults(result) {

    const resultsSection =
        document.getElementById(
            "results-section"
        );


    resultsSection
        .classList
        .remove("hidden");


    // -------------------------------------------------
    // GET SCORES
    // -------------------------------------------------

    const scores =
        result.assessment_scores || {};


    let scoreHTML = "";


    // -------------------------------------------------
    // CREATE SCORE LIST
    // -------------------------------------------------

    for (
        const [name, score]
        of Object.entries(scores)
    ) {

        const formattedName =
            name
                .replace("_svi", "")
                .replaceAll("_", " ")
                .replace(
                    /\b\w/g,
                    letter =>
                        letter.toUpperCase()
                );


        scoreHTML += `
            <li>
                <strong>
                    ${formattedName}:
                </strong>

                ${score}
            </li>
        `;
    }


    // -------------------------------------------------
    // SUPPORT RECOMMENDATIONS
    // -------------------------------------------------

    let recommendationHTML = "";


    if (
        result.support_recommendation &&
        result.support_recommendation.length > 0
    ) {

        recommendationHTML = `

            <div class="result-section">

                <h3>
                    Recommended Support
                </h3>

                <ul>

                    ${result.support_recommendation
                        .map(
                            item =>
                                `<li>${item}</li>`
                        )
                        .join("")
                    }

                </ul>

            </div>
        `;
    }


    // -------------------------------------------------
    // RISK CLASS
    // -------------------------------------------------

    let riskClass = "";


    if (
        result.risk_category === "Low"
    ) {

        riskClass = "risk-low";

    } else if (
        result.risk_category === "Moderate"
    ) {

        riskClass = "risk-moderate";

    } else if (
        result.risk_category === "High"
    ) {

        riskClass = "risk-high";

    } else if (
        result.risk_category === "Critical"
    ) {

        riskClass = "risk-critical";
    }


    // -------------------------------------------------
    // DISPLAY HTML
    // -------------------------------------------------

    document
        .getElementById("result-content")
        .innerHTML = `

        <div class="result-svi">

            ${result.final_svi}

        </div>


        <p class="result-label">

            Stress Vulnerability Index (SVI)

        </p>


        <div class="result-risk ${riskClass}">

            Risk Category:

            <strong>
                ${result.risk_category}
            </strong>

        </div>


        <div class="result-section">

            <h3>
                Assessment Scores
            </h3>

            <ul>

                ${scoreHTML}

            </ul>

        </div>


        ${recommendationHTML}


        <div class="result-section">

            <h3>
                Important Note
            </h3>

            <p>

                ${result.important_note}

            </p>

        </div>
    `;


    // -------------------------------------------------
    // SCROLL TO RESULT
    // -------------------------------------------------

    resultsSection.scrollIntoView({
        behavior: "smooth",
        block: "start"
    });
}