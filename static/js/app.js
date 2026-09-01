// Frontend Application Logic
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("fileInput");
  const btnBrowseFile = document.getElementById("btnBrowseFile");
  const btnToggleCamera = document.getElementById("btnToggleCamera");
  const cameraContainer = document.getElementById("cameraContainer");
  const cameraVideo = document.getElementById("cameraVideo");
  const btnCaptureSnap = document.getElementById("btnCaptureSnap");
  const btnCloseCamera = document.getElementById("btnCloseCamera");
  const cameraCanvas = document.getElementById("cameraCanvas");
  const previewContainer = document.getElementById("previewContainer");
  const previewImg = document.getElementById("previewImg");
  const btnRemovePreview = document.getElementById("btnRemovePreview");
  const btnAnalyze = document.getElementById("btnAnalyze");
  const sampleGrid = document.getElementById("sampleGrid");

  const placeholderView = document.getElementById("placeholderView");
  const loadingView = document.getElementById("loadingView");
  const diagnosticResultsView = document.getElementById("diagnosticResultsView");

  // Doctor modal elements
  const doctorModal = document.getElementById("doctorModal");
  const btnOpenDoctor = document.getElementById("btnOpenDoctor");
  const btnCloseDoctor = document.getElementById("btnCloseDoctor");
  const btnSendDoctor = document.getElementById("btnSendDoctor");
  const doctorInput = document.getElementById("doctorInput");
  const doctorChatHistory = document.getElementById("doctorChatHistory");
  const btnPrintReport = document.getElementById("btnPrintReport");

  // State
  let currentFile = null;
  let currentSampleFilename = null;
  let cameraStream = null;
  let currentPlantName = "";
  let currentConditionName = "";

  // 1. Fetch & Render Sample Images Gallery
  async function loadSampleGallery() {
    try {
      const res = await fetch("/api/sample-images");
      const data = await res.json();
      sampleGrid.innerHTML = "";
      
      data.samples.forEach(sample => {
        const card = document.createElement("div");
        card.className = "sample-card";
        card.innerHTML = `
          <img src="/static/sample_images/${sample.filename}" class="sample-thumb" alt="${sample.name}">
          <div class="sample-info">
            <div class="sample-plant">${sample.plant}</div>
            <div class="sample-cond">${sample.condition}</div>
          </div>
        `;
        card.addEventListener("click", () => selectSample(sample));
        sampleGrid.appendChild(card);
      });
    } catch (err) {
      console.error("Failed to load sample gallery:", err);
    }
  }

  function selectSample(sample) {
    stopCamera();
    currentFile = null;
    currentSampleFilename = sample.filename;

    previewImg.src = `/static/sample_images/${sample.filename}`;
    previewContainer.style.display = "block";
    btnAnalyze.disabled = false;
  }

  // 2. Drag & Drop File Handling
  dropzone.addEventListener("click", () => fileInput.click());
  btnBrowseFile.addEventListener("click", () => fileInput.click());

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("drag-over");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  function handleSelectedFile(file) {
    stopCamera();
    currentFile = file;
    currentSampleFilename = null;

    const reader = new FileReader();
    reader.onload = (event) => {
      previewImg.src = event.target.result;
      previewContainer.style.display = "block";
      btnAnalyze.disabled = false;
    };
    reader.readAsDataURL(file);
  }

  // 3. Camera Handling
  btnToggleCamera.addEventListener("click", async () => {
    if (cameraStream) {
      stopCamera();
      return;
    }
    try {
      cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" } });
      cameraVideo.srcObject = cameraStream;
      cameraContainer.style.display = "block";
      previewContainer.style.display = "none";
      btnToggleCamera.classList.add("active");
    } catch (err) {
      alert("Unable to access camera. Please ensure camera permissions are allowed.");
      console.error(err);
    }
  });

  function stopCamera() {
    if (cameraStream) {
      cameraStream.getTracks().forEach(track => track.stop());
      cameraStream = null;
      cameraVideo.srcObject = null;
    }
    cameraContainer.style.display = "none";
    btnToggleCamera.classList.remove("active");
  }

  btnCloseCamera.addEventListener("click", stopCamera);

  btnCaptureSnap.addEventListener("click", () => {
    if (!cameraVideo.videoWidth) return;
    cameraCanvas.width = cameraVideo.videoWidth;
    cameraCanvas.height = cameraVideo.videoHeight;
    const ctx = cameraCanvas.getContext("2d");
    ctx.drawImage(cameraVideo, 0, 0);

    cameraCanvas.toBlob((blob) => {
      currentFile = new File([blob], "snapshot.jpg", { type: "image/jpeg" });
      currentSampleFilename = null;
      previewImg.src = cameraCanvas.toDataURL("image/jpeg");
      previewContainer.style.display = "block";
      btnAnalyze.disabled = false;
      stopCamera();
    }, "image/jpeg", 0.95);
  });

  // Clear Preview
  btnRemovePreview.addEventListener("click", () => {
    currentFile = null;
    currentSampleFilename = null;
    fileInput.value = "";
    previewContainer.style.display = "none";
    btnAnalyze.disabled = true;
  });

  // 4. Analyze & Diagnosis Submission
  btnAnalyze.addEventListener("click", async () => {
    if (!currentFile && !currentSampleFilename) return;

    placeholderView.style.display = "none";
    diagnosticResultsView.style.display = "none";
    loadingView.style.display = "block";
    btnAnalyze.disabled = true;

    const formData = new FormData();
    if (currentFile) {
      formData.append("image", currentFile);
    } else if (currentSampleFilename) {
      formData.append("sample_filename", currentSampleFilename);
    }

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        body: formData
      });
      const data = await res.json();

      loadingView.style.display = "none";
      btnAnalyze.disabled = false;

      if (!data.success) {
        alert(data.error || "Analysis failed.");
        placeholderView.style.display = "block";
        return;
      }

      renderDiagnosis(data.recommendation);

    } catch (err) {
      loadingView.style.display = "none";
      btnAnalyze.disabled = false;
      placeholderView.style.display = "block";
      alert("Error contacting analysis server: " + err.message);
      console.error(err);
    }
  });

  // 5. Render Structured Diagnostic & Recommendation Card
  function renderDiagnosis(rec) {
    currentPlantName = rec.species;
    currentConditionName = rec.condition_name;

    document.getElementById("resPlantName").textContent = rec.species;
    document.getElementById("resConditionName").textContent = rec.condition_name;

    // Severity Badge
    const sevBadge = document.getElementById("resSeverityBadge");
    sevBadge.className = `badge ${rec.severity_info.class}`;
    sevBadge.innerHTML = `<i class="fa-solid ${rec.severity_info.icon}"></i> ${rec.severity_info.label}`;

    // Reliability Badge
    const relBadge = document.getElementById("resReliabilityBadge");
    relBadge.className = `badge ${rec.reliability_badge}`;
    relBadge.textContent = rec.reliability;

    // Confidence Bar
    document.getElementById("resConfidenceText").textContent = `${rec.confidence_percentage}%`;
    const confFill = document.getElementById("resConfidenceFill");
    confFill.style.width = "0%";
    setTimeout(() => {
      confFill.style.width = `${rec.confidence_percentage}%`;
    }, 100);

    // Description
    document.getElementById("resDescription").textContent = rec.description;

    // Differential List
    const diffList = document.getElementById("resDifferentialList");
    diffList.innerHTML = "";
    if (rec.top_predictions && rec.top_predictions.length > 0) {
      rec.top_predictions.forEach(p => {
        const item = document.createElement("div");
        item.className = "diff-item";
        item.innerHTML = `
          <span>${p.plant} – ${p.condition}</span>
          <div style="display:flex; align-items:center;">
            <strong>${p.confidence_percent}%</strong>
            <div class="diff-bar-bg">
              <div class="diff-bar-fill" style="width: ${p.confidence_percent}%;"></div>
            </div>
          </div>
        `;
        diffList.appendChild(item);
      });
    }

    // Disclaimer
    document.getElementById("resDisclaimerText").textContent = rec.confidence_disclaimer;

    // Immediate Actions
    const immUl = document.getElementById("resImmediateActions");
    immUl.innerHTML = "";
    (rec.immediate_actions || []).forEach(act => {
      const li = document.createElement("li");
      li.textContent = act;
      immUl.appendChild(li);
    });

    // Watering
    document.getElementById("resWaterFreq").textContent = rec.watering.frequency || "As needed";
    document.getElementById("resWaterMethod").textContent = rec.watering.method || "Base watering";
    document.getElementById("resWaterTips").textContent = rec.watering.schedule_tips || "Morning schedule";

    // Sunlight
    document.getElementById("resSunlight").textContent = rec.sunlight_environment.sunlight || "Direct sunlight";
    document.getElementById("resAirflow").textContent = rec.sunlight_environment.airflow || "Well ventilated";
    document.getElementById("resTemp").textContent = rec.sunlight_environment.temperature_humidity || "Warm & moderate humidity";

    // Soil
    document.getElementById("resSoil").textContent = rec.soil_nutrition.soil_type || "Well-draining";
    document.getElementById("resSoilPh").textContent = rec.soil_nutrition.ph_range || "6.0 - 7.0";
    document.getElementById("resFertilizer").textContent = rec.soil_nutrition.fertilizer || "Balanced fertilizer";

    // Treatments
    const orgUl = document.getElementById("resOrganicTreatments");
    orgUl.innerHTML = "";
    (rec.treatments.organic || []).forEach(t => {
      const li = document.createElement("li");
      li.textContent = t;
      orgUl.appendChild(li);
    });

    const chemUl = document.getElementById("resChemicalTreatments");
    chemUl.innerHTML = "";
    (rec.treatments.chemical || []).forEach(t => {
      const li = document.createElement("li");
      li.textContent = t;
      chemUl.appendChild(li);
    });

    // Checklist
    const chkDiv = document.getElementById("checklistItems");
    chkDiv.innerHTML = "";
    const tasks = rec.checklist_tasks || [];
    let completedCount = 0;

    tasks.forEach((task, idx) => {
      const row = document.createElement("label");
      row.className = "checklist-item";
      row.innerHTML = `
        <input type="checkbox" data-idx="${idx}">
        <span>${task}</span>
      `;
      const chk = row.querySelector("input");
      chk.addEventListener("change", () => {
        if (chk.checked) {
          row.classList.add("done");
          completedCount++;
        } else {
          row.classList.remove("done");
          completedCount--;
        }
        updateChecklistProgress(completedCount, tasks.length);
      });
      chkDiv.appendChild(row);
    });
    updateChecklistProgress(0, tasks.length);

    // Prevention
    const prevUl = document.getElementById("resPreventionList");
    prevUl.innerHTML = "";
    (rec.prevention || []).forEach(p => {
      const li = document.createElement("li");
      li.textContent = p;
      prevUl.appendChild(li);
    });

    diagnosticResultsView.style.display = "block";
  }

  function updateChecklistProgress(done, total) {
    const prog = document.getElementById("checklistProgress");
    prog.textContent = `${done}/${total} Completed`;
  }

  // 6. Print Report Trigger
  btnPrintReport.addEventListener("click", () => {
    window.print();
  });

  // 7. Plant Doctor Chat Modal
  btnOpenDoctor.addEventListener("click", () => {
    doctorModal.style.display = "flex";
    doctorInput.focus();
  });

  btnCloseDoctor.addEventListener("click", () => {
    doctorModal.style.display = "none";
  });

  doctorModal.addEventListener("click", (e) => {
    if (e.target === doctorModal) doctorModal.style.display = "none";
  });

  btnSendDoctor.addEventListener("click", sendDoctorMessage);
  doctorInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendDoctorMessage();
  });

  async function sendDoctorMessage() {
    const q = doctorInput.value.trim();
    if (!q) return;

    appendChatBubble(q, "bubble-user");
    doctorInput.value = "";

    try {
      const res = await fetch("/api/ask-doctor", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          plant: currentPlantName,
          condition: currentConditionName
        })
      });
      const data = await res.json();
      if (data.success) {
        appendChatBubble(data.answer, "bubble-doc");
      } else {
        appendChatBubble("Sorry, I encountered an issue. Please try again.", "bubble-doc");
      }
    } catch (err) {
      appendChatBubble("Network connection error.", "bubble-doc");
    }
  }

  function appendChatBubble(text, bubbleClass) {
    const b = document.createElement("div");
    b.className = `chat-bubble ${bubbleClass}`;
    b.innerHTML = text.replace(/\n/g, "<br>");
    doctorChatHistory.appendChild(b);
    doctorChatHistory.scrollTop = doctorChatHistory.scrollHeight;
  }

  // Initialize
  loadSampleGallery();
});
