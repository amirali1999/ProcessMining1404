// Disco-like Dashboard JavaScript

// ============================================
// GLOBAL FUNCTIONS (accessible everywhere)
// ============================================

// Function to initialize zoom controls (can be called after dynamic map loading)
function initializeZoomControls() {
  const processMapImg = document.getElementById("process-map-img");
  const processMapViewport = document.getElementById("process-map-viewport");
  const zoomInBtn = document.getElementById("zoom-in-btn");
  const zoomOutBtn = document.getElementById("zoom-out-btn");
  const zoomResetBtn = document.getElementById("zoom-reset-btn");
  const zoomLevelDisplay = document.getElementById("zoom-level");

  if (
    !processMapImg ||
    !processMapViewport ||
    !zoomInBtn ||
    !zoomOutBtn ||
    !zoomResetBtn ||
    !zoomLevelDisplay
  ) {
    console.warn("Zoom controls not found, skipping initialization");
    return;
  }

  let scale = 1;
  let panning = false;
  let pointX = 0;
  let pointY = 0;
  let start = { x: 0, y: 0 };

  const MIN_SCALE = 1.0;
  const MAX_SCALE = 5;
  const ZOOM_STEP = 0.25;

  function updateZoomDisplay() {
    const percentage = Math.round(scale * 100);
    zoomLevelDisplay.textContent = `${percentage}%`;
  }

  function setTransform() {
    const imgRect = processMapImg.getBoundingClientRect();
    const viewportRect = processMapViewport.getBoundingClientRect();
    const scaledWidth = processMapImg.naturalWidth * scale;
    const scaledHeight = processMapImg.naturalHeight * scale;

    if (scaledWidth > viewportRect.width) {
      const maxTranslateX = (scaledWidth - viewportRect.width) / (2 * scale);
      pointX = Math.max(-maxTranslateX, Math.min(maxTranslateX, pointX));
    } else {
      pointX = 0;
    }

    if (scaledHeight > viewportRect.height) {
      const maxTranslateY = (scaledHeight - viewportRect.height) / (2 * scale);
      pointY = Math.max(-maxTranslateY, Math.min(maxTranslateY, pointY));
    } else {
      pointY = 0;
    }

    processMapImg.style.transform = `scale(${scale}) translate(${pointX}px, ${pointY}px)`;
  }

  zoomInBtn.addEventListener("click", function () {
    if (scale < MAX_SCALE) {
      scale += ZOOM_STEP;
      scale = Math.min(scale, MAX_SCALE);
      setTransform();
      updateZoomDisplay();
    }
  });

  zoomOutBtn.addEventListener("click", function () {
    if (scale > MIN_SCALE) {
      scale -= ZOOM_STEP;
      scale = Math.max(scale, MIN_SCALE);
      setTransform();
      updateZoomDisplay();
    }
  });

  zoomResetBtn.addEventListener("click", function () {
    scale = 1;
    pointX = 0;
    pointY = 0;
    setTransform();
    updateZoomDisplay();
  });

  processMapViewport.addEventListener(
    "wheel",
    function (e) {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
        const newScale = Math.min(
          Math.max(scale + delta, MIN_SCALE),
          MAX_SCALE
        );
        if (newScale !== scale) {
          scale = newScale;
          setTransform();
          updateZoomDisplay();
        }
      }
    },
    { passive: false }
  );

  processMapViewport.addEventListener("mousedown", function (e) {
    e.preventDefault();
    start = { x: e.clientX - pointX, y: e.clientY - pointY };
    panning = true;
    processMapViewport.classList.add("grabbing");
  });

  processMapViewport.addEventListener("mouseup", function () {
    panning = false;
    processMapViewport.classList.remove("grabbing");
  });

  processMapViewport.addEventListener("mouseleave", function () {
    panning = false;
    processMapViewport.classList.remove("grabbing");
  });

  processMapViewport.addEventListener("mousemove", function (e) {
    if (!panning) return;
    e.preventDefault();
    pointX = e.clientX - start.x;
    pointY = e.clientY - start.y;
    setTransform();
  });

  processMapImg.addEventListener("dblclick", function () {
    scale = 1;
    pointX = 0;
    pointY = 0;
    setTransform();
    updateZoomDisplay();
  });

  updateZoomDisplay();
  console.log("Zoom controls initialized successfully");
}

// Function to load project's process map (global so it can be called from anywhere)
function loadProjectMap(projectName) {
  console.log("🚀 DEBUG: loadProjectMap() called with:", projectName);

  const workspace = document.querySelector("#tab-map");
  console.log("🔍 DEBUG: workspace element found:", workspace);

  if (!workspace) {
    console.error("❌ DEBUG: workspace #tab-map NOT FOUND!");
    return;
  }

  console.log("✅ DEBUG: Showing loading state");
  workspace.innerHTML = `
    <div class="workspace-placeholder">
      <p>⏳ Loading process map...</p>
      <p style="font-size: 12px; color: #999">
        Please wait while we load "${projectName}"
      </p>
    </div>
  `;

  const apiUrl = `/api/projects/${encodeURIComponent(projectName)}/`;
  console.log("🌐 DEBUG: Fetching from API:", apiUrl);

  fetch(apiUrl)
    .then((response) => {
      console.log(
        "📡 DEBUG: Response received:",
        response.status,
        response.statusText
      );

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      return response.json();
    })
    .then((data) => {
      console.log("✅ DEBUG: Project data received:", data);
      console.log("🗺️ DEBUG: SVG URL:", data.output_map_svg);
      console.log("🖼️ DEBUG: PNG URL:", data.output_map_image);

      const mapUrl = data.output_map_svg || data.output_map_image;
      console.log("🎯 DEBUG: Selected map URL:", mapUrl);

      if (mapUrl) {
        workspace.innerHTML = `
          <div class="process-map-container">
            <div class="zoom-controls">
              <button class="zoom-btn" id="zoom-in-btn" title="Zoom In">
                <span class="zoom-icon">🔍+</span>
              </button>
              <span class="zoom-level" id="zoom-level">100%</span>
              <button class="zoom-btn" id="zoom-out-btn" title="Zoom Out">
                <span class="zoom-icon">🔍−</span>
              </button>
              <button class="zoom-btn" id="zoom-reset-btn" title="Reset Zoom">
                <span class="zoom-icon">↺</span>
              </button>
            </div>
            <div class="process-map-image" id="process-map-viewport">
              <img
                id="process-map-img"
                src="${mapUrl}"
                alt="Process Map"
              />
            </div>
          </div>
        `;

        console.log(
          "✅ DEBUG: Map HTML inserted, calling initializeZoomControls()"
        );
        initializeZoomControls();
        console.log("✅ DEBUG: Map loading complete!");
      } else if (data.status === "pending" || data.status === "running") {
        workspace.innerHTML = `
          <div class="workspace-placeholder">
            <p>⏳ Processing...</p>
            <p style="font-size: 14px; margin-top: 16px; color: #333">
              <strong>${data.original_filename}</strong>
            </p>
            <p style="font-size: 12px; color: #999; margin-top: 12px">
              ${
                data.message ||
                "Generating process map... Please wait or refresh the page."
              }
            </p>
            <div style="margin-top: 20px;">
              <div style="width: 200px; height: 4px; background: #e0e0e0; border-radius: 2px; overflow: hidden; margin: 0 auto;">
                <div style="height: 100%; background: #4285f4; width: ${
                  data.progress || 0
                }%;"></div>
              </div>
              <p style="font-size: 11px; color: #999; margin-top: 8px;">${
                data.progress || 0
              }%</p>
            </div>
          </div>
        `;
      } else {
        workspace.innerHTML = `
          <div class="workspace-placeholder">
            <p>📊 Process Map Visualization Area</p>
            <p style="font-size: 14px; margin-top: 16px; color: #333">
              <strong>${data.original_filename}</strong>
            </p>
            <p style="font-size: 12px; color: #999; margin-top: 12px">
              No process map available for this project.
            </p>
          </div>
        `;
      }
    })
    .catch((error) => {
      console.error("❌ DEBUG: Error loading project:", error);
      console.error("❌ DEBUG: Error stack:", error.stack);
      workspace.innerHTML = `
        <div class="workspace-placeholder">
          <p>❌ Error loading project</p>
          <p style="font-size: 12px; color: #999">
            ${error.message}
          </p>
        </div>
      `;
    });
}

// ============================================
// END GLOBAL FUNCTIONS
// ============================================

document.addEventListener("DOMContentLoaded", function () {
  // Tab Switching
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabContents = document.querySelectorAll(".tab-content");
  const dashboardContainer = document.querySelector(".dashboard-container");

  tabButtons.forEach((button) => {
    button.addEventListener("click", function () {
      const targetTab = this.dataset.tab;

      // Remove active class from all buttons and contents
      tabButtons.forEach((btn) => btn.classList.remove("active"));
      tabContents.forEach((content) => content.classList.remove("active"));

      // Add active class to clicked button and corresponding content
      this.classList.add("active");
      const targetContent = document.getElementById(`tab-${targetTab}`);
      if (targetContent) {
        targetContent.classList.add("active");
      }

      // Set data attribute for layout switching
      dashboardContainer.setAttribute("data-active-tab", targetTab);
      console.log(`Switched to ${targetTab} tab`);
    });
  });

  // Set initial tab state on page load
  if (dashboardContainer) {
    dashboardContainer.setAttribute("data-active-tab", "map");
  }

  // Zoom Slider - Live Update
  const zoomSlider = document.getElementById("zoom-slider");
  const zoomPercent = document.getElementById("zoom-percent");

  if (zoomSlider && zoomPercent) {
    zoomSlider.addEventListener("input", function () {
      zoomPercent.textContent = this.value + "%";
    });
  }

  // Placeholder handlers for other controls

  // Icon buttons
  const iconButtons = document.querySelectorAll(".icon-button");
  iconButtons.forEach((btn) => {
    btn.addEventListener("click", function () {
      console.log("Icon button clicked:", this.textContent);
    });
  });

  // Log dropdown - Load project's process map
  const logDropdown = document.getElementById("log-dropdown");
  console.log("🔍 DEBUG: logDropdown element found:", logDropdown);

  if (logDropdown) {
    console.log("✅ DEBUG: Adding event listener to dropdown");

    logDropdown.addEventListener("change", function () {
      const projectName = this.value;
      console.log("🎯 DEBUG: Project selected:", projectName);
      console.log("🎯 DEBUG: Project value type:", typeof projectName);
      console.log("🎯 DEBUG: Project value length:", projectName.length);

      if (projectName) {
        console.log("✅ DEBUG: Calling loadProjectMap()");
        // Load the process map for this project
        loadProjectMap(projectName);
      } else {
        console.log("❌ DEBUG: Empty project name, not loading");
      }
    });

    console.log("✅ DEBUG: Event listener added successfully");
  } else {
    console.error("❌ DEBUG: logDropdown element NOT FOUND!");
  }

  // Search input
  const searchInput = document.getElementById("search-input");
  if (searchInput) {
    searchInput.addEventListener("input", function () {
      console.log("Search query:", this.value);
    });
  }

  // Vertical sliders (Activities/Paths)
  const activitiesSlider = document.getElementById("activities-slider");
  const pathsSlider = document.getElementById("paths-slider");

  if (activitiesSlider) {
    activitiesSlider.addEventListener("input", function () {
      console.log("Activities slider:", this.value + "%");
    });
  }

  if (pathsSlider) {
    pathsSlider.addEventListener("input", function () {
      console.log("Paths slider:", this.value + "%");
    });
  }

  // Frequency dropdown
  const frequencyDropdown = document.getElementById("frequency-dropdown");
  if (frequencyDropdown) {
    frequencyDropdown.addEventListener("change", function () {
      console.log("Frequency option:", this.value);
    });
  }

  // Add secondary metrics button
  const addMetricsBtn = document.getElementById("add-metrics-btn");
  if (addMetricsBtn) {
    addMetricsBtn.addEventListener("click", function () {
      console.log("Add secondary metrics clicked");
    });
  }

  // Bottom toolbar buttons
  const toolbarButtons = document.querySelectorAll(".toolbar-button");
  toolbarButtons.forEach((btn) => {
    btn.addEventListener("click", function () {
      console.log("Toolbar button clicked:", this.textContent);
    });
  });

  // ============================================
  // STATISTICS TAB FUNCTIONALITY
  // ============================================

  // Statistics Navigation Items - Left Panel Selection
  const statNavItems = document.querySelectorAll(".stat-nav-item");
  statNavItems.forEach((item) => {
    item.addEventListener("click", function () {
      // Remove active class from all items
      statNavItems.forEach((navItem) => navItem.classList.remove("active"));

      // Add active class to clicked item
      this.classList.add("active");

      // Get the stat type
      const statType = this.dataset.stat;
      console.log("Statistics view selected:", statType);

      // Update attribute header (UI-only for now)
      const title = this.querySelector(".stat-nav-title").textContent;
      const subtitle = this.querySelector(".stat-nav-subtitle").textContent;

      const attributeTitle = document.querySelector(".attribute-title");
      const attributeSubtitle = document.querySelector(".attribute-subtitle");

      if (attributeTitle) attributeTitle.textContent = title;
      if (attributeSubtitle) attributeSubtitle.textContent = subtitle;
    });
  });

  // Segmented Control - Filter Buttons
  const segmentButtons = document.querySelectorAll(".segment-btn");
  segmentButtons.forEach((button) => {
    button.addEventListener("click", function () {
      // Remove active class from all segment buttons
      segmentButtons.forEach((btn) => btn.classList.remove("active"));

      // Add active class to clicked button
      this.classList.add("active");

      // Get the segment type
      const segmentType = this.dataset.segment;
      console.log("Segment selected:", segmentType);
    });
  });

  // Chart Tool Buttons
  const chartToolButtons = document.querySelectorAll(".chart-tool-btn");
  chartToolButtons.forEach((btn) => {
    btn.addEventListener("click", function () {
      console.log("Chart tool clicked:", this.title);
    });
  });

  // ============================================
  // CASES TAB FUNCTIONALITY
  // ============================================

  // Variant Item Click Handlers
  const variantItems = document.querySelectorAll(".case-variant-item");
  variantItems.forEach((item) => {
    item.addEventListener("click", function () {
      // Remove active class from all variant items
      variantItems.forEach((varItem) => varItem.classList.remove("active"));

      // Add active class to clicked item
      this.classList.add("active");

      // Get the variant type
      const variantType = this.dataset.variant;
      console.log("Variant selected:", variantType);

      // Update UI based on selection (placeholder for future backend integration)
      const variantTitle = this.querySelector(".variant-title").textContent;
      console.log("Variant title:", variantTitle);
    });
  });

  // Case Item Click Handlers - Update Case Header ID
  const caseItems = document.querySelectorAll(".case-list-item");
  caseItems.forEach((item) => {
    item.addEventListener("click", function () {
      // Remove active class from all case items
      caseItems.forEach((caseItem) => caseItem.classList.remove("active"));

      // Add active class to clicked item
      this.classList.add("active");

      // Get the case ID
      const caseId = this.dataset.case;
      console.log("Case selected:", caseId);

      // Update case header with the selected case ID
      const caseIdElement = document.querySelector(".case-id");
      if (caseIdElement) {
        caseIdElement.textContent = caseId;
        console.log("Updated case header to:", caseId);
      }

      // Update case subtitle
      const caseSubtitle = this.querySelector(".case-subtitle").textContent;
      const caseHeaderSubtitle = document.querySelector(
        ".case-header .case-subtitle"
      );
      if (caseHeaderSubtitle) {
        caseHeaderSubtitle.textContent = `Case with ${caseSubtitle}`;
      }
    });
  });

  // Graph/Table View Toggle Handlers
  const viewToggleButtons = document.querySelectorAll(".view-toggle-btn");
  viewToggleButtons.forEach((button) => {
    button.addEventListener("click", function () {
      // Remove active class from all toggle buttons
      viewToggleButtons.forEach((btn) => btn.classList.remove("active"));

      // Add active class to clicked button
      this.classList.add("active");

      // Get the view type
      const viewType = this.dataset.view;
      console.log("View toggled to:", viewType);

      // In future: Show/hide timeline or table based on selection
      // For now, just log the action
      if (viewType === "graph") {
        console.log("Showing graph view (timeline visualization)");
      } else if (viewType === "table") {
        console.log("Showing table view (events list)");
      }
    });
  });

  console.log("Dashboard initialized");
});

// ============================================
// FILE UPLOAD & PROCESSING MODAL
// ============================================

document.addEventListener("DOMContentLoaded", function () {
  const openBtn = document.getElementById("open-btn");
  const fileInput = document.getElementById("file-input");
  const uploadControls = document.getElementById("upload-controls");
  const selectedFilename = document.getElementById("selected-filename");
  const uploadBtn = document.getElementById("upload-btn");
  const uploadModal = document.getElementById("upload-modal");
  const modalCloseBtn = document.getElementById("modal-close-btn");
  const modalCancelBtn = document.getElementById("modal-cancel-btn");
  const modalStartBtn = document.getElementById("modal-start-btn");

  // Toggle buttons for cleaning option
  const cleaningYesBtn = document.getElementById("cleaning-yes");
  const cleaningNoBtn = document.getElementById("cleaning-no");

  // Project name input and error message
  const projectNameInput = document.getElementById("project-name");
  const projectNameError = document.getElementById("project-name-error");

  let selectedFile = null;

  // Step 1: Open button triggers file picker
  if (openBtn && fileInput) {
    openBtn.addEventListener("click", function () {
      fileInput.click();
    });
  }

  // Step 2: File selected - show filename and AUTOMATICALLY open modal
  if (fileInput) {
    fileInput.addEventListener("change", function (e) {
      const file = e.target.files[0];
      if (file) {
        selectedFile = file;
        selectedFilename.textContent = file.name;
        uploadControls.classList.add("visible");
        uploadControls.style.display = "flex";

        // Branch based on file extension
        const fileName = file.name.toLowerCase();

        if (fileName.endsWith(".csv")) {
          // CSV flow: open upload modal first, then upload after Start
          openUploadModal();
        } else if (fileName.endsWith(".xes") || fileName.endsWith(".xes.gz")) {
          // XES flow: open upload options modal (existing behavior)
          openUploadModal();
        } else {
          alert(
            "Unsupported file type. Please select a .csv, .xes, or .xes.gz file."
          );
          resetFileSelection();
        }
      }
    });
  }

  // Step 3: Upload button reopens modal (optional - if user closed it)
  if (uploadBtn) {
    uploadBtn.addEventListener("click", function () {
      if (selectedFile) {
        openUploadModal();
      }
    });
  }

  // Function to open the upload modal
  function openUploadModal() {
    if (uploadModal) {
      uploadModal.style.display = "flex";
      // Clear previous project name and error
      if (projectNameInput) {
        projectNameInput.value = "";
        projectNameInput.classList.remove("error");
      }
      if (projectNameError) {
        projectNameError.style.display = "none";
      }
      // Update Start button state
      updateStartButtonState();
    }
  }

  // Validate project name and update Start button state
  function updateStartButtonState() {
    if (!projectNameInput || !modalStartBtn) return;

    const projectName = projectNameInput.value.trim();
    if (projectName.length === 0) {
      modalStartBtn.disabled = true;
      modalStartBtn.style.opacity = "0.5";
      modalStartBtn.style.cursor = "not-allowed";
    } else {
      modalStartBtn.disabled = false;
      modalStartBtn.style.opacity = "1";
      modalStartBtn.style.cursor = "pointer";
    }
  }

  // Add input event listener to project name field
  if (projectNameInput) {
    projectNameInput.addEventListener("input", function () {
      // Hide error message when user starts typing
      if (projectNameError) {
        projectNameError.style.display = "none";
        projectNameInput.classList.remove("error");
      }
      // Update Start button state
      updateStartButtonState();
    });
  }

  // Modal close handlers
  if (modalCloseBtn) {
    modalCloseBtn.addEventListener("click", closeModal);
  }

  if (modalCancelBtn) {
    modalCancelBtn.addEventListener("click", function () {
      closeModal();
      resetFileSelection();
    });
  }

  // Close modal on overlay click
  if (uploadModal) {
    uploadModal.addEventListener("click", function (e) {
      if (e.target === uploadModal) {
        closeModal();
      }
    });
  }

  function closeModal() {
    uploadModal.style.display = "none";
  }

  // Reset file selection (called when Cancel is clicked)
  function resetFileSelection() {
    selectedFile = null;
    fileInput.value = "";
    selectedFilename.textContent = "";
    uploadControls.classList.remove("visible");
    uploadControls.style.display = "none";
    // Clear project name input
    if (projectNameInput) {
      projectNameInput.value = "";
      projectNameInput.classList.remove("error");
    }
    if (projectNameError) {
      projectNameError.style.display = "none";
    }
  }

  // Toggle button handling for cleaning option
  if (cleaningYesBtn && cleaningNoBtn) {
    cleaningYesBtn.addEventListener("click", function () {
      cleaningYesBtn.classList.add("active");
      cleaningNoBtn.classList.remove("active");
    });

    cleaningNoBtn.addEventListener("click", function () {
      cleaningNoBtn.classList.add("active");
      cleaningYesBtn.classList.remove("active");
    });
  }

  // Step 4: Start button - submit form and redirect to progress page
  if (modalStartBtn) {
    modalStartBtn.addEventListener("click", function () {
      // Validate project name
      const projectName = projectNameInput ? projectNameInput.value.trim() : "";

      if (!projectName || projectName.length === 0) {
        // Show error message
        if (projectNameError) {
          projectNameError.style.display = "block";
        }
        if (projectNameInput) {
          projectNameInput.classList.add("error");
          projectNameInput.focus();
        }
        return;
      }

      // Check if this is a CSV file
      const fileName = selectedFile ? selectedFile.name.toLowerCase() : "";
      const isCSV = fileName.endsWith(".csv");

      if (isCSV) {
        // CSV flow: upload with project name to import session
        uploadCSVForMapping(selectedFile, projectName);
      } else {
        // XES flow: create job and redirect to progress page
        const cleaningEnabled = cleaningYesBtn.classList.contains("active");
        const miningMethod = document.getElementById("mining-method").value;

        // Create FormData for file upload
        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append("project_name", projectName);
        formData.append("cleaning_enabled", cleaningEnabled);
        formData.append("mining_method", miningMethod);

        // Disable button and show loading state
        modalStartBtn.disabled = true;
        modalStartBtn.textContent = "Starting...";

        // Submit to backend
        fetch("/jobs/create/", {
          method: "POST",
          body: formData,
          headers: {
            "X-CSRFToken": getCookie("csrftoken"),
          },
        })
          .then((response) => {
            if (!response.ok) {
              // Log detailed error information
              console.error(
                "HTTP Error:",
                response.status,
                response.statusText
              );
              return response.text().then((text) => {
                console.error("Response body:", text);

                // Try to parse JSON for license errors
                try {
                  const errorData = JSON.parse(text);
                  if (errorData.error && errorData.upgrade_url) {
                    // License limitation error
                    const upgradeBtn = confirm(
                      errorData.error +
                        "\n\n" +
                        (errorData.message || "") +
                        "\n\nWould you like to upgrade now?"
                    );
                    if (upgradeBtn) {
                      window.location.href = errorData.upgrade_url;
                    }
                    throw new Error(errorData.error);
                  }
                } catch (e) {
                  // Not JSON or different error format
                }

                throw new Error(
                  `HTTP ${response.status}: ${response.statusText}`
                );
              });
            }
            return response.json();
          })
          .then((data) => {
            if (data.job_id && data.progress_url) {
              // Redirect to progress page
              window.location.href = data.progress_url;
            } else if (data.error) {
              alert("Error: " + data.error);
              modalStartBtn.disabled = false;
              modalStartBtn.textContent = "Start";
            }
          })
          .catch((error) => {
            console.error("Upload error:", error);
            alert(
              "Failed to upload file. Please try again.\n\nError: " +
                error.message +
                "\n\nCheck browser console for details."
            );
            modalStartBtn.disabled = false;
            modalStartBtn.textContent = "Start";
          });
      }
    });
  }

  // Function to upload CSV file for mapping
  function uploadCSVForMapping(file, projectName) {
    const formData = new FormData();
    formData.append("file", file);
    formData.append("project_name", projectName);

    // Disable button and show loading state
    modalStartBtn.disabled = true;
    modalStartBtn.textContent = "Uploading...";

    fetch("/import/csv/create/", {
      method: "POST",
      body: formData,
      headers: {
        "X-CSRFToken": getCookie("csrftoken"),
      },
    })
      .then((response) => {
        if (!response.ok) {
          // Try to get error message from response
          return response.text().then((text) => {
            console.error("CSV upload error response:", text);
            let errorMsg = "Upload failed";
            try {
              const data = JSON.parse(text);
              errorMsg = data.error || errorMsg;
            } catch (e) {
              // Response is not JSON (probably HTML error page)
              errorMsg = `HTTP ${response.status}: ${response.statusText}`;
            }
            throw new Error(errorMsg);
          });
        }
        return response.json();
      })
      .then((data) => {
        if (data.redirect_url) {
          // Redirect to CSV import mapping page
          window.location.href = data.redirect_url;
        } else {
          throw new Error("No redirect URL received");
        }
      })
      .catch((error) => {
        console.error("CSV upload error:", error);
        alert("Failed to upload CSV file: " + error.message);
        modalStartBtn.disabled = false;
        modalStartBtn.textContent = "Start";
      });
  }

  // Helper function to get CSRF token
  function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== "") {
      const cookies = document.cookie.split(";");
      for (let i = 0; i < cookies.length; i++) {
        const cookie = cookies[i].trim();
        if (cookie.substring(0, name.length + 1) === name + "=") {
          cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
          break;
        }
      }
    }
    return cookieValue;
  }

  // ============================================
  // ZOOM AND PAN FUNCTIONALITY FOR PROCESS MAP
  // ============================================

  // Initialize zoom controls on page load (for initial map)
  initializeZoomControls();
});
