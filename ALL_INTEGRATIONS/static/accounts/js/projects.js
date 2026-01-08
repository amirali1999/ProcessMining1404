// projects.js - Project Browser page interactivity
// Handles dataset selection, details panel updates, and button interactions

// Helper function to get CSRF token from cookies
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

document.addEventListener("DOMContentLoaded", function () {
  console.log("🚀 Projects.js loaded and DOMContentLoaded fired");

  // === DOM Elements ===
  const datasetItems = document.querySelectorAll(".dataset-item");
  const workspaceDatasetName = document.querySelector(
    ".workspace-dataset-name"
  );
  const notesContent = document.querySelector(".notes-content");

  console.log("Found dataset items:", datasetItems.length);

  // Details panel fields
  const detailActivities = document.getElementById("detail-activities");
  const detailStart = document.getElementById("detail-start");
  const detailEnd = document.getElementById("detail-end");
  const detailEvents = document.getElementById("detail-events");
  const detailCases = document.getElementById("detail-cases");

  // Map preview
  const mapPreviewFrame = document.querySelector(".map-preview-frame");

  // Action buttons
  const reloadBtn = document.querySelector(".reload-btn");
  const viewBtn = document.querySelector(".details-btn");

  // Toolbar buttons
  const toolbarBtns = document.querySelectorAll(".toolbar-tool");

  console.log("Found toolbar buttons:", toolbarBtns.length);
  console.log("Reload button:", reloadBtn ? "Found" : "Not found");
  console.log("View button:", viewBtn ? "Found" : "Not found");

  // === Dataset Selection Handler ===
  datasetItems.forEach((item) => {
    item.addEventListener("click", function () {
      // Remove active class from all items
      datasetItems.forEach((i) => i.classList.remove("active"));

      // Add active class to clicked item
      this.classList.add("active");

      // Extract data attributes
      const name =
        this.dataset.name || this.querySelector(".dataset-title").textContent;
      const activities = this.dataset.activities || "N/A";
      const events = this.dataset.events || "N/A";
      const cases = this.dataset.cases || "N/A";
      const start = this.dataset.start || "N/A";
      const end = this.dataset.end || "N/A";
      const subtitle = this.querySelector(".dataset-subtitle").textContent;

      // Update workspace header
      if (workspaceDatasetName) {
        workspaceDatasetName.textContent = name;
      }

      // Update details panel
      if (detailActivities) detailActivities.textContent = activities;
      if (detailStart) detailStart.textContent = start;
      if (detailEnd) detailEnd.textContent = end;
      if (detailEvents) detailEvents.textContent = events;
      if (detailCases) detailCases.textContent = cases;

      // Update notes section
      if (notesContent) {
        const startDate = start.split(" ")[0];
        const endDate = end.split(" ")[0];
        notesContent.textContent = `Log read from file "${name}.xes" on ${subtitle}. Process contains ${cases} cases with ${events} events across ${activities} distinct activities. Time range: ${startDate} to ${endDate}.`;
      }

      // Update process map preview
      const mapUrl = this.dataset.mapUrl;
      const mapPreview = document.querySelector(".map-preview");

      console.log("🗺️ Map URL for project:", name, "=>", mapUrl);
      console.log("📊 Map preview element:", mapPreview);

      if (mapUrl && mapUrl.trim() !== "" && mapPreview) {
        console.log("✅ Loading map image from:", mapUrl);
        // Show the process map image/SVG
        mapPreview.innerHTML = `
          <div style="width: 100%; height: 100%; overflow: auto; display: flex; align-items: center; justify-content: center; background: #f9f9f9;">
            <img src="${mapUrl}" alt="Process Map for ${name}" style="max-width: 100%; max-height: 100%; object-fit: contain;" onerror="console.error('❌ Failed to load image:', '${mapUrl}')" onload="console.log('✅ Image loaded successfully')" />
          </div>
        `;
      } else {
        console.log("⚠️ No map URL available, showing placeholder");
        // Show placeholder if no map available
        mapPreview.innerHTML = `
          <div class="map-placeholder">
            <div style="text-align: center; padding: 40px;">
              <p style="font-size: 24px; margin-bottom: 10px;">📊</p>
              <p style="font-size: 14px; color: #666;">Process Map Not Available</p>
              <p style="font-size: 12px; color: #999; margin-top: 5px;">
                This project doesn't have a generated process map yet.
              </p>
            </div>
          </div>
        `;
      }

      // Add visual feedback
      this.style.animation = "none";
      setTimeout(() => {
        this.style.animation = "";
      }, 10);
    });

    // Add double-click handler to open dashboard
    item.addEventListener("dblclick", function () {
      const name =
        this.dataset.name || this.querySelector(".dataset-title").textContent;
      if (name) {
        window.location.href = `/dashboard/${encodeURIComponent(name)}/`;
      }
    });
  });

  // === Map Preview Click Handler ===
  if (mapPreviewFrame) {
    mapPreviewFrame.addEventListener("click", function () {
      const activeName =
        document.querySelector(".dataset-item.active .dataset-title")
          ?.textContent || "this dataset";
      alert(
        `Map preview for "${activeName}" would open here.\n\nIn the full implementation, this would show the process mining visualization.`
      );
    });
  }

  // === Reload Button ===
  if (reloadBtn) {
    reloadBtn.addEventListener("click", function (e) {
      e.preventDefault();
      const activeName =
        document.querySelector(".dataset-item.active .dataset-title")
          ?.textContent || "this dataset";

      // Show loading state
      this.classList.add("loading");
      const btnLabel = this.querySelector(".btn-label");
      if (btnLabel) {
        btnLabel.textContent = "Reloading...";
      }

      setTimeout(() => {
        this.classList.remove("loading");
        if (btnLabel) {
          btnLabel.textContent = "Reload";
        }
        alert(`Dataset "${activeName}" reloaded successfully!`);
      }, 800);
    });
  }

  // === View Details Button ===
  if (viewBtn) {
    viewBtn.addEventListener("click", function (e) {
      e.preventDefault();
      const activeName =
        document.querySelector(".dataset-item.active .dataset-title")
          ?.textContent || null;

      if (activeName) {
        // Redirect to dashboard for this specific project
        window.location.href = `/dashboard/${encodeURIComponent(activeName)}/`;
      } else {
        alert("لطفاً ابتدا یک پروژه را انتخاب کنید.");
      }
    });
  }

  // === Toolbar Button Handlers ===
  toolbarBtns.forEach((btn) => {
    btn.addEventListener("click", function (e) {
      e.preventDefault();

      // Store button reference at the beginning
      const currentButton = this;
      const label =
        currentButton.querySelector(".tool-label")?.textContent.trim() ||
        "Action";
      const activeName =
        document.querySelector(".dataset-item.active .dataset-title")
          ?.textContent || "selected dataset";

      console.log("Button clicked:", label, "for project:", activeName);

      switch (label) {
        case "Filter":
          alert(
            `Filter dialog for "${activeName}" would open here.\n\nOptions would include:\n- Activity filters\n- Time range filters\n- Case filters\n- Attribute filters`
          );
          break;
        case "TimeWarp":
          alert(
            `TimeWarp controls for "${activeName}" would appear here.\n\nThis allows you to:\n- Adjust time-based filters\n- Analyze process evolution\n- Compare time periods`
          );
          break;
        case "Copy":
          alert(
            `Dataset "${activeName}" copied to clipboard (metadata).\n\nIn the full version, this would copy:\n- Dataset configuration\n- Applied filters\n- Current view settings`
          );
          break;
        case "Delete":
        case "در حال حذف...":
          // Prevent double-click during deletion
          if (label === "در حال حذف...") {
            console.log("Already deleting, ignoring click");
            return;
          }

          if (
            confirm(
              `آیا مطمئن هستید که می‌خواهید پروژه "${activeName}" را حذف کنید؟\n\nاین عملیات قابل بازگشت نیست و تمام فایل‌ها و داده‌های مرتبط حذف خواهند شد.`
            )
          ) {
            const toolLabel = currentButton.querySelector(".tool-label");

            // Show loading state
            if (toolLabel) {
              toolLabel.textContent = "در حال حذف...";
            }
            currentButton.style.pointerEvents = "none";
            currentButton.style.opacity = "0.6";
            currentButton.disabled = true;

            // Get CSRF token
            const csrftoken =
              document.querySelector("[name=csrfmiddlewaretoken]")?.value ||
              getCookie("csrftoken");

            console.log("Sending delete request for:", activeName);

            // Send delete request
            fetch(`/projects/delete/${encodeURIComponent(activeName)}/`, {
              method: "POST",
              headers: {
                "X-CSRFToken": csrftoken,
                "Content-Type": "application/json",
              },
            })
              .then((response) => {
                console.log("Delete response status:", response.status);
                return response.json();
              })
              .then((data) => {
                console.log("Delete response data:", data);
                if (data.success) {
                  // Show success message briefly then reload
                  alert(
                    `✅ پروژه "${activeName}" با موفقیت حذف شد.\n\n${data.deleted_jobs} job حذف شد.`
                  );

                  // Reload page to reflect changes
                  window.location.reload();
                } else {
                  alert(`❌ خطا در حذف پروژه:\n${data.error}`);
                  // Reset button state
                  if (toolLabel) {
                    toolLabel.textContent = "Delete";
                  }
                  currentButton.style.pointerEvents = "";
                  currentButton.style.opacity = "1";
                  currentButton.disabled = false;
                }
              })
              .catch((error) => {
                console.error("Delete error:", error);
                alert(`❌ خطا در حذف پروژه:\n${error.message}`);
                // Reset button state
                if (toolLabel) {
                  toolLabel.textContent = "Delete";
                }
                currentButton.style.pointerEvents = "";
                currentButton.style.opacity = "1";
                currentButton.disabled = false;
              });
          }
          break;
        case "Export":
          // Trigger file download
          const exportUrl = `/projects/export/${encodeURIComponent(
            activeName
          )}/`;
          const exportLabel = currentButton.querySelector(".tool-label");

          // Show loading state
          if (exportLabel) {
            exportLabel.textContent = "Exporting...";
          }
          currentButton.style.pointerEvents = "none";

          // Create a hidden link and trigger download
          const link = document.createElement("a");
          link.href = exportUrl;
          link.download = ""; // Let the server specify filename
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);

          // Reset button state after a moment
          setTimeout(() => {
            if (exportLabel) {
              exportLabel.textContent = "Export";
            }
            currentButton.style.pointerEvents = "";
          }, 1000);
          break;
        default:
          alert(`"${label}" clicked for dataset: ${activeName}`);
      }

      // Add click animation
      currentButton.style.transform = "scale(0.95)";
      setTimeout(() => {
        currentButton.style.transform = "";
      }, 150);
    });
  });

  // === Hover Effects for Dataset Items ===
  datasetItems.forEach((item) => {
    item.addEventListener("mouseenter", function () {
      if (!this.classList.contains("active")) {
        this.style.transform = "translateX(4px)";
      }
    });

    item.addEventListener("mouseleave", function () {
      this.style.transform = "";
    });
  });

  // === Keyboard Navigation ===
  document.addEventListener("keydown", function (e) {
    const activeIndex = Array.from(datasetItems).findIndex((item) =>
      item.classList.contains("active")
    );

    if (e.key === "ArrowDown" && activeIndex < datasetItems.length - 1) {
      e.preventDefault();
      datasetItems[activeIndex + 1].click();
      datasetItems[activeIndex + 1].scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    } else if (e.key === "ArrowUp" && activeIndex > 0) {
      e.preventDefault();
      datasetItems[activeIndex - 1].click();
      datasetItems[activeIndex - 1].scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    } else if (e.key === "Enter" && activeIndex >= 0) {
      e.preventDefault();
      mapPreviewFrame?.click();
    }
  });

  // === Initialize with first dataset selected ===
  if (
    datasetItems.length > 0 &&
    !document.querySelector(".dataset-item.active")
  ) {
    datasetItems[0].classList.add("active");
    // Trigger click to load the map
    datasetItems[0].click();
  } else if (
    datasetItems.length > 0 &&
    document.querySelector(".dataset-item.active")
  ) {
    // If there's already an active item, trigger its click to load the map
    document.querySelector(".dataset-item.active").click();
  }

  console.log(
    "✅ Project Browser initialized with",
    datasetItems.length,
    "datasets"
  );
});
