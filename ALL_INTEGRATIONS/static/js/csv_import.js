// CSV Import Page JavaScript

document.addEventListener("DOMContentLoaded", function () {
  const columnHeaders = document.querySelectorAll(".column-header");
  const selectedColumnName = document.getElementById("selected-column-name");
  const startImportBtn = document.getElementById("start-import-btn");
  const encodingDropdown = document.getElementById("encoding-dropdown");
  const iconButtons = document.querySelectorAll(".header-icon-btn");

  let selectedColumn = null;
  let columnMappings = {
    caseId: null,
    activity: null,
    timestamp: null,
    resource: null,
  };

  // Column selection handler
  columnHeaders.forEach((header) => {
    header.addEventListener("click", function () {
      // Remove previous selection
      columnHeaders.forEach((h) => h.classList.remove("selected"));

      // Add selection to clicked column
      this.classList.add("selected");

      // Update selected column name
      selectedColumn = this.dataset.column;
      selectedColumnName.textContent = selectedColumn;

      console.log("Selected column:", selectedColumn);
    });
  });

  // Icon button handler for assigning column roles
  iconButtons.forEach((btn, index) => {
    btn.addEventListener("click", function () {
      if (!selectedColumn) {
        alert("Please select a column first by clicking on a column header.");
        return;
      }

      // Determine which role this button represents
      const roles = ["caseId", "activity", "timestamp", "resource"];
      const roleNames = ["Case ID", "Activity", "Timestamp", "Resource"];
      const role = roles[index];
      const roleName = roleNames[index];

      // Clear previous mapping for this role
      const prevColumn = columnMappings[role];
      if (prevColumn) {
        const prevHeader = document.querySelector(
          `.column-header[data-column="${prevColumn}"]`
        );
        if (prevHeader) {
          prevHeader.classList.remove(`mapped-${role}`);
          prevHeader.removeAttribute("data-role");
        }
      }

      // Set new mapping
      columnMappings[role] = selectedColumn;

      // Add visual indicator to the column header
      const currentHeader = document.querySelector(
        `.column-header[data-column="${selectedColumn}"]`
      );
      if (currentHeader) {
        // Remove any existing role classes
        currentHeader.classList.remove(
          "mapped-caseId",
          "mapped-activity",
          "mapped-timestamp",
          "mapped-resource"
        );
        // Add new role class
        currentHeader.classList.add(`mapped-${role}`);
        currentHeader.setAttribute("data-role", roleName);
      }

      // Add visual feedback to the button - Keep button active
      this.classList.add("active");

      console.log(`Mapped "${selectedColumn}" to ${roleName}`);
      console.log("Current mappings:", columnMappings);

      // Show confirmation message temporarily
      const roleEmojis = ["👤", "⚙️", "🕐", "👥"];
      const originalText = selectedColumnName.textContent;
      selectedColumnName.textContent = `${selectedColumn} → ${roleEmojis[index]} ${roleName}`;

      // Clear the confirmation message after a moment, but keep column selected
      setTimeout(() => {
        // Only clear if user hasn't selected another column
        if (selectedColumnName.textContent.includes("→")) {
          selectedColumnName.textContent = selectedColumn;
        }
      }, 1500);
    });
  });

  // Encoding dropdown change handler
  if (encodingDropdown) {
    encodingDropdown.addEventListener("change", function () {
      console.log("Encoding changed to:", this.value);
      // TODO: Reload CSV with new encoding
    });
  }

  // Start import button handler
  if (startImportBtn) {
    startImportBtn.addEventListener("click", function () {
      // Check if at least case ID and activity are mapped
      if (!columnMappings.caseId || !columnMappings.activity) {
        alert(
          "Please map at least the Case ID (👤) and Activity (⚙️) columns before importing."
        );
        return;
      }

      // Disable button to prevent double-clicks
      startImportBtn.disabled = true;
      startImportBtn.textContent = "Processing...";

      // Get session ID from URL
      const urlParts = window.location.pathname.split("/");
      const sessionId = urlParts[urlParts.indexOf("csv") + 1];

      // Send mappings to backend (discovery options will be set in next step)
      fetch(`/import/csv/${sessionId}/process/`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCookie("csrftoken"),
        },
        body: JSON.stringify({
          columnMappings: columnMappings,
        }),
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.success) {
            // Redirect to progress page to watch the discovery process
            window.location.href = data.redirect_url;
          } else {
            // Show error message
            alert(`❌ Error: ${data.error || "Unknown error occurred"}`);
            startImportBtn.disabled = false;
            startImportBtn.textContent = "Start import & discovery";
          }
        })
        .catch((error) => {
          console.error("Import error:", error);
          alert(`❌ Error: ${error.message || "Failed to import CSV"}`);
          startImportBtn.disabled = false;
          startImportBtn.textContent = "Start import & discovery";
        });
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

  console.log("CSV Import page initialized");
});
