export function setupColumnToggle(tableSelector, selectId, toggleBtnId, storageKey) {
    const table = document.querySelector(tableSelector);
    const select = document.getElementById(selectId);
    const toggleBtn = document.getElementById(toggleBtnId);

    if (!table || !select || !toggleBtn) return;

    let hiddenColumns = JSON.parse(localStorage.getItem(storageKey) || "[]");
    const ths = table.querySelectorAll("thead th");

    // Hide a column
    function hideColumn(idx) {
        ths[idx].style.display = "none";
        table.querySelectorAll("tbody tr").forEach(row => {
            if (row.cells[idx]) row.cells[idx].style.display = "none";
        });
    }

    // Show a column
    function showColumn(idx) {
        ths[idx].style.display = "";
        table.querySelectorAll("tbody tr").forEach(row => {
            if (row.cells[idx]) row.cells[idx].style.display = "";
        });
    }

    // Get column index by name
    function getColumnIndex(name) {
        let index = -1;
        ths.forEach((th, idx) => {
            const text = th.textContent.trim();
            if (text === name.trim()) index = idx;
        });
        return index;
    }

    // Update toggle button text
    function updateButton() {
        const colName = select.value;
        if (!colName) {
            toggleBtn.textContent = "Select column";
            toggleBtn.disabled = true;
            return;
        }
        toggleBtn.disabled = false;
        toggleBtn.textContent = hiddenColumns.includes(colName) ? "Unhide Column" : "Hide Column";
    }

    // Update dropdown marks
    function updateSelectMarks() {
        Array.from(select.options).forEach(opt => {
            if (!opt.value) return;
            opt.textContent = hiddenColumns.includes(opt.value) ? opt.value + " ⚫" : opt.value;
        });
    }

    // Apply hidden columns on load
    hiddenColumns.forEach(name => {
        const idx = getColumnIndex(name);
        if (idx !== -1) hideColumn(idx);
    });

    updateButton();
    updateSelectMarks();

    // Update button when selection changes
    select.addEventListener("change", updateButton);

    // Toggle hide/unhide on button click
    toggleBtn.addEventListener("click", () => {
        const colName = select.value;
        if (!colName) return;
        const idx = getColumnIndex(colName);
        if (idx === -1) return alert("Column not found!");

        if (hiddenColumns.includes(colName)) {
            // Unhide
            hiddenColumns = hiddenColumns.filter(c => c !== colName);
            showColumn(idx);
        } else {
            // Hide
            hiddenColumns.push(colName);
            hideColumn(idx);
        }

        localStorage.setItem(storageKey, JSON.stringify(hiddenColumns));
        updateButton();
        updateSelectMarks();
    });
}
