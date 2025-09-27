function validateExcelFile(formId, inputId, requiredColumns) {
    const form = document.getElementById(formId);
    const fileInput = document.getElementById(inputId);

    form.addEventListener("submit", function (e) {
        e.preventDefault(); // stop default submission

        const file = fileInput.files[0];
        if (!file) {
            Swal.fire("Error", "Please select a file first.", "error");
            return;
        }

        const reader = new FileReader();
        reader.onload = function (evt) {
            const data = new Uint8Array(evt.target.result);
            const workbook = XLSX.read(data, { type: "array" });
            const sheet = workbook.Sheets[workbook.SheetNames[0]];
            const json = XLSX.utils.sheet_to_json(sheet, { header: 1 });

            if (json.length === 0) {
                Swal.fire("Error", "The Excel file is empty.", "error");
                return;
            }

            // Normalize headers
            const headers = json[0].map(h => h.toString().trim().toLowerCase().replace(/\s+/g, "_"));

            // Check missing columns
            const missing = requiredColumns.filter(c => !headers.includes(c));

            if (missing.length > 0) {
                Swal.fire({
                    icon: "warning",
                    title: "Data Mismatch",
                    html: `The uploaded file is missing columns:<br><b>${missing.join(", ")}</b><br>Do you want to proceed anyway?`,
                    showCancelButton: true,
                    confirmButtonText: "Proceed Anyway",
                    cancelButtonText: "Cancel"
                }).then((result) => {
                    if (result.isConfirmed) {
                        form.submit(); // proceed with upload
                    } else {
                        fileInput.value = ""; // reset file input
                    }
                });
            } else {
                form.submit(); // all good, submit normally
            }
        };
        reader.readAsArrayBuffer(file);
    });
}
