document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("uploadForm");

    form.addEventListener("submit", function (event) {
        event.preventDefault(); // stop default form submit

        const formData = new FormData(form);

        fetch(form.action, {
            method: "POST",
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === "error") {
                alert(data.message); // 🔥 shows popup
            } else {
                alert("Upload successful!");
                window.location.reload(); // refresh page if needed
            }
        })
        .catch(error => {
            alert("Something went wrong: " + error);
        });
    });
});
