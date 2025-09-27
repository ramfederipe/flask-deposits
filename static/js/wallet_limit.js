document.addEventListener("DOMContentLoaded", function() {

    function initDataTable(id) {
        const el = document.getElementById(id);
        if (el && !$.fn.DataTable.isDataTable('#' + id)) {
            $('#' + id).DataTable({
                dom: 'Bfrtip',
                buttons: ['csv', 'excel'],
                pageLength: 25,
                scrollX: true,
                order: [[0, 'asc']],
                responsive: true
            });
        }
    }

    // Initialize all tables
    document.querySelectorAll('.data-table').forEach(table => {
        initDataTable(table.id);
    });

    const tabButtons = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');

    function activateTab(tabId) {
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabContents.forEach(tab => tab.classList.remove('active'));

        const button = document.querySelector(`.tab-button[data-tab="${tabId}"]`);
        if (button) button.classList.add('active');

        const content = document.getElementById(tabId);
        if (content) content.classList.add('active');

        // Adjust DataTables if tab becomes visible
        content.querySelectorAll('table').forEach(tbl => {
            if ($.fn.DataTable.isDataTable('#' + tbl.id)) {
                setTimeout(() => $('#'+tbl.id).DataTable().columns.adjust().responsive.recalc(), 200);
            }
        });

        // Save active tab
        localStorage.setItem("activeTab", tabId);
    }

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            activateTab(tabId);
        });
    });

    // Restore last active tab
    const lastTab = localStorage.getItem("activeTab") || "wallet-tab";
    activateTab(lastTab);

});
