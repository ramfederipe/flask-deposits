// depositFilters.js
export function setupDepositFilters(tableSelector, options = {}) {
    const table = $(tableSelector).DataTable({
        pageLength: 25,
        dom: 'Bfrtip',
        buttons: [
            { extend: 'csvHtml5', text: 'Export CSV', className: 'bg-green-600 text-white p-2 rounded' },
            { extend: 'excelHtml5', text: 'Export Excel', className: 'bg-blue-600 text-white p-2 rounded' }
        ]
    });

    // Column hide/show
    const hiddenColumnsKey = options.hiddenColumnsKey || 'hiddenColumnsDeposit';
    const hideSelect = $('#hideColumnSelect');
    const toggleBtn = $('#toggleColumnBtn');

    const hiddenColumns = JSON.parse(localStorage.getItem(hiddenColumnsKey)) || [];
    hiddenColumns.forEach(idx => table.column(idx).visible(false));

    toggleBtn.on('click', function() {
        const colName = hideSelect.val();
        if (!colName) return;

        let colIndex = -1;
        table.columns().every(function(idx) {
            if ($(this.header()).text().trim() === colName) colIndex = idx;
        });
        if (colIndex === -1) { alert("Column not found!"); return; }

        const visible = table.column(colIndex).visible();
        table.column(colIndex).visible(!visible);

        let hidden = JSON.parse(localStorage.getItem(hiddenColumnsKey)) || [];
        if (!visible) { hidden = hidden.filter(i => i !== colIndex); }
        else { if (!hidden.includes(colIndex)) hidden.push(colIndex); }
        localStorage.setItem(hiddenColumnsKey, JSON.stringify(hidden));
    });

    // Filters
    $('#searchInput').on('keyup', () => table.search($('#searchInput').val()).draw());
    $('#merchantFilter').on('change', () => table.column(options.columns.merchant).search($('#merchantFilter').val()).draw());
    $('#bankFilter').on('change', () => table.column(options.columns.bank).search($('#bankFilter').val()).draw());
    $('#statusFilter').on('change', () => table.column(options.columns.status).search($('#statusFilter').val()).draw());
    $('#depositTypeFilter').on('change', () => table.column(options.columns.deposit_type).search($('#depositTypeFilter').val()).draw());

    // Date range filters
    $.fn.dataTable.ext.search.push((settings, data) => {
        const createdStart = $('#createdStart').val();
        const createdEnd = $('#createdEnd').val();
        const updatedStart = $('#updatedStart').val();
        const updatedEnd = $('#updatedEnd').val();

        const createdVal = new Date(data[options.columns.created_time]);
        const updatedVal = new Date(data[options.columns.updated_time]);

        if (createdStart && createdVal < new Date(createdStart)) return false;
        if (createdEnd && createdVal > new Date(createdEnd)) return false;
        if (updatedStart && updatedVal < new Date(updatedStart)) return false;
        if (updatedEnd && updatedVal > new Date(updatedEnd)) return false;

        return true;
    });

    $('#createdStart, #createdEnd, #updatedStart, #updatedEnd').on('change', () => table.draw());

    // External Export button
    if (options.exportBtn) {
        $(options.exportBtn).on('click', () => table.button('.buttons-csv').trigger());
    }
}
