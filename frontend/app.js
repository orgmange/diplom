const API_URL = "/api/v1";

function renderResultItem(filename, text, subtext = "") {
    const item = document.createElement("div");
    item.className = "result-item";
    item.innerHTML = `
        <div class="filename">${filename} <span class="score">${subtext}</span></div>
        <div class="preview">${text}</div>
    `;
    return item;
}

async function updateStatus() {
    try {
        const response = await fetch(`${API_URL}/status`);
        const data = await response.json();
        document.getElementById("status-display").innerHTML = `
            Files processed (OCR): ${data.ocr_files}<br>
            Files cleaned: ${data.cleaned_files}<br>
            Vectorized docs (OCR): ${data.vectorized_count}<br>
            <b>Current Embedding: ${data.current_model}</b>
        `;
    } catch (error) {
        console.error("Error fetching status:", error);
        document.getElementById("status-display").innerText = "Ошибка соединения с API";
    }
}

async function updateModels() {
    const selector = document.getElementById("model-selector");
    const checkboxContainer = document.getElementById("llm-benchmark-model-checkboxes");
    if (!selector) return;
    try {
        const response = await fetch(`${API_URL}/rag/models`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        const currentVal = selector.value;
        selector.innerHTML = "";

        data.models.forEach(model => {
            const opt = document.createElement("option");
            opt.value = model;
            opt.innerText = model;
            if (model === currentVal) opt.selected = true;
            selector.appendChild(opt);
        });

        if (checkboxContainer) {
            checkboxContainer.innerHTML = "";
            data.models.forEach(model => {
                const label = document.createElement("label");
                label.className = "checkbox-label";
                const cb = document.createElement("input");
                cb.type = "checkbox";
                cb.value = model;
                cb.className = "model-checkbox";
                label.appendChild(cb);
                label.appendChild(document.createTextNode(" " + model));
                checkboxContainer.appendChild(label);
            });
        }
    } catch (error) {
        console.error("Error fetching models:", error);
        selector.innerHTML = `<option value="">Ошибка: ${error.message}</option>`;
        if (checkboxContainer) checkboxContainer.innerText = `Ошибка: ${error.message}`;
    }
}

function toggleAllModelCheckboxes(checked) {
    const boxes = document.querySelectorAll("#llm-benchmark-model-checkboxes .model-checkbox");
    boxes.forEach(cb => cb.checked = checked);
}

function getSelectedBenchmarkModels() {
    const boxes = document.querySelectorAll("#llm-benchmark-model-checkboxes .model-checkbox:checked");
    return Array.from(boxes).map(cb => cb.value);
}

async function cancelBenchmark() {
    try {
        const response = await fetch(`${API_URL}/rag/benchmark/cancel`, { method: "POST" });
        const data = await response.json();
        console.log("Cancellation requested:", data);
    } catch (error) {
        console.error("Error cancelling benchmark:", error);
    }
}

async function skipStructuringModel() {
    try {
        const response = await fetch(`${API_URL}/rag/benchmark/structuring/skip`, { method: "POST" });
        const data = await response.json();
        console.log("Skip requested:", data);
    } catch (error) {
        console.error("Error skipping model:", error);
    }
}

let benchmarkProgressInterval = null;

async function pollStructuringProgress() {
    const progressEl = document.getElementById("struct-benchmark-progress");
    const modelNameEl = document.getElementById("progress-model-name");
    const fileCountEl = document.getElementById("progress-file-count");
    const barFillEl = document.getElementById("progress-bar-fill");
    const currentFileEl = document.getElementById("progress-current-file");
    const streamBoxEl = document.getElementById("struct-benchmark-live-stream");
    const streamContentEl = document.getElementById("live-stream-content");

    try {
        const response = await fetch(`${API_URL}/rag/benchmark/structuring/progress`);
        const data = await response.json();

        if (data.is_running) {
            progressEl.classList.remove("hidden");
            modelNameEl.innerText = `Модель: ${data.current_model || "Загрузка..."}`;
            fileCountEl.innerText = `Файл: ${data.processed_files}/${data.total_files}`;

            const percent = data.total_files > 0 ? (data.processed_files / data.total_files * 100) : 0;
            barFillEl.style.width = `${percent}%`;
            currentFileEl.innerText = data.current_file ? `Обработка: ${data.current_file}` : "Ожидание...";

            if (data.current_stream && data.current_stream.length > 0) {
                streamBoxEl.classList.remove("hidden");
                streamContentEl.innerText = data.current_stream;
                streamBoxEl.scrollTop = streamBoxEl.scrollHeight;
            } else {
                streamContentEl.innerText = "";
            }
        } else {
            stopProgressPolling();
        }
    } catch (error) {
        console.error("Error polling progress:", error);
    }
}

function startProgressPolling() {
    stopProgressPolling();
    benchmarkProgressInterval = setInterval(pollStructuringProgress, 1000);
}

function stopProgressPolling() {
    if (benchmarkProgressInterval) {
        clearInterval(benchmarkProgressInterval);
        benchmarkProgressInterval = null;
    }
    const progressEl = document.getElementById("struct-benchmark-progress");
    if (progressEl) progressEl.classList.add("hidden");
    const streamBoxEl = document.getElementById("struct-benchmark-live-stream");
    if (streamBoxEl) streamBoxEl.classList.add("hidden");
}

async function runStructuringBenchmark() {
    const btn = document.getElementById("run-struct-benchmark-btn");
    const cancelBtn = document.getElementById("cancel-struct-benchmark-btn");
    const embedSelector = document.getElementById("llm-benchmark-embedding-selector");
    const summary = document.getElementById("struct-benchmark-summary");
    const resultsDiv = document.getElementById("struct-benchmark-results");

    const modelNames = getSelectedBenchmarkModels();
    const embeddingModel = embedSelector ? embedSelector.value : null;
    if (modelNames.length === 0) {
        alert("Выбери хотя бы одну LLM модель");
        return;
    }

    btn.disabled = true;
    btn.innerText = "Выполняется...";
    if (cancelBtn) cancelBtn.classList.remove("hidden");
    const skipBtn = document.getElementById("skip-struct-benchmark-btn");
    if (skipBtn) skipBtn.classList.remove("hidden");

    summary.innerHTML = `Запуск бенчмарка для ${modelNames.length} модел${modelNames.length === 1 ? "и" : "ей"}: ${modelNames.join(", ")}...`;
    resultsDiv.innerText = "";

    startProgressPolling();

    const temperature = parseFloat(document.getElementById("bench-temp")?.value || 0.0);
    const numCtx = parseInt(document.getElementById("bench-ctx")?.value || 16383);
    const timeout = parseInt(document.getElementById("bench-timeout")?.value || 60);
    const structuredOutput = document.getElementById("bench-structured")?.checked ?? true;

    try {
        const response = await fetch(`${API_URL}/rag/benchmark/structuring/run-multi`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                model_names: modelNames,
                embedding_model: embeddingModel,
                temperature: temperature,
                num_ctx: numCtx,
                timeout: timeout,
                structured_output: structuredOutput
            })
        });
        const reports = await response.json();

        stopProgressPolling();

        if (!Array.isArray(reports) || reports.length === 0) {
            summary.innerText = "Нет результатов.";
            return;
        }

        let summaryHtml = `<h3>Сводка по ${reports.length} модел${reports.length === 1 ? "и" : "ям"}</h3>`;
        summaryHtml += `<table class="benchmark-table"><thead><tr>
            <th>Модель</th><th>Параметры</th><th>Файлов</th><th>Шаблон</th>
            <th>Precision</th><th>Recall</th><th>F1</th>
            <th>Fuzzy</th><th>Время (с)</th>
        </tr></thead><tbody>`;
        reports.forEach(report => {
            const paramsText = `T=${report.temperature}, Ctx=${report.num_ctx}, TO=${report.timeout}, Struct=${report.structured_output ? 'ON' : 'OFF'}`;
            summaryHtml += `<tr>
                <td><b>${report.model_name}</b></td>
                <td><small>${paramsText}</small></td>
                <td>${report.total_files}</td>
                <td>${(report.template_accuracy * 100).toFixed(1)}% (${report.correct_templates_count}/${report.total_files})</td>
                <td>${(report.avg_precision * 100).toFixed(1)}%</td>
                <td>${(report.avg_recall * 100).toFixed(1)}%</td>
                <td><b>${(report.avg_f1 * 100).toFixed(1)}%</b></td>
                <td>${(report.avg_fuzzy_score * 100).toFixed(1)}%</td>
                <td>${report.avg_processing_time}</td>
            </tr>`;
        });
        summaryHtml += `</tbody></table>`;
        summary.innerHTML = summaryHtml;

        let detailsHtml = "";
        reports.forEach(report => {
            detailsHtml += `<h3>Детали: ${report.model_name}</h3>`;
            detailsHtml += `<div id="struct-detail-${report.model_name.replace(/[^a-zA-Z0-9]/g, '_')}"></div>`;
        });
        resultsDiv.innerHTML = detailsHtml;

        reports.forEach(report => {
            const safeId = `struct-detail-${report.model_name.replace(/[^a-zA-Z0-9]/g, '_')}`;
            renderStructuringBenchmarkTable(safeId, report.items);
        });

        updateBenchmarkHistory();
    } catch (error) {
        summary.innerText = `Ошибка бенчмарка: ${error.message}`;
    } finally {
        btn.disabled = false;
        btn.innerText = "Запустить LLM бенчмарк";
        if (cancelBtn) cancelBtn.classList.add("hidden");
        const skipBtn = document.getElementById("skip-struct-benchmark-btn");
        if (skipBtn) skipBtn.classList.add("hidden");
        stopProgressPolling();
    }
}

let currentHistorySort = "timestamp";
let currentHistorySortDesc = true;

function setHistorySort(column) {
    if (currentHistorySort === column) {
        currentHistorySortDesc = !currentHistorySortDesc;
    } else {
        currentHistorySort = column;
        currentHistorySortDesc = true;
    }
    updateBenchmarkHistory();
}

async function updateBenchmarkHistory() {
    const listDiv = document.getElementById("history-list");
    if (!listDiv) return;

    try {
        const response = await fetch(`${API_URL}/rag/benchmark/structuring/reports`);
        let reports = await response.json();

        if (reports.length === 0) {
            listDiv.innerText = "История пуста";
            return;
        }

        // Client-side sorting
        reports.sort((a, b) => {
            let valA = a[currentHistorySort];
            let valB = b[currentHistorySort];

            if (currentHistorySort === "model_name") {
                valA = valA || "";
                valB = valB || "";
                if (valA < valB) return currentHistorySortDesc ? 1 : -1;
                if (valA > valB) return currentHistorySortDesc ? -1 : 1;
                return 0;
            }

            if (valA === undefined) valA = 0;
            if (valB === undefined) valB = 0;

            return currentHistorySortDesc ? (valB - valA) : (valA - valB);
        });

        const getSortIcon = (col) => {
            if (currentHistorySort !== col) return "";
            return currentHistorySortDesc ? " ↓" : " ↑";
        };

        let tableHtml = `<table class="benchmark-table" style="width: 100%; text-align: center; margin-top: 10px;">
            <thead>
                <tr>
                    <th style="text-align: left; cursor: pointer;" onclick="setHistorySort('model_name')">Модель${getSortIcon('model_name')}</th>
                    <th style="cursor: pointer;" onclick="setHistorySort('total_files')">Файлов${getSortIcon('total_files')}</th>
                    <th style="cursor: pointer;" onclick="setHistorySort('precision')">Precision${getSortIcon('precision')}</th>
                    <th style="cursor: pointer;" onclick="setHistorySort('recall')">Recall${getSortIcon('recall')}</th>
                    <th style="cursor: pointer;" onclick="setHistorySort('f1')">F1${getSortIcon('f1')}</th>
                    <th style="cursor: pointer;" onclick="setHistorySort('fuzzy')">Fuzzy${getSortIcon('fuzzy')}</th>
                    <th style="cursor: pointer;" onclick="setHistorySort('timestamp')">Дата${getSortIcon('timestamp')}</th>
                    <th></th>
                </tr>
            </thead>
            <tbody>`;

        tableHtml += reports.map(r => {
            const date = new Date(r.timestamp * 1000).toLocaleString();
            const prec = ((r.precision || 0) * 100).toFixed(1);
            const rec = ((r.recall || 0) * 100).toFixed(1);
            const f1 = ((r.f1 || 0) * 100).toFixed(1);
            const fuzzy = ((r.fuzzy || 0) * 100).toFixed(1);

            return `
                <tr class="history-item" style="cursor: pointer;" onclick="loadReport('${r.filename}')">
                    <td style="text-align: left; padding: 10px;"><b>${r.model_name}</b></td>
                    <td style="padding: 10px;">${r.total_files}</td>
                    <td style="padding: 10px;">${prec}%</td>
                    <td style="padding: 10px;">${rec}%</td>
                    <td style="padding: 10px;"><b>${f1}%</b></td>
                    <td style="padding: 10px;">${fuzzy}%</td>
                    <td style="padding: 10px;"><small>${date}</small></td>
                    <td style="padding: 10px; text-align: right;">
                        <button class="delete-history-btn" style="position: static; display: inline-block;" onclick="deleteReport('${r.filename}', event)" title="Удалить этот отчет">×</button>
                    </td>
                </tr>
            `;
        }).join("");

        tableHtml += `</tbody></table>`;
        listDiv.innerHTML = tableHtml;
    } catch (error) {
        console.error("Error loading history:", error);
        listDiv.innerText = "Ошибка загрузки истории";
    }
}

async function loadReport(filename) {
    const summary = document.getElementById("struct-benchmark-summary");
    const resultsDiv = document.getElementById("struct-benchmark-results");

    summary.innerHTML = `<span class="loading">Загрузка отчета ${filename}...</span>`;
    resultsDiv.innerHTML = "";

    try {
        const response = await fetch(`${API_URL}/rag/benchmark/structuring/reports/${filename}`);
        const report = await response.json();

        summary.innerHTML = `
            <i>Загружен исторический отчет: ${filename}</i><br>
            Модель: <b>${report.model_name}</b><br>
            Обработано файлов: ${report.total_files}<br>
            С эталоном: ${report.files_with_reference}<br>
            Точность шаблона: <b>${(report.template_accuracy * 100).toFixed(1)}%</b> (${report.correct_templates_count}/${report.total_files})<br>
            Среднее время: ${report.avg_processing_time} сек.<br>
            Параметры: <small>T=${report.temperature}, Ctx=${report.num_ctx}, TO=${report.timeout}, Struct=${report.structured_output ? 'ON' : 'OFF'}</small><br>
            <b>Precision: ${((report.avg_precision || 0) * 100).toFixed(1)}% | 
            Recall: ${((report.avg_recall || 0) * 100).toFixed(1)}% | 
            F1: ${((report.avg_f1 || 0) * 100).toFixed(1)}%</b><br>
            Fuzzy: ${((report.avg_fuzzy_score || 0) * 100).toFixed(1)}%
        `;

        renderStructuringBenchmarkTable("struct-benchmark-results", report.items);
        window.scrollTo({ top: summary.offsetTop - 20, behavior: 'smooth' });
    } catch (error) {
        summary.innerText = `Ошибка загрузки отчета: ${error.message}`;
    }
}

async function deleteReport(filename, event) {
    if (event) event.stopPropagation();
    if (!confirm(`Удалить этот отчет (${filename})?`)) return;

    try {
        const response = await fetch(`${API_URL}/rag/benchmark/structuring/reports/${filename}`, {
            method: "DELETE"
        });
        if (response.ok) {
            updateBenchmarkHistory();
        } else {
            alert("Ошибка при удалении отчета");
        }
    } catch (error) {
        console.error("Error deleting report:", error);
    }
}

async function clearBenchmarkHistory() {
    if (!confirm("Вы уверены, что хотите удалить ВСЮ историю запусков? Это действие необратимо.")) return;

    try {
        const response = await fetch(`${API_URL}/rag/benchmark/structuring/reports`, {
            method: "DELETE"
        });
        if (response.ok) {
            updateBenchmarkHistory();
            document.getElementById("struct-benchmark-summary").innerText = "История очищена.";
            document.getElementById("struct-benchmark-results").innerHTML = "";
        } else {
            alert("Ошибка при очистке истории");
        }
    } catch (error) {
        console.error("Error clearing history:", error);
    }
}

function renderStructuringBenchmarkTable(containerId, items) {
    const el = document.getElementById(containerId);
    if (!items || items.length === 0) {
        el.innerText = "Нет данных";
        return;
    }

    const rows = items.map(item => {
        const rowClass = item.is_reference_found ? (item.f1 > 0.9 ? "ok-row" : "bad-row") : "neutral-row";
        const precText = item.is_reference_found ? `${((item.precision || 0) * 100).toFixed(0)}%` : "N/A";
        const recallText = item.is_reference_found ? `${((item.recall || 0) * 100).toFixed(0)}%` : "N/A";
        const f1Text = item.is_reference_found ? `${(item.f1 * 100).toFixed(0)}%` : "N/A";
        const fuzzyText = item.is_reference_found ? `${(item.avg_fuzzy_score * 100).toFixed(0)}%` : "N/A";

        const typeMatchHtml = item.is_type_correct
            ? `<span class="type-tag type-ok">OK</span>`
            : `<span class="type-tag type-err">ERR (${item.detected_type})</span>`;

        let tooltipContent = "Сравнение полей:\n\n";
        if (item.field_metrics && item.field_metrics.length > 0) {
            item.field_metrics.forEach(fm => {
                const status = fm.is_exact_match ? "✓" : "✗";
                tooltipContent += `${status} ${fm.field_name}: ${fm.actual}\n   (Exp: ${fm.expected})\n\n`;
            });
        } else if (item.reference_json) {
            const keys = Object.keys(item.reference_json);
            keys.forEach(key => {
                const expected = item.reference_json[key];
                const actual = item.result_json[key];
                const formatVal = (val) => {
                    if (val === null || val === undefined) return 'MISSING';
                    if (typeof val === 'object') return JSON.stringify(val);
                    return String(val);
                };
                tooltipContent += `${key}: ${formatVal(actual)}\n   (Exp: ${formatVal(expected)})\n\n`;
            });
        } else {
            tooltipContent = "Эталонный JSON не найден";
        }

        return `
            <tr class="${rowClass}">
                <td class="tooltip">
                    ${item.filename}
                    <span class="tooltiptext">${tooltipContent}</span>
                </td>
                <td>
                    ${item.expected_type}
                    ${typeMatchHtml}
                </td>
                <td>${item.processing_time} сек.</td>
                <td>${precText}</td>
                <td>${recallText}</td>
                <td><b>${f1Text}</b></td>
                <td>${fuzzyText}</td>
            </tr>
        `;
    }).join("");

    el.innerHTML = `
        <table class="benchmark-table">
            <thead>
                <tr>
                    <th>Файл (наведи для деталей)</th>
                    <th>Тип (Ожидаемый)</th>
                    <th>Время</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                    <th>Fuzzy</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

async function runFullReindex() {
    const model = document.getElementById("model-selector").value;
    const resultDiv = document.getElementById("structure-result");

    if (!confirm(`Вы уверены, что хотите очистить базу и переиндексировать всё с помощью ${model}?`)) {
        return;
    }

    resultDiv.innerHTML = `<span class="loading">🔄 Выполняется полная переиндексация с ${model}...</span>`;

    try {
        const response = await fetch(`${API_URL}/rag/reindex`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ model_name: model })
        });
        const data = await response.json();
        resultDiv.innerHTML = `
            <div style="color: green; font-weight: bold;">✅ Переиндексация завершена!</div>
            Модель: ${data.embedding_model}<br>
            Проиндексировано примеров: ${data.indexed_examples}<br>
            <b>Всего векторов: ${data.total}</b>
        `;
    } catch (error) {
        resultDiv.innerText = "Ошибка переиндексации: " + error.message;
    } finally {
        updateStatus();
    }
}

async function runOCR() {
    const btn = document.querySelector("button[onclick='runOCR()']");
    const resultDiv = document.getElementById("ocr-result");
    btn.disabled = true;
    btn.innerText = "Выполняется...";
    resultDiv.innerHTML = "Сканирование изображений...";

    try {
        const response = await fetch(`${API_URL}/ocr/scan`, { method: "POST" });
        const data = await response.json();
        const processedFiles = Array.isArray(data.processed_files) ? data.processed_files : [];
        resultDiv.innerHTML = "";

        if (processedFiles.length === 0) {
            resultDiv.innerText = "Новых изображений не найдено.";
        } else {
            processedFiles.forEach(file => {
                resultDiv.appendChild(renderResultItem(file, "Задача на распознавание создана и завершена.", "OK"));
            });
        }
    } catch (error) {
        resultDiv.innerText = "Ошибка: " + error.message;
    } finally {
        btn.disabled = false;
        btn.innerText = "Запустить OCR";
        updateStatus();
    }
}

async function runClean() {
    const btn = document.querySelector("button[onclick='runClean()']");
    const resultDiv = document.getElementById("clean-result");
    btn.disabled = true;
    btn.innerText = "Выполняется...";
    resultDiv.innerHTML = "Очистка XML файлов...";

    try {
        const response = await fetch(`${API_URL}/ocr/clean`, { method: "POST" });
        const data = await response.json();
        const cleanedFiles = Array.isArray(data.cleaned_files) ? data.cleaned_files : [];
        resultDiv.innerHTML = "";

        if (cleanedFiles.length === 0) {
            resultDiv.innerText = "Новых XML для очистки не найдено.";
        } else {
            cleanedFiles.forEach(file => {
                resultDiv.appendChild(renderResultItem(file.filename, file.snippet, "Cleaned"));
            });
        }
    } catch (error) {
        resultDiv.innerText = "Ошибка: " + error.message;
    } finally {
        btn.disabled = false;
        btn.innerText = "Очистить данные";
        updateStatus();
    }
}

async function runIndex() {
    const resultDiv = document.getElementById("index-result");
    resultDiv.innerHTML = "Векторизация и сохранение в Qdrant...";

    try {
        const response = await fetch(`${API_URL}/rag/index`, { method: "POST" });
        const data = await response.json();
        resultDiv.innerHTML = "";

        if (data.indexed_files.length === 0) {
            resultDiv.innerText = "Нет новых файлов для индексации.";
        } else {
            data.indexed_files.forEach(file => {
                resultDiv.appendChild(renderResultItem(file, "Документ успешно проиндексирован.", "Indexed"));
            });
        }
    } catch (error) {
        resultDiv.innerText = "Ошибка: " + error.message;
    } finally {
        updateStatus();
    }
}

async function runIndexExamples() {
    const resultDiv = document.getElementById("index-result");
    resultDiv.innerHTML = "Индексация примеров из data/examples...";

    try {
        const response = await fetch(`${API_URL}/rag/index_examples`, { method: "POST" });
        const data = await response.json();
        resultDiv.innerHTML = `Проиндексировано примеров: ${data.indexed_examples.length}`;
    } catch (error) {
        resultDiv.innerText = "Ошибка: " + error.message;
    } finally {
        updateStatus();
    }
}

async function runSearch() {
    const query = document.getElementById("search-query").value;
    if (!query) return;

    const resultsDiv = document.getElementById("search-results");
    resultsDiv.innerHTML = "Поиск по смыслу...";

    try {
        const response = await fetch(`${API_URL}/rag/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                query: query,
                limit: 5,
                doc_type: null // Можно добавить селектор типов позже
            })
        });
        const results = await response.json();

        resultsDiv.innerHTML = "";
        if (results.length === 0) {
            resultsDiv.innerText = "Ничего не найдено.";
            return;
        }

        results.forEach(res => {
            const scorePercent = (res.score * 100).toFixed(1) + "%";
            const previewText = res.raw_text || res.cleaned_text || res.text || "Нет текста для превью";
            resultsDiv.appendChild(renderResultItem(res.filename, previewText, scorePercent));
        });
    } catch (error) {
        resultsDiv.innerText = "Ошибка поиска: " + error.message;
    }
}

function renderBenchmarkTable(containerId, group) {
    const el = document.getElementById(containerId);
    if (!group || !group.items || group.items.length === 0) {
        el.innerText = "Нет данных";
        return;
    }
    const rows = group.items.map(item => {
        let altHtml = "";
        if (item.alternatives && item.alternatives.length > 0) {
            altHtml = '<div class="alt-list">' + item.alternatives.map(a =>
                `<span>${a.type}: ${(a.score * 100).toFixed(1)}%</span>`
            ).join("") + '</div>';
        }

        return `
            <tr class="${item.is_correct ? "ok-row" : "bad-row"}">
                <td>${item.filename}</td>
                <td>${item.expected_type || "-"}</td>
                <td>
                    <b>${item.predicted_type || "-"}</b>
                    ${altHtml}
                </td>
                <td>${item.predicted_filename || "-"}</td>
                <td>${item.score !== null && item.score !== undefined ? (item.score * 100).toFixed(1) + "%" : "-"}</td>
                <td>${item.is_correct ? "OK" : "ERR"}</td>
            </tr>
        `;
    }).join("");
    el.innerHTML = `
        <table class="benchmark-table">
            <thead>
                <tr>
                    <th>Файл</th>
                    <th>Ожидали</th>
                    <th>Предсказали (Топ-3)</th>
                    <th>Шаблон</th>
                    <th>Схожесть</th>
                    <th>Статус</th>
                </tr>
            </thead>
            <tbody>${rows}</tbody>
        </table>
    `;
}

async function updateBenchmarkModels() {
    const selector = document.getElementById("embedding-model-selector");
    const llmEmbedSelector = document.getElementById("llm-benchmark-embedding-selector");
    try {
        const response = await fetch(`${API_URL}/rag/benchmark/models`);
        const data = await response.json();
        if (selector) {
            selector.innerHTML = "";
            data.models.forEach(model => {
                const opt = document.createElement("option");
                opt.value = model;
                opt.innerText = model;
                selector.appendChild(opt);
            });
        }
        if (llmEmbedSelector) {
            llmEmbedSelector.innerHTML = "";
            data.models.forEach(model => {
                const opt = document.createElement("option");
                opt.value = model;
                opt.innerText = model;
                llmEmbedSelector.appendChild(opt);
            });
        }
    } catch (error) {
        if (selector) selector.innerHTML = `<option value="">Ошибка загрузки: ${error.message}</option>`;
    }
}

async function runEmbeddingBenchmark() {
    const btn = document.getElementById("run-benchmark-btn");
    const cancelBtn = document.getElementById("cancel-embedding-benchmark-btn");
    const selector = document.getElementById("embedding-model-selector");
    const summary = document.getElementById("benchmark-summary");
    const rawBox = document.getElementById("benchmark-raw");
    const cleanBox = document.getElementById("benchmark-clean");
    const embeddingModel = selector.value;
    if (!embeddingModel) {
        alert("Выбери embedding-модель");
        return;
    }
    btn.disabled = true;
    btn.innerText = "Тестируется...";
    if (cancelBtn) cancelBtn.classList.remove("hidden");
    summary.innerText = "Запуск пайплайна...";
    rawBox.innerText = "";
    cleanBox.innerText = "";
    try {
        const response = await fetch(`${API_URL}/rag/benchmark/run`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ embedding_model: embeddingModel })
        });
        const report = await response.json();

        // ВАЖНО: Убеждаемся, что мы не обращаемся к report.prepared
        summary.innerHTML = `
            Модель: <b>${report.embedding_model}</b><br>
            Индексировано примеров: ${report.indexed ? report.indexed.total_count : 'н/д'}<br>
            <b>Clean accuracy: ${(report.clean_tests.accuracy * 100).toFixed(1)}% (${report.clean_tests.correct}/${report.clean_tests.total})</b>
        `;

        if (rawBox) rawBox.innerHTML = "<i>Поиск по 'raw' данным (OCR XML) больше не поддерживается. Система перешла на Unified Clean RAG.</i>";
        renderBenchmarkTable("benchmark-clean", report.clean_tests);
        updateStatus();
    } catch (error) {
        summary.innerText = `Ошибка бенчмарка: ${error.message}`;
    } finally {
        btn.disabled = false;
        btn.innerText = "Запустить бенчмарк";
        if (cancelBtn) cancelBtn.classList.add("hidden");
    }
}

// Initial load
updateStatus();
updateModels();
updateBenchmarkModels();
updateBenchmarkHistory();
