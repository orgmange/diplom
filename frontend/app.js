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
            Vectorized docs (OCR): ${data.vectorized_count}
        `;
    } catch (error) {
        console.error("Error fetching status:", error);
        document.getElementById("status-display").innerText = "Ошибка соединения с API";
    }
}

async function updateModels() {
    const selectors = [
        document.getElementById("model-selector"),
        document.getElementById("search-model-selector")
    ].filter(s => !!s);
    try {
        const response = await fetch(`${API_URL}/rag/models`);
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        selectors.forEach(selector => {
            const currentVal = selector.value;
            selector.innerHTML = selector.id === "search-model-selector" ? '<option value="">(по умолчанию)</option>' : "";
            data.models.forEach(model => {
                const opt = document.createElement("option");
                opt.value = model;
                opt.innerText = model;
                if (model === currentVal) opt.selected = true;
                selector.appendChild(opt);
            });
        });
    } catch (error) {
        console.error("Error fetching models:", error);
        selectors.forEach(selector => {
            selector.innerHTML = `<option value="">Ошибка: ${error.message}</option>`;
        });
    }
}

// Псевдоним для совместимости
async function updateSearchModels() { await updateModels(); }

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

async function runStructure() {
    const filename = document.getElementById("filename-input").value;
    const model = document.getElementById("model-selector").value;
    const reindex = document.getElementById("reindex-checkbox").checked;
    const resultDiv = document.getElementById("structure-result");
    
    if (!filename) {
        alert("Введите имя файла!");
        return;
    }
    
    let msg = `Структурирование с помощью ${model}...`;
    if (reindex) {
        msg = `🔄 Очистка базы и полная переиндексация с ${model}... ` + msg;
    }
    resultDiv.innerText = msg;

    try {
        const response = await fetch(`${API_URL}/rag/structure`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ 
                filename: filename, 
                model_name: model,
                reindex: reindex 
            })
        });
        const data = await response.json();
        
        // Превращаем JSON в красиво отформатированный текст
        resultDiv.innerHTML = `<pre class="json-code">${JSON.stringify(data, null, 2)}</pre>`;
    } catch (error) {
        resultDiv.innerText = "Ошибка структурирования: " + error.message;
    } finally {
        updateStatus();
    }
}

async function runSearch() {
    const query = document.getElementById("search-query").value;
    const model = document.getElementById("search-model-selector")?.value;
    if (!query) return;

    const resultsDiv = document.getElementById("search-results");
    resultsDiv.innerHTML = `Поиск по смыслу (модель: ${model || "default"})...`;

    try {
        const response = await fetch(`${API_URL}/rag/search`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, limit: 5, model: model || undefined })
        });
        const results = await response.json();

        resultsDiv.innerHTML = "";
        if (results.length === 0) {
            resultsDiv.innerText = "Ничего не найдено.";
            return;
        }

        results.forEach(res => {
            const scorePercent = (res.score * 100).toFixed(1) + "%";
            // Проверяем все возможные поля текста для совместимости со старыми и новыми индексами
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
    const rows = group.items.map(item => `
        <tr class="${item.is_correct ? "ok-row" : "bad-row"}">
            <td>${item.filename}</td>
            <td>${item.expected_type || "-"}</td>
            <td>${item.predicted_type || "-"}</td>
            <td>${item.predicted_filename || "-"}</td>
            <td>${item.score !== null && item.score !== undefined ? Number(item.score).toFixed(4) : "-"}</td>
            <td>${item.is_correct ? "OK" : "ERR"}</td>
        </tr>
    `).join("");
    el.innerHTML = `
        <table class="benchmark-table">
            <thead>
                <tr>
                    <th>Файл</th>
                    <th>Ожидали</th>
                    <th>Предсказали</th>
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
    try {
        const response = await fetch(`${API_URL}/rag/benchmark/models`);
        const data = await response.json();
        selector.innerHTML = "";
        data.models.forEach(model => {
            const opt = document.createElement("option");
            opt.value = model;
            opt.innerText = model;
            selector.appendChild(opt);
        });
    } catch (error) {
        selector.innerHTML = `<option value="">Ошибка загрузки: ${error.message}</option>`;
    }
}

async function runEmbeddingBenchmark() {
    const btn = document.getElementById("run-benchmark-btn");
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
        summary.innerHTML = `
            Модель: <b>${report.embedding_model}</b><br>
            Подготовлено тестов: xml=${report.prepared.prepared_xml}, clean=${report.prepared.prepared_clean}<br>
            Индексировано: raw=${report.indexed.raw_count}, clean=${report.indexed.clean_count}, total=${report.indexed.total_count}<br>
            Raw accuracy: ${(report.raw_tests.accuracy * 100).toFixed(1)}% (${report.raw_tests.correct}/${report.raw_tests.total})<br>
            Clean accuracy: ${(report.clean_tests.accuracy * 100).toFixed(1)}% (${report.clean_tests.correct}/${report.clean_tests.total})<br>
            Общий результат: ${(report.overall.accuracy * 100).toFixed(1)}% (${report.overall.correct}/${report.overall.total})
        `;
        renderBenchmarkTable("benchmark-raw", report.raw_tests);
        renderBenchmarkTable("benchmark-clean", report.clean_tests);
        updateStatus();
    } catch (error) {
        summary.innerText = `Ошибка бенчмарка: ${error.message}`;
    } finally {
        btn.disabled = false;
        btn.innerText = "Запустить бенчмарк";
    }
}

updateStatus();
updateModels();
updateBenchmarkModels();
