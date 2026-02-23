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
    try {
        const response = await fetch(`${API_URL}/rag/models`);
        const data = await response.json();
        const selector = document.getElementById("model-selector");
        if (selector) {
            selector.innerHTML = "";
            data.models.forEach(model => {
                const opt = document.createElement("option");
                opt.value = model;
                opt.innerText = model;
                selector.appendChild(opt);
            });
        }
    } catch (error) {
        console.error("Error fetching models:", error);
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
        resultDiv.innerHTML = "";
        
        if (data.processed_files.length === 0) {
            resultDiv.innerText = "Новых изображений не найдено.";
        } else {
            data.processed_files.forEach(file => {
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
        resultDiv.innerHTML = "";
        
        if (data.cleaned_files.length === 0) {
            resultDiv.innerText = "Новых XML для очистки не найдено.";
        } else {
            data.cleaned_files.forEach(file => {
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
    const resultDiv = document.getElementById("structure-result");
    
    if (!filename) {
        alert("Введите имя файла!");
        return;
    }
    
    resultDiv.innerText = `Поиск примеров и структурирование с помощью ${model}...`;

    try {
        const response = await fetch(`${API_URL}/rag/structure`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: filename, model_name: model })
        });
        const data = await response.json();
        
        // Превращаем JSON в красиво отформатированный текст
        resultDiv.innerHTML = `<pre class="json-code">${JSON.stringify(data, null, 2)}</pre>`;
    } catch (error) {
        resultDiv.innerText = "Ошибка структурирования: " + error.message;
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
            body: JSON.stringify({ query: query, limit: 5 })
        });
        const results = await response.json();

        resultsDiv.innerHTML = "";
        if (results.length === 0) {
            resultsDiv.innerText = "Ничего не найдено.";
            return;
        }

        results.forEach(res => {
            const scorePercent = (res.score * 100).toFixed(1) + "%";
            resultsDiv.appendChild(renderResultItem(res.filename, res.text, scorePercent));
        });
    } catch (error) {
        resultsDiv.innerText = "Ошибка поиска: " + error.message;
    }
}

// Initial load
updateStatus();
