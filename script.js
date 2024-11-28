const availableDatasets = [];
let selectedDatasets = [];
let chart;

// Populate dataset selection options dynamically
function populateDatasetOptions(datasets) {
  const datasetOptions = document.getElementById('datasetOptions');
  datasetOptions.innerHTML = '';
  datasets.forEach(dataset => {
    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.value = dataset;
    checkbox.onchange = () => handleDatasetSelection(dataset, checkbox.checked);

    const label = document.createElement('label');
    label.innerText = dataset;
    label.style.marginRight = '10px';

    datasetOptions.appendChild(checkbox);
    datasetOptions.appendChild(label);
  });
}

function handleDatasetSelection(dataset, isSelected) {
  if (isSelected) {
    selectedDatasets.push(dataset);
  } else {
    selectedDatasets = selectedDatasets.filter(item => item !== dataset);
  }
}

function selectAllDatasets() {
  selectedDatasets = [...availableDatasets];
  const checkboxes = document.getElementById('datasetOptions').querySelectorAll('input');
  checkboxes.forEach(checkbox => {
    checkbox.checked = true;
  });
}

function uploadFiles() {
  const formData = new FormData(document.getElementById('fileForm'));
  fetch('http://localhost:5000/upload', {
    method: 'POST',
    body: formData
  })
  .then(response => response.json())
  .then(data => {
    availableDatasets.push(...data.datasets);
    populateDatasetOptions(availableDatasets);
    alert('Files uploaded and datasets loaded.');
  })
  .catch(error => console.error('Error uploading files:', error));
}

function startPrediction() {
  const timeStep = document.getElementById('timeStep').value;
  const predictionWeeks = document.getElementById('predictionWeeks').value;

  fetch('http://localhost:5000/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      datasets: selectedDatasets,
      timeStep: parseInt(timeStep),
      predictionWeeks: parseInt(predictionWeeks)
    })
  })
  .then(response => response.json())
  .then(data => {
    updateChart(data.labels, data.predictedPrices);
  })
  .catch(error => console.error('Error during prediction:', error));
}

function updateChart(labels, data) {
  if (chart) {
    chart.destroy();
  }

  const ctx = document.getElementById('predictionChart').getContext('2d');
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [{
        label: 'Future Predictions',
        data: data,
        borderColor: '#007bff',
        fill: false,
      }]
    },
    options: {
      scales: {
        x: {
          display: true,
          title: { display: true, text: 'Weeks' }
        },
        y: {
          display: true,
          title: { display: true, text: 'Price' }
        }
      }
    }
  });
}
