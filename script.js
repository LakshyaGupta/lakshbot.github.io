// Placeholder data for datasets
const availableDatasets = ["SP500", "FEDFUNDS", "DJIA", "NASDAQCOM", "NASDAQ100", "GDPC1", "UNRATE", "DCOILWTICO", "DGS10", "M2V", "WM2NS"];
let selectedDatasets = [];
let chart;

// Populate dataset selection options
const datasetOptions = document.getElementById('datasetOptions');
availableDatasets.forEach(dataset => {
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

function handleDatasetSelection(dataset, isSelected) {
  if (isSelected) {
    selectedDatasets.push(dataset);
  } else {
    selectedDatasets = selectedDatasets.filter(item => item !== dataset);
  }
}

function selectAllDatasets() {
  selectedDatasets = [...availableDatasets];
  datasetOptions.querySelectorAll('input').forEach(checkbox => {
    checkbox.checked = true;
  });
}

function startPrediction() {
  const timeStep = document.getElementById('timeStep').value;
  const predictionWeeks = document.getElementById('predictionWeeks').value;

  // Placeholder data for chart
  const labels = Array.from({ length: predictionWeeks }, (_, i) => `Week ${i + 1}`);
  const data = Array.from({ length: predictionWeeks }, () => Math.random() * 100);

  updateChart(labels, data);
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
          title: {
            display: true,
            text: 'Weeks'
          }
        },
        y: {
          display: true,
          title: {
            display: true,
            text: 'Price'
          }
        }
      }
    }
  });
}
