import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgChartsModule } from 'ng2-charts';
import { ChartData, ChartOptions } from 'chart.js';

@Component({
  selector: 'app-model-details',
  standalone: true,
  imports: [CommonModule, NgChartsModule],
  templateUrl: './model-details.html',
  styleUrls: ['./model-details.css']
})
export class ModelDetailsComponent {
  @Input() model: any;

  chartOptions: ChartOptions<'radar'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { display: true } },
    scales: { r: { beginAtZero: true, max: 100 } }
  };

  getChartData(): ChartData<'radar', number[], string> {
    if (!this.model) {
      return {
        labels: ['Accuracy', 'Precision', 'Recall'],
        datasets: [
          {
            label: 'No Data',
            data: [0, 0, 0],
            backgroundColor: ['#e5e7eb', '#e5e7eb', '#e5e7eb']
          }
        ]
      };
    }

    return {
      labels: ['Accuracy', 'Precision', 'Recall'],
      datasets: [
        {
          label: this.model.name,
          data: [
            this.model.accuracy || 0,
            this.model.precision || 0,
            this.model.recall || 0
          ],
          backgroundColor: ['#3b82f6', '#8b5cf6', '#10b981']
        }
      ]
    };
  }
}
