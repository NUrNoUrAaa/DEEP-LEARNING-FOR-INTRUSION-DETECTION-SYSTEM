import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgApexchartsModule } from 'ng-apexcharts';
import { Router } from '@angular/router';
import { Overview, TimeSeries, AttackTypes, AlertItem, ModelInfo, AnalyticsResponse } from '../../models/models';
import { DataService } from '../../services/data';
import { Subscription } from 'rxjs';


// export type ChartOptions = {
//   series: ApexAxisChartSeries;
//   chart: ApexChart;
//   xaxis: ApexXAxis;
//   dataLabels: ApexDataLabels;
//   stroke: ApexStroke;
//   tooltip: ApexTooltip;
//   legend: ApexLegend;
//   grid: ApexGrid;
// };


export type ChartOptions = any;


@Component({
  selector: 'app-dashboard',
  imports: [CommonModule, NgApexchartsModule],
  templateUrl: './dashboard.html',
  styleUrl: './dashboard.css'
})
export class Dashboard implements OnInit, OnDestroy {
  overview: Overview | null = null;
  timeseries: TimeSeries | null = null;
  attacks: AttackTypes | null = null;
  recentAlerts: AlertItem[] = [];
  displayedAlerts: AlertItem[] = [];
  preferredModel: ModelInfo | null = null;
  showAllAlerts: boolean = false;

  // chart options
  trafficChartOptions: ChartOptions = {};
  donutOptions: ChartOptions = {};

  // Notebook Insights Data
  modelComparison = [
    { name: 'Logistic Regression', accuracy: 92.0, precision: 83.0, recall: 93.0, f1: 87.0, latency: 12 },
    { name: 'Random Forest', accuracy: 99.9, precision: 99.7, recall: 99.9, f1: 99.8, latency: 85 },
    { name: 'XGBoost', accuracy: 99.4, precision: 99.0, recall: 99.5, f1: 99.2, latency: 45 },
    { name: 'MLP Model 3', accuracy: 98.7, precision: 98.2, recall: 98.7, f1: 98.4, latency: 53 }
  ];

  attackTypes = [
    { name: 'DoS', percentage: 25 },
    { name: 'DDoS', percentage: 20 },
    { name: 'Port Scan', percentage: 18 },
    { name: 'Brute Force', percentage: 15 },
    { name: 'Web Attacks', percentage: 12 },
    { name: 'Infiltration', percentage: 5 },
    { name: 'Botnet', percentage: 3 },
    { name: 'Other', percentage: 2 }
  ];

  topFeatures = [
    'Flow Duration',
    'Total Fwd Packets',
    'Total Backward Packets',
    'Flow IAT Mean',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Average Packet Size',
    'Subflow Fwd Packets'
  ];

  // Dataset Statistics
  datasetStats = {
    totalRecords: 2800000,
    benignPercentage: 85,
    attackPercentage: 15,
    features: 41,
    attackTypes: 9,
    trainTestSplit: '80/20'
  };

  // MLP Model Architectures
  mlpArchitectures = [
    { name: 'Model 1', layers: '8→4→2→1', params: 450, accuracy: 98.2 },
    { name: 'Model 2', layers: '16→8→4→2→1', params: 850, accuracy: 98.5 },
    { name: 'Model 3', layers: '32→16→8→4→2→1', params: 1600, accuracy: 98.7 },
    { name: 'Model 4', layers: '128→32→16→8→4→2→1', params: 4500, accuracy: 98.4 },
    { name: 'Model 5', layers: '256→128→64→32→16→8→4→2→1', params: 9200, accuracy: 98.1 }
  ];

  // Preprocessing Steps
  preprocessingSteps = [
    { step: 1, name: 'Data Loading', desc: 'Load 8 parquet files from CICIDS2017' },
    { step: 2, name: 'Cleaning', desc: 'Remove infinity, missing data, duplicates' },
    { step: 3, name: 'Feature Selection', desc: 'XGBoost reduces 41 to 33 features' },
    { step: 4, name: 'Scaling', desc: 'MinMaxScaler normalization [0,1]' },
    { step: 5, name: 'Balancing', desc: 'SMOTE oversampling for class balance' }
  ];

  // Key Metrics
  keyMetrics = [
    { label: 'Best Model Accuracy', value: '99.9%', icon: '🎯' },
    { label: 'Average Latency', value: '53ms', icon: '⚡' },
    { label: 'Feature Importance', value: '41→33', icon: '🔧' },
    { label: 'Class Balance', value: 'SMOTE', icon: '⚖️' }
  ];

  // Attack Types Breakdown
  attackBreakdown = [
    { type: 'DoS', count: 700000, percentage: 25 },
    { type: 'DDoS', count: 560000, percentage: 20 },
    { type: 'Port Scan', count: 504000, percentage: 18 },
    { type: 'Brute Force', count: 420000, percentage: 15 },
    { type: 'Web Attacks', count: 336000, percentage: 12 },
    { type: 'Infiltration', count: 140000, percentage: 5 },
    { type: 'Botnet', count: 84000, percentage: 3 },
    { type: 'Other', count: 56000, percentage: 2 }
  ];

  private analyticsSub?: Subscription;
  private themeObserver?: MutationObserver;
  private overviewSub?: Subscription;
  private timeSeriesSub?: Subscription;
  private attackTypesSub?: Subscription;
  private alertsSub?: Subscription;

  constructor(private ds: DataService, private router: Router) {}

  ngOnInit() {
    // Load last selected model from localStorage
    this.loadLastSelectedModel();

    // Load analytics and watch for model changes
    this.analyticsSub = this.ds.watchAnalytics().subscribe((data: AnalyticsResponse) => {
      this.preferredModel = data.preferredModel ?? null;
      if (this.preferredModel) {
        this.saveLastSelectedModel(this.preferredModel.name);
      }
      this.loadDashboardData();
    });

    this.ds.getAnalytics().subscribe();

    // Watch for theme changes
    this.themeObserver = new MutationObserver(() => {
      this.updateChartsTheme();
    });

    this.themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });

    // Initial load
    this.loadDashboardData();
  }

  ngOnDestroy() {
    this.analyticsSub?.unsubscribe();
    this.themeObserver?.disconnect();
    this.overviewSub?.unsubscribe();
    this.timeSeriesSub?.unsubscribe();
    this.attackTypesSub?.unsubscribe();
    this.alertsSub?.unsubscribe();
  }

  private loadDashboardData() {
    // Unsubscribe from previous subscriptions
    this.overviewSub?.unsubscribe();
    this.timeSeriesSub?.unsubscribe();
    this.attackTypesSub?.unsubscribe();
    this.alertsSub?.unsubscribe();

    // Load overview
    this.overviewSub = this.ds.getOverview().subscribe(o => {
      this.overview = o;
    });

    // Load time series
    this.timeSeriesSub = this.ds.getTimeSeries().subscribe(ts => {
      this.timeseries = ts;
      this.updateTrafficChart();
    });

    // Load attack types
    this.attackTypesSub = this.ds.getAttackTypes().subscribe(a => {
      this.attacks = a;
      this.updateDonutChart();
    });

    // Load alerts
    this.alertsSub = this.ds.getAlerts().subscribe(arr => {
      this.recentAlerts = arr;
      this.updateDisplayedAlerts();
    });
  }

  updateDisplayedAlerts() {
    this.displayedAlerts = this.showAllAlerts ? this.recentAlerts : this.recentAlerts.slice(0, 5);
  }

  toggleShowAllAlerts() {
    // Navigate to Alerts page
    this.router.navigate(['/alerts']);
  }

  private updateTrafficChart() {
    const isDark = document.documentElement.classList.contains('dark');
    if (!this.timeseries) return;

    this.trafficChartOptions = {
      series: [{ name: 'Traffic', data: this.timeseries.series }],
      chart: { type: 'line', height: 320, toolbar: { show: false } },
      xaxis: { categories: this.timeseries.categories },
      stroke: { curve: 'smooth', width: 3 },
      tooltip: { theme: isDark ? 'dark' : 'light' }
    };
  }

  private updateDonutChart() {
    if (!this.attacks) return;

    this.donutOptions = {
      series: this.attacks.values,
      chart: { type: 'donut', height: 260 },
      labels: this.attacks.labels,
      legend: { position: 'bottom' }
    };
  }

  private updateChartsTheme() {
    this.updateTrafficChart();
    this.updateDonutChart();
  }

  private saveLastSelectedModel(modelName: string) {
    localStorage.setItem('lastSelectedModel', modelName);
    localStorage.setItem('modelSelectionTime', Date.now().toString());
  }

  private loadLastSelectedModel() {
    const lastModel = localStorage.getItem('lastSelectedModel');
    if (lastModel && lastModel !== 'undefined' && lastModel !== '') {
      // Delay to ensure service is ready
      setTimeout(() => {
        this.ds.saveSettings({ selectedModel: lastModel }).subscribe(
          () => {
            console.log('Model restored from localStorage:', lastModel);
          },
          (error) => {
            console.error('Failed to restore model:', error);
          }
        );
      }, 500);
    }
  }

}
