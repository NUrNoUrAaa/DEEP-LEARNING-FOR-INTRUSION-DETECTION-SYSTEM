import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgApexchartsModule } from 'ng-apexcharts';
import { AlertItem, AnalyticsResponse, Overview, ModelInfo } from '../../models/models';
import { DataService } from '../../services/data';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-analytics',
  standalone: true,
  imports: [CommonModule, NgApexchartsModule],
  templateUrl: './analytics.html',
  styleUrls: ['./analytics.css']
})
export class Analytics implements OnInit, OnDestroy {
  Math = Math;
  overview: Overview = {
    totalTraffic: '0',
    detected: 0,
    accuracy: 0,
    activeAlerts: 0,
    benignCount: 0
  };
  analytics: AnalyticsResponse | null = null;
  preferredModel: ModelInfo | null = null;

  recentAlerts: AlertItem[] = [];
  private themeObserver?: MutationObserver;
  private analyticsSub?: Subscription;

  pageSize = 8;
  currentPage = 1;
  get totalPages() {
    return Math.max(1, Math.ceil(this.recentAlerts.length / this.pageSize));
  }
  get pagedAlerts() {
    const start = (this.currentPage - 1) * this.pageSize;
    return this.recentAlerts.slice(start, start + this.pageSize);
  }
  goPage(n: number) {
    if (n < 1) n = 1;
    if (n > this.totalPages) n = this.totalPages;
    this.currentPage = n;
  }

  trafficChartOptions: any = {};
  barChartOptions: any = {};
  areaChartOptions: any = {};
  donutOptions: any = {};
  radarOptions: any = {};
  heatmapOptions: any = {};
  featureImportanceOptions: any = {};
  modelComparisonOptions: any = {};

  // Model Comparison Data
  modelComparison = [
    { name: 'Logistic Regression', accuracy: 92.0, precision: 83.0, recall: 93.0, f1: 87.0, latency: 12 },
    { name: 'Random Forest', accuracy: 99.9, precision: 99.7, recall: 99.9, f1: 99.8, latency: 85 },
    { name: 'XGBoost', accuracy: 99.4, precision: 99.0, recall: 99.5, f1: 99.2, latency: 45 },
    { name: 'MLP Model 3', accuracy: 98.7, precision: 98.2, recall: 98.7, f1: 98.4, latency: 53 }
  ];

  // Feature Importance Data
  topFeatures = [
    { name: 'Flow Duration', importance: 95 },
    { name: 'Total Fwd Packets', importance: 88 },
    { name: 'Total Backward Packets', importance: 85 },
    { name: 'Flow IAT Mean', importance: 82 },
    { name: 'Fwd Packet Length Mean', importance: 78 },
    { name: 'Bwd Packet Length Mean', importance: 75 },
    { name: 'Flow Bytes/s', importance: 72 },
    { name: 'Flow Packets/s', importance: 68 }
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

  // Performance Metrics
  performanceMetrics = [
    { label: 'Best Model Accuracy', value: '99.9%', icon: '🎯', color: 'success' },
    { label: 'Average Latency', value: '53ms', icon: '⚡', color: 'info' },
    { label: 'Total Records', value: '2.8M', icon: '📊', color: 'primary' },
    { label: 'Detection Rate', value: '99.7%', icon: '🛡️', color: 'success' }
  ];

  constructor(private ds: DataService) {}

  ngOnInit() {
    this.analyticsSub = this.ds.watchAnalytics().subscribe((data: AnalyticsResponse) => {
      this.analytics = data;
      this.overview = data.overview;
      this.recentAlerts = data.recentAlerts ?? [];
      this.preferredModel = data.preferredModel ?? null;
      this.currentPage = 1;
      this.configureCharts();
    });
    this.ds.getAnalytics().subscribe();

    this.themeObserver = new MutationObserver(() => {
      this.configureCharts();
    });

    this.themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['class']
    });
  }

  ngOnDestroy() {
    this.themeObserver?.disconnect();
    this.analyticsSub?.unsubscribe();
  }

  private configureCharts() {
    const analytics = this.analytics;
    const isDark = document.documentElement.classList.contains('dark');
    const textColor = isDark ? '#FFFFFFFF' : '#000000FF';
    const gridColor = isDark ? '#334155' : '#e6e6e6';

    this.trafficChartOptions = {
      series: analytics?.trafficTrend.series ?? [],
      chart: { type: 'line', height: 300, animations: { enabled: true, easing: 'easeinout', speed: 700 }, foreColor: textColor },
      stroke: { curve: 'smooth', width: 3 },
      xaxis: { categories: analytics?.trafficTrend.categories ?? [], labels: { style: { colors: textColor } } },
      colors: isDark ? ['#7dd3fc', '#f87171'] : ['#2563eb', '#dc2626'],
      grid: { borderColor: gridColor },
      tooltip: { theme: isDark ? 'dark' : 'light' },
      dataLabels: { enabled: true, style: { colors: [textColor] } },
      theme: { mode: isDark ? 'dark' : 'light' }
    };

    this.barChartOptions = {
      series: [{ name: 'Attacks', data: analytics?.topAttackLabels.values ?? [] }],
      chart: { type: 'bar', height: 260, animations: { enabled: true, speed: 700 }, foreColor: textColor },
      xaxis: { categories: analytics?.topAttackLabels.labels ?? [], labels: { style: { colors: textColor } } },
      plotOptions: { bar: { borderRadius: 8, columnWidth: '55%' } },
      colors: [isDark ? '#fb7185' : '#ef4444'],
      grid: { borderColor: gridColor },
      tooltip: { theme: isDark ? 'dark' : 'light' },
      dataLabels: { enabled: true, style: { colors: [textColor] } },
      theme: { mode: isDark ? 'dark' : 'light' }
    };

    this.areaChartOptions = {
      series: analytics?.detectionTrend.series ?? [],
      chart: { type: 'area', height: 260, animations: { enabled: true, speed: 700 }, foreColor: textColor },
      xaxis: { categories: analytics?.detectionTrend.categories ?? [], labels: { style: { colors: textColor } } },
      stroke: { curve: 'smooth' },
      fill: { type: 'gradient', gradient: { shade: isDark ? 'dark' : 'light', opacityFrom: 0.6, opacityTo: 0.1 } },
      colors: [isDark ? '#06b6d4' : '#2563eb'],
      grid: { borderColor: gridColor },
      tooltip: { theme: isDark ? 'dark' : 'light' },
      dataLabels: { enabled: true, style: { colors: [textColor] } },
      theme: { mode: isDark ? 'dark' : 'light' }
    };

    this.donutOptions = {
      series: analytics?.topAttackLabels.values ?? [],
      chart: { type: 'donut', height: 260, foreColor: textColor },
      labels: analytics?.topAttackLabels.labels ?? [],
      legend: { position: 'bottom', labels: { colors: textColor } },
      colors: isDark ? ['#34d399', '#fb7185', '#fbbf24', '#60a5fa', '#c084fc'] : ['#10b981', '#ef4444', '#f59e0b', '#3b82f6', '#a855f7'],
      tooltip: { theme: isDark ? 'dark' : 'light' },
      dataLabels: { enabled: true, style: { colors: [textColor] } },
      theme: { mode: isDark ? 'dark' : 'light' }
    };

    this.radarOptions = {
      series: analytics?.modelRadar.series ?? [],
      chart: { type: 'radar', height: 300, foreColor: textColor, animations: { enabled: true } },
      xaxis: { categories: analytics?.modelRadar.labels ?? [], labels: { style: { colors: textColor } } },
      colors: isDark ? ['#a78bfa', '#60a5fa', '#f472b6', '#34d399'] : ['#7c3aed', '#2563eb', '#ec4899', '#10b981'],
      tooltip: { theme: isDark ? 'dark' : 'light' },
      dataLabels: { enabled: true, style: { colors: [textColor] } },
      theme: { mode: isDark ? 'dark' : 'light' }
    };

    const heatmap = analytics?.heatmap;
    const heatmapSeries = (heatmap?.days ?? []).map((day, idx) => {
      const row = heatmap?.matrix[idx] ?? [];
      return {
        name: day,
        data: row.map((value, hourIdx) => ({
          x: heatmap?.hours?.[hourIdx] ?? hourIdx.toString(),
          y: value
        }))
      };
    });

    this.heatmapOptions = {
      series: heatmapSeries,
      chart: { type: 'heatmap', height: 200, foreColor: textColor },
      dataLabels: { enabled: true, style: { colors: ['#000000FF'] } },
      colors: ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899'],
      plotOptions: { heatmap: { radius: 4 } },
      tooltip: { theme: isDark ? 'dark' : 'light' },
      theme: { mode: isDark ? 'dark' : 'light' },
      grid: { borderColor: gridColor }
    };

    // Feature Importance Chart
    this.featureImportanceOptions = {
      series: [{ name: 'Importance Score', data: this.topFeatures.map(f => f.importance) }],
      chart: { type: 'bar', height: 300, foreColor: textColor, animations: { enabled: true } },
      xaxis: { categories: this.topFeatures.map(f => f.name), labels: { style: { colors: textColor } } },
      plotOptions: { bar: { borderRadius: 8, columnWidth: '70%', horizontal: true } },
      colors: [isDark ? '#60a5fa' : '#3b82f6'],
      grid: { borderColor: gridColor },
      tooltip: { theme: isDark ? 'dark' : 'light' },
      dataLabels: { enabled: true, style: { colors: [textColor] } },
      theme: { mode: isDark ? 'dark' : 'light' }
    };

    // Model Comparison Chart (Accuracy)
    this.modelComparisonOptions = {
      series: [
        { name: 'Accuracy', data: this.modelComparison.map(m => m.accuracy) },
        { name: 'Precision', data: this.modelComparison.map(m => m.precision) },
        { name: 'Recall', data: this.modelComparison.map(m => m.recall) }
      ],
      chart: { type: 'bar', height: 300, foreColor: textColor, animations: { enabled: true } },
      xaxis: { categories: this.modelComparison.map(m => m.name), labels: { style: { colors: textColor } } },
      plotOptions: { bar: { borderRadius: 8, columnWidth: '60%' } },
      colors: isDark ? ['#34d399', '#60a5fa', '#f472b6'] : ['#10b981', '#3b82f6', '#ec4899'],
      grid: { borderColor: gridColor },
      tooltip: { theme: isDark ? 'dark' : 'light' },
      dataLabels: { enabled: true, style: { colors: [textColor] } },
      theme: { mode: isDark ? 'dark' : 'light' }
    };
  }

  progressPct(value: number) {
    return Math.max(0, Math.min(100, value)) + '%';
  }

  asNumber(value: string | number | undefined | null): number {
    if (typeof value === 'number') {
      return value;
    }
    if (typeof value === 'string') {
      const parsed = Number(value);
      return Number.isNaN(parsed) ? 0 : parsed;
    }
    return 0;
  }

  severityColor(sev: string) {
    switch (sev) {
      case 'High': return 'text-red-500 font-bold';
      case 'Medium': return 'text-yellow-500 font-semibold';
      case 'Low': return 'text-green-500 font-medium';
      default: return '';
    }
  }
}
