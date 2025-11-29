import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, of, BehaviorSubject } from 'rxjs';
import { catchError, map, filter, tap } from 'rxjs/operators';
import {
  Overview,
  TimeSeries,
  AttackTypes,
  AlertItem,
  AnalyticsResponse,
  ModelInfo,
  SettingsState,
} from '../models/models';

// ============================================================
// ==================== INTERFACES ==========================
// ============================================================

export interface PredictionResult {
  prediction: 'Attack' | 'Benign';
  confidence: number;
  risk_level: 'Critical' | 'High' | 'Medium' | 'Low';
}

export interface PredictionResponse {
  success: boolean;
  records_processed: number;
  attack_count: number;
  benign_count: number;
  results: PredictionResult[];
  error?: string;
}

export interface FileUploadResponse {
  success: boolean;
  records_processed: number;
  attack_count: number;
  benign_count: number;
  results: any[];
  error?: string;
}

export interface ModelStats {
  model_loaded: boolean;
  features: string[];
  features_count: number;
  feature_ranges: { [key: string]: { min: number; max: number } };
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  features_count: number;
  message?: string;
}

export interface SettingsUpdatePayload {
  darkMode?: boolean;
  notificationsEnabled?: boolean;
  selectedModel?: string;
  account?: {
    username?: string;
    email?: string;
  };
}

// ============================================================
// ==================== SERVICE =============================
// ============================================================

@Injectable({
  providedIn: 'root'
})
export class DataService {
  private apiUrl = 'http://localhost:5000/api';
  private analyticsState$ = new BehaviorSubject<AnalyticsResponse | null>(null);

  constructor(private http: HttpClient) {
    this.checkApiHealth();
    this.getAnalytics().subscribe();
  }

  // ============================================================
  // ==================== HEALTH & STATUS ======================
  // ============================================================

  checkApiHealth(): void {
    this.http.get<HealthResponse>(`${this.apiUrl}/health`).pipe(
      catchError(error => {
        console.warn('API health check failed:', error);
        return of({ status: 'error', model_loaded: false, features_count: 0 } as HealthResponse);
      })
    ).subscribe(response => {
      if (response.status === 'error') {
        console.error('API not available');
      } else {
        console.log('✓ API connection established');
      }
    });
  }

  // ============================================================
  // ==================== PREDICTIONS ==========================
  // ============================================================

  /**
   * Make predictions on a single record or multiple records
   * @param data Single record or array of records
   * @returns Observable with prediction results
   */
  predictData(data: any | any[]): Observable<PredictionResponse> {
    return this.http.post<PredictionResponse>(
      `${this.apiUrl}/predict`,
      { data }
    ).pipe(
      catchError(error => {
        console.error('Prediction error:', error);
        return of({
          success: false,
          records_processed: 0,
          attack_count: 0,
          benign_count: 0,
          results: [],
          error: error.error?.error || 'Prediction failed'
        } as PredictionResponse);
      })
    );
  }

  /**
   * Upload and predict on CSV file
   * @param file CSV file to upload
   * @returns Observable with file prediction results
   */
  predictFile(file: File): Observable<FileUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    return this.http.post<FileUploadResponse>(
      `${this.apiUrl}/predict/file`,
      formData
    ).pipe(
      catchError(error => {
        console.error('File upload error:', error);
        return of({
          success: false,
          records_processed: 0,
          attack_count: 0,
          benign_count: 0,
          results: [],
          error: error.error?.error || 'File upload failed'
        } as FileUploadResponse);
      })
    );
  }

  // ============================================================
  // ==================== SYNTHETIC DATA =======================
  // ============================================================

  /**
   * Generate synthetic benign-like data
   * @param rows Number of rows to generate
   * @returns Observable with generated data
   */
  generateBenignData(rows: number = 20): Observable<any> {
    return this.http.post<any>(
      `${this.apiUrl}/generate/benign`,
      { rows }
    ).pipe(
      catchError(error => {
        console.error('Generate benign data error:', error);
        return of({ success: false, data: [], error: 'Failed to generate data' });
      })
    );
  }

  /**
   * Generate synthetic attack-like data
   * @param rows Number of rows to generate
   * @returns Observable with generated data
   */
  generateAttackData(rows: number = 20): Observable<any> {
    return this.http.post<any>(
      `${this.apiUrl}/generate/attack`,
      { rows }
    ).pipe(
      catchError(error => {
        console.error('Generate attack data error:', error);
        return of({ success: false, data: [], error: 'Failed to generate data' });
      })
    );
  }

  // ============================================================
  // ==================== MODEL STATS ==========================
  // ============================================================

  /**
   * Get model and feature statistics
   * @returns Observable with model statistics
   */
  getModelStats(): Observable<ModelStats> {
    return this.http.get<ModelStats>(`${this.apiUrl}/stats`).pipe(
      catchError(error => {
        console.error('Stats error:', error);
        return of({
          model_loaded: false,
          features: [],
          features_count: 0,
          feature_ranges: {}
        });
      })
    );
  }

  // ============================================================
  // ==================== ANALYTICS & MODELS =====================
  // ============================================================

  getAnalytics(): Observable<AnalyticsResponse> {
    return this.http.get<AnalyticsResponse>(`${this.apiUrl}/analytics`).pipe(
      tap(data => this.analyticsState$.next(data)),
      catchError(error => {
        console.error('Analytics error:', error);
        const fallback = this.buildEmptyAnalytics();
        this.analyticsState$.next(fallback);
        return of(fallback);
      })
    );
  }

  watchAnalytics(): Observable<AnalyticsResponse> {
    return this.analyticsState$.pipe(
      filter((value): value is AnalyticsResponse => value !== null)
    );
  }

  getModelBenchmarks(): Observable<ModelInfo[]> {
    return this.http.get<ModelInfo[]>(`${this.apiUrl}/models`).pipe(
      catchError(error => {
        console.error('Models catalog error:', error);
        return of([] as ModelInfo[]);
      })
    );
  }

  // ============================================================
  // ==================== SETTINGS ==============================
  // ============================================================

  getSettingsState(): Observable<SettingsState> {
    return this.http.get<SettingsState>(`${this.apiUrl}/settings`).pipe(
      catchError(error => {
        console.error('Settings fetch error:', error);
        return of(this.buildDefaultSettingsState());
      })
    );
  }

  saveSettings(update: SettingsUpdatePayload): Observable<SettingsState> {
    return this.http.post<{ success: boolean; settings: SettingsState }>(
      `${this.apiUrl}/settings`,
      update
    ).pipe(
      map(response => response.settings),
      tap(() => this.getAnalytics().subscribe()),
      catchError(error => {
        console.error('Settings save error:', error);
        return of(this.buildDefaultSettingsState());
      })
    );
  }

  // ============================================================
  // ==================== DASHBOARD DATA =======================
  // ============================================================

  getOverview(): Observable<Overview> {
    return this.http.get<Overview>(`${this.apiUrl}/overview`).pipe(
      catchError(error => {
        console.error('Overview error:', error);
        return of({
          totalTraffic: '0',
          detected: 0,
          accuracy: 0,
          activeAlerts: 0,
          benignCount: 0
        } as Overview);
      })
    );
  }

  getTimeSeries(): Observable<TimeSeries> {
    return this.http.get<TimeSeries>(`${this.apiUrl}/time-series`).pipe(
      catchError(error => {
        console.error('Time series error:', error);
        return of({
          categories: [],
          series: []
        } as TimeSeries);
      })
    );
  }

  getAttackTypes(): Observable<AttackTypes> {
    return this.http.get<AttackTypes>(`${this.apiUrl}/attack-types`).pipe(
      catchError(error => {
        console.error('Attack types error:', error);
        return of({
          labels: [],
          values: []
        } as AttackTypes);
      })
    );
  }

  getAlerts(): Observable<AlertItem[]> {
    return this.http.get<AlertItem[]>(`${this.apiUrl}/alerts`).pipe(
      catchError(error => {
        console.error('Alerts error:', error);
        return of([] as AlertItem[]);
      })
    );
  }

  // ============================================================
  // ==================== UTILITY METHODS ======================
  // ============================================================

  setApiUrl(url: string): void {
    this.apiUrl = url;
  }

  private buildEmptyAnalytics(): AnalyticsResponse {
    return {
      overview: { totalTraffic: '0', detected: 0, accuracy: 0, activeAlerts: 0, benignCount: 0 },
      trafficTrend: { categories: [], series: [] },
      topAttackLabels: { labels: [], values: [] },
      detectionTrend: { categories: [], series: [] },
      heatmap: { days: [], hours: [], matrix: [] },
      modelRadar: { labels: [], series: [] },
      recentAlerts: [],
      severity: {},
      preferredModel: undefined
    };
  }

  private buildDefaultSettingsState(): SettingsState {
    return {
      darkMode: false,
      notificationsEnabled: true,
      selectedModel: 'Random Forest',
      models: [],
      account: {
        username: 'cyber_admin',
        email: 'admin@cyber-ids.com'
      }
    };
  }
}
