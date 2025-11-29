import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { DataService, SettingsUpdatePayload } from '../../services/data';
import { SettingsState } from '../../models/models';

@Component({
  selector: 'app-settings',
  imports: [CommonModule, FormsModule],
  templateUrl: './settings.html',
  styleUrl: './settings.css'
})
export class Settings implements OnInit {
  darkMode = false;
  notificationsEnabled = true;
  models: string[] = [];
  selectedModel = '';
  account = {
    username: '',
    email: ''
  };

  constructor(private ds: DataService) {}

  ngOnInit(): void {
    this.loadSettings();
  }

  private loadSettings() {
    // Load dark mode from localStorage first
    const savedDarkMode = localStorage.getItem('darkMode');
    if (savedDarkMode !== null) {
      this.darkMode = savedDarkMode === 'true';
      document.documentElement.classList.toggle('dark', this.darkMode);
    }

    this.ds.getSettingsState().subscribe((state: SettingsState) => {
      this.darkMode = state.darkMode;
      this.notificationsEnabled = state.notificationsEnabled;
      this.models = state.models ?? [];
      this.selectedModel = state.selectedModel;
      this.account = { ...state.account };
      document.documentElement.classList.toggle('dark', this.darkMode);
      localStorage.setItem('darkMode', this.darkMode.toString());
    });
  }

  toggleTheme() {
    this.darkMode = !this.darkMode;
    document.documentElement.classList.toggle('dark', this.darkMode);
    localStorage.setItem('darkMode', this.darkMode.toString());
  }

  saveSettings() {
    const payload: SettingsUpdatePayload = {
      darkMode: this.darkMode,
      notificationsEnabled: this.notificationsEnabled,
      selectedModel: this.selectedModel,
      account: { ...this.account }
    };

    this.ds.saveSettings(payload).subscribe(updated => {
      this.darkMode = updated.darkMode;
      this.notificationsEnabled = updated.notificationsEnabled;
      this.models = updated.models ?? this.models;
      this.selectedModel = updated.selectedModel;
      this.account = { ...updated.account };
      document.documentElement.classList.toggle('dark', this.darkMode);
      localStorage.setItem('darkMode', this.darkMode.toString());
      alert('Settings saved successfully!');
    });
  }
}
