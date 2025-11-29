import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { Sidebar } from '../sidebar/service/sidebar';

@Component({
  selector: 'app-navbar',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './navbar.html',
  styleUrls: ['./navbar.css']
})
export class Navbar implements OnInit {
  theme: 'light' | 'dark' = 'light';

  constructor(private sidebar: Sidebar) {}

  ngOnInit() {
    const stored = localStorage.getItem('theme') as 'light'|'dark'|null;
    this.theme = stored ?? 'light';
    document.documentElement.classList.toggle('dark', this.theme === 'dark');
  }

  toggleTheme() {
    this.theme = this.theme === 'light' ? 'dark' : 'light';
    document.documentElement.classList.toggle('dark', this.theme === 'dark');
    localStorage.setItem('theme', this.theme);
  }

  toggleSidebar() {
    this.sidebar.toggle();
  }
}
