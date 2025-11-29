import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { Layout } from './layout/layout/layout';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-root',
  imports: [ Layout , FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App {
  protected readonly title = signal('ids-dashboard');
}
