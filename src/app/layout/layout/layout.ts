import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterOutlet } from '@angular/router';
import { Navbar } from '../navbar/navbar';
import { Sidebar } from '../sidebar/sidebar';


@Component({
  selector: 'app-layout',
  imports: [CommonModule, RouterOutlet, Navbar, Sidebar],
  templateUrl: './layout.html',
  styleUrl: './layout.css'
})
export class Layout {

}
