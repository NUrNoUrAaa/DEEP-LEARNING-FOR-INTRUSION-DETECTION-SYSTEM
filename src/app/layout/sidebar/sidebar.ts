import { Component, OnInit, OnDestroy, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule } from '@angular/router';
import { Subscription } from 'rxjs';
import { Sidebar as SidebarService } from './service/sidebar';


@Component({
  selector: 'app-sidebar',
  standalone: true,
  imports: [CommonModule, RouterModule],
  templateUrl: './sidebar.html',
  styleUrls: ['./sidebar.css']
})
export class Sidebar  implements OnInit, OnDestroy {
  isOpen = true;
  isMobile = false;
  sub!: Subscription;

  constructor(public sb:SidebarService ) {}

  ngOnInit() {
    this.checkScreen();
    this.sub = this.sb.open.subscribe(v => this.isOpen = v);
  }

  ngOnDestroy() { this.sub?.unsubscribe(); }

  @HostListener('window:resize', [])
  onResize() { this.checkScreen(); }

  checkScreen() {
    const w = (typeof window !== 'undefined') ? window.innerWidth : 1200;
    this.isMobile = w < 1024;
    if (this.isMobile) this.sb.closeSidebar();
    else this.sb.openSidebar();
  }
}
