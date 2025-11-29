import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class Sidebar {
  private open$ = new BehaviorSubject<boolean>(true);
  open = this.open$.asObservable();

  toggle() { this.open$.next(!this.open$.value); }
  openSidebar() { this.open$.next(true); }
  closeSidebar() { this.open$.next(false); }
  get value() { return this.open$.value; }
}
