import { Routes } from '@angular/router';
import { Dashboard } from './pages/dashboard/dashboard';
import { Analytics } from './pages/analytics/analytics';
import { Alerts } from './pages/alerts/alerts';
import { Models } from './pages/models/models';
import { Settings } from './pages/settings/settings';
import { Layout } from './layout/layout/layout';
import { ModelDetailsComponent } from './pages/model-details/model-details';

export const routes: Routes = [
  {
   path: '',
    component: Layout,
    children: [
  { path: '', redirectTo: 'dashboard', pathMatch: 'full' },
  { path: 'dashboard', component: Dashboard },
  { path: 'analytics', component: Analytics },
  { path: 'alerts', component: Alerts },
  { path: 'models', component: Models },
  { path: 'settings', component: Settings },
  { path: 'model-details', component: ModelDetailsComponent }
]
}
];
