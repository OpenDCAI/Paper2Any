import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { AuthProvider } from './components/AuthProvider';
import { RuntimeConfigProvider } from './components/RuntimeConfigProvider';
import { AuthGate } from './components/auth';
import './i18n'; // Import i18n configuration
import './styles/globals.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <AuthProvider>
      <RuntimeConfigProvider>
        <AuthGate>
          <App />
        </AuthGate>
      </RuntimeConfigProvider>
    </AuthProvider>
  </React.StrictMode>
);
