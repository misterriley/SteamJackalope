import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import ErrorBoundary from './components/ErrorBoundary.tsx'
import { ContextMenuProvider } from './context/ContextMenuContext'
import { UserProvider } from './context/UserContext'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <UserProvider>
        <ContextMenuProvider>
          <App />
        </ContextMenuProvider>
      </UserProvider>
    </ErrorBoundary>
  </StrictMode>,
)
