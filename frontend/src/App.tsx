import './App.css'
import { AudioRecorder } from './components/AudioRecorder'

function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>🎙️ Audio Streaming POC</h1>
        <p>Real-time audio capture and streaming</p>
      </header>

      <main className="app-main">
        <AudioRecorder />
      </main>

      <footer className="app-footer">
        <p>Audio is streamed to the backend and saved as WAV files</p>
      </footer>
    </div>
  )
}

export default App
