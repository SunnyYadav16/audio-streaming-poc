import { useState } from 'react'
import { AudioRecorder } from './components/AudioRecorder'
import { ConversationSession } from './components/ConversationSession'

type AppMode = 'home' | 'solo' | 'conversation';

function App() {
  const [mode, setMode] = useState<AppMode>('home');

  if (mode === 'solo') {
    return <AudioRecorder onBack={() => setMode('home')} />;
  }

  if (mode === 'conversation') {
    return <ConversationSession onBack={() => setMode('home')} />;
  }

  /* ---- Home / mode selector ---- */
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

        .home-root {
          min-height: 100vh; width: 100vw;
          display: flex; flex-direction: column;
          align-items: center; justify-content: center;
          gap: 40px;
          background: #0a0a0f; color: #e4e4e7;
          font-family: 'DM Sans', sans-serif;
        }

        .home-brand {
          text-align: center;
        }
        .home-logo {
          font-family: 'JetBrains Mono', monospace;
          font-size: 16px; font-weight: 700;
          letter-spacing: 4px; text-transform: uppercase;
          color: #6366f1; margin-bottom: 8px;
        }
        .home-tagline {
          font-size: 14px; color: #52525b;
        }

        .home-cards {
          display: flex; gap: 24px; flex-wrap: wrap;
          justify-content: center; padding: 0 24px;
        }

        .home-card {
          width: 300px; padding: 32px 28px;
          background: linear-gradient(165deg, #15151e 0%, #0f0f17 100%);
          border: 1px solid rgba(255,255,255,0.06);
          border-radius: 16px; cursor: pointer;
          transition: all 0.25s; text-align: center;
          display: flex; flex-direction: column;
          align-items: center; gap: 16px;
        }
        .home-card:hover {
          border-color: rgba(99,102,241,0.3);
          transform: translateY(-4px);
          box-shadow: 0 12px 40px rgba(0,0,0,0.3);
        }

        .home-card-icon {
          width: 64px; height: 64px; border-radius: 16px;
          display: flex; align-items: center; justify-content: center;
          font-size: 28px;
        }
        .home-card.solo .home-card-icon {
          background: rgba(99,102,241,0.12);
        }
        .home-card.convo .home-card-icon {
          background: rgba(34,211,238,0.12);
        }

        .home-card-title {
          font-size: 18px; font-weight: 700; color: #fafafa;
        }
        .home-card-desc {
          font-size: 13px; color: #71717a; line-height: 1.6;
        }

        .home-card.solo:hover .home-card-icon {
          background: rgba(99,102,241,0.2);
        }
        .home-card.convo:hover .home-card-icon {
          background: rgba(34,211,238,0.2);
        }
      `}</style>

      <div className="home-root">
        <div className="home-brand">
          <div className="home-logo">CourtAccess AI</div>
          <div className="home-tagline">Real-time speech transcription & translation</div>
        </div>

        <div className="home-cards">
          <div className="home-card solo" onClick={() => setMode('solo')}>
            <div className="home-card-icon">🎙</div>
            <div className="home-card-title">Solo Mode</div>
            <div className="home-card-desc">
              Record, transcribe, and translate your speech.
              Ideal for single-user dictation or testing.
            </div>
          </div>

          <div className="home-card convo" onClick={() => setMode('conversation')}>
            <div className="home-card-icon">💬</div>
            <div className="home-card-title">Conversation</div>
            <div className="home-card-desc">
              Two-person translated conversation.
              Each user speaks their language and hears the translation.
            </div>
          </div>
        </div>
      </div>
    </>
  );
}

export default App
