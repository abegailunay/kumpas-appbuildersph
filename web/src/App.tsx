import { Capture } from './pages/Capture'
import { Debug } from './pages/Debug'

const ROUTES: Record<string, { label: string; page: () => React.JSX.Element }> = {
  '/debug': { label: 'Debug', page: Debug },
  '/capture': { label: 'Capture', page: Capture },
}

function Home() {
  return (
    <div className="page">
      <h1>Kumpas</h1>
      <p>Learn Filipino Sign Language with on-device AI. The Teach mode arrives in the next step.</p>
      <ul>
        <li><a href="/debug">Debug</a>: record a sign and see the model's top-3 and timings.</li>
        <li><a href="/capture">Capture</a>: record your own attempts for fine-tuning.</li>
      </ul>
    </div>
  )
}

export default function App() {
  const Page = ROUTES[location.pathname]?.page ?? Home
  return (
    <>
      <nav className="nav">
        <a href="/" className="brand">Kumpas</a>
        {Object.entries(ROUTES).map(([path, r]) => (
          <a key={path} href={path} className={location.pathname === path ? 'active' : ''}>{r.label}</a>
        ))}
      </nav>
      <Page />
    </>
  )
}
