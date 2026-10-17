import { useState } from 'react'
import type { ReactNode } from 'react'
import { ChevronRight } from 'lucide-react'

export function usePersistentFlag(key: string, defaultValue = false) {
  const [value, setValue] = useState(() => {
    try {
      const saved = localStorage.getItem('stockregression-ui:' + key)
      return saved == null ? defaultValue : saved === 'true'
    } catch { return defaultValue }
  })
  const update = (next: boolean) => {
    setValue(next)
    try { localStorage.setItem('stockregression-ui:' + key, String(next)) } catch { /* Folding still works without local storage. */ }
  }
  return [value, update] as const
}

export default function Disclosure({ id, title, summary, children, defaultOpen = false, className = '' }: {
  id: string; title: string; summary?: ReactNode; children: ReactNode; defaultOpen?: boolean; className?: string
}) {
  const [open, setOpen] = usePersistentFlag(id, defaultOpen)
  return <details className={`disclosure ${className}`} open={open} data-disclosure={id} onToggle={event => {
    if (event.currentTarget.open !== open) setOpen(event.currentTarget.open)
  }}>
    <summary><ChevronRight size={15} /><span>{title}</span>{summary != null && <small>{summary}</small>}</summary>
    <div className="disclosure-body">{children}</div>
  </details>
}
