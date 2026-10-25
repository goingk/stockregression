import { useEffect, useMemo, useRef, useState } from 'react'
import type { DragEvent, ReactNode } from 'react'
import { GripVertical, RotateCcw } from 'lucide-react'
import Disclosure from './Disclosure'
import { Button } from './components'
import {
  RESEARCH_SECTION_ORDER_KEY,
  defaultResearchSectionOrder,
  dropResearchSection,
  moveResearchSection,
  normalizeResearchSectionOrder,
} from './researchSections'

export type ResearchSection = {
  id: string
  title: string
  summary?: ReactNode
  defaultOpen?: boolean
  content: ReactNode
}

type DropTarget = { id: string; after: boolean }

function readStoredOrder(available: readonly string[]): string[] {
  try {
    const raw = localStorage.getItem(RESEARCH_SECTION_ORDER_KEY)
    return normalizeResearchSectionOrder(raw ? JSON.parse(raw) : null, available)
  } catch {
    return normalizeResearchSectionOrder(defaultResearchSectionOrder, available)
  }
}

function persistOrder(order: readonly string[]) {
  try { localStorage.setItem(RESEARCH_SECTION_ORDER_KEY, JSON.stringify(order)) } catch { /* Layout still works without storage. */ }
}

export default function ResearchSectionStack({ sections, headerAction }: { sections: ResearchSection[]; headerAction?: ReactNode }) {
  const ids = useMemo(() => sections.map(section => section.id), [sections])
  const idsKey = ids.join('|')
  const [order, setOrder] = useState<string[]>(() => readStoredOrder(ids))
  const [dragging, setDragging] = useState<string | null>(null)
  const [dropTarget, setDropTarget] = useState<DropTarget | null>(null)
  const [liveMessage, setLiveMessage] = useState('')
  const draggingRef = useRef<string | null>(null)

  useEffect(() => {
    setOrder(current => {
      const next = normalizeResearchSectionOrder(current, ids)
      return next.join('|') === current.join('|') ? current : next
    })
  }, [idsKey])

  useEffect(() => { persistOrder(order) }, [order])

  const byId = useMemo(() => new Map(sections.map(section => [section.id, section])), [sections])
  const ordered = order.map(id => byId.get(id)).filter((section): section is ResearchSection => Boolean(section))

  const updateOrder = (next: string[], message: string) => {
    if (next.join('|') === order.join('|')) return
    setOrder(next)
    setLiveMessage(message)
  }

  const moveBy = (id: string, delta: number) => {
    const next = moveResearchSection(order, id, delta, ids)
    const nextIndex = next.indexOf(id)
    const section = byId.get(id)
    updateOrder(next, section ? `${section.title} 已移动到第 ${nextIndex + 1} 位` : '分区顺序已更新')
  }

  const drop = (source: string, target: string, after: boolean) => {
    const next = dropResearchSection(order, source, target, after, ids)
    const section = byId.get(source)
    const targetSection = byId.get(target)
    updateOrder(next, section && targetSection ? `${section.title} 已移动到“${targetSection.title}”${after ? '之后' : '之前'}` : '分区顺序已更新')
  }

  const scrollNearEdge = (clientY: number) => {
    const edge = 90
    if (clientY < edge) window.scrollBy({ top: -14, behavior: 'auto' })
    else if (clientY > window.innerHeight - edge) window.scrollBy({ top: 14, behavior: 'auto' })
  }

  const startDrag = (event: DragEvent<HTMLElement>, id: string) => {
    event.stopPropagation()
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/plain', id)
    const section = event.currentTarget.closest<HTMLElement>('[data-research-section]')
    if (section) event.dataTransfer.setDragImage(section, 18, 18)
    draggingRef.current = id
    setDragging(id)
    setDropTarget(null)
  }

  const endDrag = () => {
    draggingRef.current = null
    setDragging(null)
    setDropTarget(null)
  }

  const dragOver = (event: DragEvent<HTMLDivElement>, id: string) => {
    if (!draggingRef.current) return
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
    const rect = event.currentTarget.getBoundingClientRect()
    setDropTarget({ id, after: event.clientY > rect.top + rect.height / 2 })
    scrollNearEdge(event.clientY)
  }

  const resetOrder = () => {
    const next = normalizeResearchSectionOrder(defaultResearchSectionOrder, ids)
    setOrder(next)
    setLiveMessage('已恢复默认分区顺序')
  }

  return <div className='research-section-stack'>
    <div className='research-section-stack-head'>
      <div><strong>研究配置</strong><span>拖动分区左侧把手调整顺序；聚焦把手后也可用上下方向键移动。</span></div>
      <div className='research-section-stack-actions'>
        {headerAction}
        <Button variant='quiet' onClick={resetOrder}><RotateCcw size={14} />恢复默认顺序</Button>
      </div>
    </div>
    <div className='research-section-list'>
      {ordered.map(section => {
        const indicator = dropTarget?.id === section.id ? dropTarget.after ? 'drop-after' : 'drop-before' : ''
        return <div
          key={section.id}
          data-research-section={section.id}
          className={`research-section ${dragging === section.id ? 'dragging' : ''} ${indicator}`.trim()}
          onDragOver={event => dragOver(event, section.id)}
          onDragLeave={event => {
            if (event.currentTarget.contains(event.relatedTarget as Node | null)) return
            setDropTarget(current => current?.id === section.id ? null : current)
          }}
          onDrop={event => {
            event.preventDefault()
            const source = event.dataTransfer.getData('text/plain') || draggingRef.current
            const rect = event.currentTarget.getBoundingClientRect()
            if (source && source !== section.id) drop(source, section.id, event.clientY > rect.top + rect.height / 2)
            endDrag()
          }}
        >
          <Disclosure
            id={section.id}
            title={section.title}
            summary={section.summary}
            defaultOpen={section.defaultOpen}
            leading={<button
              type='button'
              className='research-section-grip'
              draggable
              aria-label={`拖动调整“${section.title}”顺序`}
              title='拖动调整顺序；聚焦后可用上下方向键移动'
              onClick={event => { event.preventDefault(); event.stopPropagation() }}
              onPointerDown={event => event.stopPropagation()}
              onDragStart={event => startDrag(event, section.id)}
              onDragEnd={endDrag}
              onKeyDown={event => {
                if (event.key !== 'ArrowUp' && event.key !== 'ArrowDown') return
                event.preventDefault()
                event.stopPropagation()
                moveBy(section.id, event.key === 'ArrowUp' ? -1 : 1)
              }}
            ><GripVertical size={15} /></button>}
          >{section.content}</Disclosure>
        </div>
      })}
    </div>
    <span className='sr-only' role='status' aria-live='polite'>{liveMessage}</span>
  </div>
}
