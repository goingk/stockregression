import type { CSSProperties, ReactNode } from 'react'

type TableViewportProps = {
  children: ReactNode
  className?: string
  owner?: 'local' | 'parent'
  stickyKeyColumn?: number
  maxHeight?: string
  label?: string
}

export default function TableViewport({
  children,
  className,
  owner = 'local',
  stickyKeyColumn,
  maxHeight,
  label,
}: TableViewportProps) {
  const style = maxHeight ? { '--table-viewport-max-height': maxHeight } as CSSProperties : undefined
  return <div
    className={['table-wrap', 'table-viewport', `table-viewport--${owner}`, className].filter(Boolean).join(' ')}
    data-owner={owner}
    data-sticky-key-column={stickyKeyColumn}
    aria-label={label}
    role={label ? 'region' : undefined}
    tabIndex={owner === 'local' ? 0 : undefined}
    style={style}
  >{children}</div>
}
