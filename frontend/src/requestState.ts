/** Helpers for request races and retryable read caches. */

export interface RequestTicket<Key> {
  key: Key
  revision: number
}

/** Marks the newest request for a view and rejects responses from older ones. */
export class LatestRequest<Key> {
  private revision = 0
  private key: Key | null = null

  begin(key: Key): RequestTicket<Key> {
    this.key = key
    this.revision += 1
    return { key, revision: this.revision }
  }

  isCurrent(ticket: RequestTicket<Key>): boolean {
    return ticket.revision === this.revision && Object.is(ticket.key, this.key)
  }

  /** Invalidate one key's in-flight work without disturbing a newer key. */
  invalidate(key?: Key): void {
    if (arguments.length === 0 || Object.is(this.key, key)) this.revision += 1
  }
}

/**
 * A promise cache for immutable resources. Successful requests stay cached; a
 * rejected request is removed only while it is still the value stored for the
 * key, so a late failure cannot evict a newer retry.
 */
export class PromiseCache<Key, Value> {
  private values = new Map<Key, Promise<Value>>()

  get(key: Key): Promise<Value> | undefined {
    return this.values.get(key)
  }

  getOrCreate(key: Key, create: () => Promise<Value>): Promise<Value> {
    const existing = this.values.get(key)
    if (existing) return existing

    let request: Promise<Value>
    try {
      request = Promise.resolve(create())
    } catch (error) {
      request = Promise.reject(error)
    }
    this.values.set(key, request)
    void request.catch(() => {
      if (this.values.get(key) === request) this.values.delete(key)
    })
    return request
  }

  /** Remove a specific failed request, preserving a newer value for the key. */
  forget(key: Key, expected: Promise<Value>): void {
    if (this.values.get(key) === expected) this.values.delete(key)
  }
}
