type Preview = Record<string, unknown> | null
type Estimate = { key: string; status: string; preview: Preview }
type Validated = { status: string; preview: Preview }

/** Both async completion and page validation must describe this exact result. */
export function canSubmitResearchEstimate(estimate: Estimate, key: string, validated: Validated, dateReady: boolean, validation = '') {
  return dateReady && !validation && estimate.key === key && estimate.status === 'ready'
    && validated.status === 'ready' && validated.preview === estimate.preview
    && typeof estimate.preview?.data_version === 'string' && Boolean(estimate.preview.data_version.trim())
}
